import copy  # For deep copying model state dictionaries
import torch
from openfgl.flcore.base import BaseServer  # Base server class from OpenFGL
from collections import OrderedDict
import torch.nn.functional as F  # PyTorch functional API

# Helper functions
def _is_bn_buffer(k: str) -> bool:
    """Check if a parameter key corresponds to a BatchNorm buffer (running stats)."""
    return ("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k)

def _is_backbone(k: str) -> bool:
    """Check if a parameter key belongs to the GNN backbone (convolution layers)."""
    return k.startswith("convs.") or k.startswith("batch_norms.")

def _is_neck_head(k: str) -> bool:
    """Check if a parameter key belongs to neck/head layers (task-specific layers)."""
    return k.startswith("lin1.") or k.startswith("batch_norm1.") or k.startswith("lin2.")

class GFedALAServer(BaseServer):
    """Server-side implementation of G-FedALA (Graph-Aware Federated Adaptive Local Aggregation).
    Key responsibilities:
    1. Collect model updates and graph embeddings from clients
    2. Compute parameter-space and graph-space distances
    3. Calculate adaptive aggregation weights using softmax over negative distances
    4. Aggregate models with split strategy (backbone: similarity-weighted, head: sample-weighted)"""

    def __init__(self, args, global_data, data_dir, message_pool, device):
        """ Initialize G-FedALA server with hyperparameters. Args:
            args: Configuration containing hyperparameters
            global_data: Global validation/test data (if any)
            data_dir: Directory for data storage
            message_pool: Shared dictionary for client-server communication
            device: torch.device for computations (CPU/GPU)"""
        super(GFedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

        # lambda_graph: Controls the contribution of model parameter similarity versus graph-embedding similarity when computing client aggregation weights
        # gala_temperature: Controls how sharply client similarity scores influence aggregation weights
        # gala_warmup_rounds: Number of initial training rounds during which graph-structure similarity is incorporated into aggregation weight computation

        # G-FedALA specific hyperparameters
        self.lambda_graph = args.lambda_graph  # λ: Balance between param (λ) and graph (1-λ) distances
        self.gala_temperature = args.gala_temperature  # τ: Softmax temperature for aggregation weights
        self.gala_warmup_rounds = args.gala_warmup_rounds  # Rounds to use graph+param distances

    def execute(self):
        """Main server aggregation logic executed each round. Process:
        1. Collect client models and graph embeddings
        2. Compute distances (parameter and graph)
        3. Calculate aggregation weights
        4. Aggregate backbone (similarity-weighted) and head (sample-weighted)"""
        with torch.no_grad(): # No gradients needed for aggregation
            sampled = self.message_pool["sampled_clients"] # List of client IDs participating this round
            global_state = self.task.model.state_dict() # Current global model parameters
            
            # Use neck/head parameters for distance computation
            # Since local training is only 1 epoch, backbone changes are minimal.
            # Head parameters are more meaningful because ALA adapts them to local data.
            dist_keys = []
            for k, v in global_state.items():
                if _is_neck_head(k) and torch.is_floating_point(v): # Only floating-point tensors
                    if _is_bn_buffer(k): # Skip BatchNorm running statistics
                        continue
                    dist_keys.append(k)

            if len(dist_keys) == 0: # Safety check
                raise RuntimeError("dist_keys is empty. Check neck/head key filters and model state_dict keys.")

            eps = 1e-8 # Small constant to prevent division by zero
            
            # Collect graph embeddings from all sampled clients
            graph_embs = [
                self.message_pool[f"client_{cid}"]["graph_emb"].to(self.device)
                for cid in sampled
            ]
            
            # Compute global graph prototype (average of client embeddings)
            h_global = torch.stack(graph_embs).mean(dim=0) # [embedding_dim]
            h_global_norm = h_global.norm() + eps # L2 norm for normalization

            round_id = self.message_pool.get("round", 0) # Current communication round

            # Compute param distances (Using neck/head)
            param_dists = []  # Parameter-space distances
            graph_dists = []  # Graph-space distances (only during warm up)
        
            for cid in sampled:
                client_state = self.message_pool[f"client_{cid}"]["state"] # Client's model state

                # Compute layer-wise parameter distance
                per_key = []
                for k in dist_keys:
                    diff = client_state[k].to(self.device) - global_state[k] # Parameter difference
                    global_norm = global_state[k].norm() + eps # Global parameter norm
                    relative_diff = (diff.norm() / global_norm).pow(2) # Normalized squared distance
                    per_key.append(relative_diff)
                param_dist = torch.sqrt(torch.stack(per_key).mean()) # Root mean square distance across layers
                param_dists.append(param_dist)
                
                # Compute graph distance only during warmup phase
                if round_id < self.gala_warmup_rounds:
                    client_emb = self.message_pool[f"client_{cid}"]["graph_emb"].to(self.device)
                    graph_dist = (client_emb - h_global).norm() / (h_global_norm + eps) # Normalized L2 distance
                    graph_dists.append(graph_dist)
            
            # Normalize parameter distances across clients for fair comparison
            param_dists = torch.stack(param_dists) # [num_clients]
            
            p_std = param_dists.std() # Standard deviation
            if p_std > eps: # If there's variation
                param_dists = (param_dists - param_dists.mean()) / p_std # Z-score normalization
            else:
                # All clients have same distance
                param_dists = torch.zeros_like(param_dists)
            
            # Normalize graph distances during warm up
            if round_id < self.gala_warmup_rounds:
                graph_dists = torch.stack(graph_dists) # [num_clients]
                
                # Z-score normalization for graph distances
                g_std = graph_dists.std()
                if g_std > eps:
                    graph_dists = (graph_dists - graph_dists.mean()) / g_std # Z-score normalization
                else:
                    graph_dists = torch.zeros_like(graph_dists)

            # Compute aggregation logits
            logits = []
            for i, cid in enumerate(sampled):
                # Two-phase strategy: Warm up uses graph+param, post-warm up uses only param
                if round_id < self.gala_warmup_rounds:
                    total_dist = (
                        self.lambda_graph * param_dists[i] # Parameter component
                        + (1 - self.lambda_graph) * graph_dists[i] # Graph component
                    )
                else:
                    total_dist = param_dists[i] # Only parameter distance

                logit = -total_dist # Negative distance (higher similarity -> Higher weight)
                logits.append(logit)

            logits = torch.stack(logits) # [num_clients]
            # Apply temperature-scaled softmax to get aggregation weights
            alphas = torch.softmax(logits / self.gala_temperature, dim=0) # [num_clients]

            # Split aggregation strategy: Separate backbone and neck/head parameters
            backbone_keys = [k for k, v in global_state.items()
                             if _is_backbone(k) and torch.is_floating_point(v)]
            neck_head_keys = [k for k, v in global_state.items()
                              if _is_neck_head(k) and torch.is_floating_point(v)]
            
            agg_state = OrderedDict((k, v.clone()) for k, v in global_state.items()) # Initialize with global state
            
            # Backbone aggregation: Similarity-weighted (Uses adaptive alphas)
            for k in backbone_keys:
                agg_state[k].zero_() # Reset to zero before accumulation

            for alpha, cid in zip(alphas, sampled):
                client_state = self.message_pool[f"client_{cid}"]["state"]
                a = float(alpha) # Convert to Python float
                for k in backbone_keys:
                    agg_state[k].add_(client_state[k].to(self.device), alpha=a) # Weighted sum

            # Neck+Head aggregation: Sample-weighted (FedAvg-style)
            ns = torch.tensor(
                [float(self.message_pool[f"client_{cid}"]["num_samples"]) for cid in sampled],
                device=self.device,
                dtype=torch.float32
            ) # Number of samples per client
            ws = ns / (ns.sum() + 1e-8) # Sample-based weights (Normalized to sum to 1)

            for k in neck_head_keys:
                agg_state[k].zero_() # Reset to zero

            for w, cid in zip(ws, sampled):
                client_state = self.message_pool[f"client_{cid}"]["state"]
                ww = float(w)
                for k in neck_head_keys:
                    agg_state[k].add_(client_state[k].to(self.device), alpha=ww) # Weighted sum

            self.task.model.load_state_dict(agg_state, strict=True) # Update global model

    def send_message(self):
        """Send updated global model to clients for next round."""
        self.message_pool["server"] = {
            "state": copy.deepcopy(self.task.model.state_dict())  # Deep copy to prevent shared references
        }