import numpy as np
import torch
from openfgl.flcore.base import BaseClient # Base client class from OpenFGL
import random
from torch_geometric.loader import DataLoader # PyTorch Geometric data loader for graph batches
import torch.nn as nn # PyTorch neural network modules
import copy # For deep copying model states
import torch.nn.functional as F  # PyTorch functional API for operations like normalization
from torch_geometric.nn import global_mean_pool # Graph pooling function

class GFedALAClient(BaseClient):
    """Client-side implementation of G-FedALA (Graph-Aware Federated Adaptive Local Aggregation). Key responsibilities:
    1. Receive global model from server
    2. Perform local initialization with adaptive weight learning (ALA)
    3. Train model on local data for 1 epoch
    4. Compute graph-level embedding representing client's data distribution
    5. Send updated model and graph embedding back to server"""

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """Initialize G-FedALA client with configuration and ALA hyperparameters.Args:
            args: Configuration containing training hyperparameters
            client_id: Unique identifier for this client
            data: Client's local graph dataset
            data_dir: Directory for data storage
            message_pool: Shared dictionary for client-server communication
            device: torch.device for computations (CPU/GPU)"""
        
        super(GFedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        # ALA adaptation hyperparameters
        self.start_phase = True  # Flag for first round (unused, can be removed)
        self.ala_weights = None  # Adaptive mixing weights w ∈ [0,1] for each head parameter
        self.ala_reset_each_round = False  # If True, reinitialize weights to 1.0 each round (we keep previous)
        self.ala_data_ratio = 0.8  # Use 80% of training graphs for ALA weight learning
        self.ala_eta = 1.0  # Learning rate η for ALA weight updates (gradient descent step size)
        self.max_init_epochs = 20  # Maximum epochs to converge ALA weights in round 1
        self.converge_eps = 1e-3  # Convergence threshold: stop if max|Δw| < eps
        self.converge_patience = 2  # Require convergence for 2 consecutive epochs
        self.device = device  # Store device reference
        self._ala_epoch_stats = None  # Statistics from most recent ALA epoch
        self.last_ala_stats = None  # Statistics to send to server for logging
        # Graph embedding for G-FedALA
        self.graph_embedding = None  # Client's graph-level embedding (computed after training)

    def execute(self):
        """Main client execution for one federated round. Steps:
        1. Get global model from server
        2. Perform local initialization (ALA adaptation)
        3. Train on local data for 1 epoch
        4. Compute graph embedding representing data distribution"""

        global_state = self.message_pool["server"]["state"]  # Receive global model parameters
        self.local_initialization(global_state)  # ALA: adaptive mixing of global/local models
        self.task.train()  # Standard local training (1 epoch on local graphs)

        # Compute graph-level embedding after training (when model best represents local data)
        self.graph_embedding = self._compute_graph_embedding()

    def send_message(self):
        """Send updated model, sample count, and graph embedding to server. Message contents:
        - num_samples: Number of training samples (for sample-weighted aggregation)
        - state: Model parameters after local training
        - ala: ALA statistics for telemetry/logging
        - graph_emb: Graph-level embedding for structure-aware aggregation"""

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,  # Dataset size for weighted averaging
            "state": copy.deepcopy(self.task.model.state_dict()),  # Deep copy prevents shared refs
            "ala": copy.deepcopy(self.last_ala_stats),  # ALA diagnostics for W&B logging
            "graph_emb": self.graph_embedding.cpu()  # Move to CPU for transmission
        }

    ### Below are helper methods

    def _compute_graph_embedding(self):
        """Compute client-level graph embedding by averaging normalized graph representations.
        Process:
        1. For each training graph: Get pooled representation from GNN
        2. Mean-pool to ensure proper averaging (Handle variable node counts)
        3. L2-normalize each graph embedding
        4. Average across all training graphs
        5. Final L2-normalization for scale invariance
        Returns:
            torch.Tensor: [embedding_dim] client-level graph embedding"""
        
        self.task.model.eval() # Evaluation mode
        sum_emb = None  # Accumulator for summing graph embeddings
        n_graphs_total = 0  # Counter for total number of graphs processed
        
        with torch.no_grad():  # No gradients needed (feature extraction only)
            for batch in self.task.train_dataloader:  # Iterate over training graph batches
                batch = batch.to(self.device)  # Move batch to GPU/CPU
                
                # Forward pass: Get graph-level pooled embeddings (Ignore classification logits)
                glo_pooled_graph_emb, _ = self.task.model(batch)  # [num_graphs_in_batch, embedding_dim]
                
                # Mean-pool node embeddings accounting for varying graph sizes
                # bincount counts nodes per graph; clamp_min prevents division by zero
                num_nodes = torch.bincount(batch.batch, minlength=batch.num_graphs).to(glo_pooled_graph_emb.dtype)  # [num_graphs_in_batch]
                num_nodes = num_nodes.clamp_min(1.0).unsqueeze(1)  # [num_graphs_in_batch, 1]
                mean_pooled_graph_emb = glo_pooled_graph_emb / num_nodes  # [num_graphs_in_batch, embedding_dim]
                
                # L2-normalize each graph embedding for scale-invariant comparison
                mean_pooled_graph_emb = F.normalize(mean_pooled_graph_emb, dim=1)  # [num_graphs_in_batch, embedding_dim]
                
                # Accumulate sum of embeddings across batches
                batch_sum = mean_pooled_graph_emb.sum(dim=0)  # [embedding_dim]
                sum_emb = batch_sum if sum_emb is None else (sum_emb + batch_sum)  # Running sum
                
                n_graphs_total += int(batch.num_graphs)  # Update graph counter

        # Average embeddings across all training graphs
        client_emb = sum_emb / float(n_graphs_total)  # [embedding_dim]
        # Final L2-normalization for consistent scale across clients
        client_emb = F.normalize(client_emb, dim=0)  # [embedding_dim]
        return client_emb

    @staticmethod
    def _head_params(model):
        """Extract task-specific head layer parameters from GIN model. Head consists of:
        - lin1: First linear layer
        - batch_norm1: Batch normalization (Learnable affine parameters)
        - lin2: Final classification layer
        Note: For other GNN architectures, modify this function accordingly.
        Returns:
            list: All trainable parameters in the head layers"""
        
        return (
            list(model.lin1.parameters()) +  # First linear transformation
            list(model.batch_norm1.parameters()) +  # Batch norm scale/shift
            list(model.lin2.parameters())  # Output layer
        )
            
    def _w_stats(self):
        """Compute diagnostic statistics for current ALA mixing weights.
        Tracks:
        - w_mean: Average weight value
        - w_min/w_max: Range of weights
        - w_frac_0: Fraction of weights ≈ 0 (Local model dominance)
        - w_frac_1: Fraction of weights ≈ 1 (Global model dominance)
        Returns:
            dict: Statistics for telemetry / W&B logging
        """

        if self.ala_weights is None or len(self.ala_weights) == 0:
            return {}  # No weights initialized yet
        
        with torch.no_grad():
            flat = torch.cat([w.detach().flatten() for w in self.ala_weights])  # Flatten all weights
            eps = 1e-6  # Threshold for "close to 0 or 1"
            return {
                "w_mean": float(flat.mean().item()),
                "w_min": float(flat.min().item()),
                "w_max": float(flat.max().item()),
                "w_frac_0": float((flat <= eps).float().mean().item()),  # Fraction ≈ 0
                "w_frac_1": float((flat >= 1.0 - eps).float().mean().item()),  # Fraction ≈ 1
            }

    def _build_ala_loader(self):
        """Create DataLoader for ALA adaptation using subset of training graphs.
        Process:
        1. Get training graph indices from mask
        2. Sample ala_data_ratio fraction (default 80%)
        3. Create DataLoader with sampled graphs
        Returns:
            DataLoader or None: Loader for ALA adaptation, or None if no training data"""
        
        # Get indices of training graphs from boolean mask
        train_idx = self.task.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.detach().cpu().tolist()  # Convert to Python list

        if len(train_idx) == 0:
            return None  # No training data available

        # Sample subset of training graphs for ALA
        m = max(1, int(self.ala_data_ratio * len(train_idx)))  # At least 1 graph
        sampled_idx = random.sample(train_idx, m) if m < len(train_idx) else train_idx

        # Gather sampled graph objects
        sampled_graphs = [self.task.data[i] for i in sampled_idx]

        # Create DataLoader (Small batch size for stable ALA convergence)
        return DataLoader(sampled_graphs, batch_size=self.args.batch_size, shuffle=False)

    def _apply_head_mix_(self, params_t, params_l, params_g):
        """Apply ALA mixing formula to head parameters (in-place update). Formula (Eq. 4 in FedALA paper): θ_t = θ_l + (θ_g - θ_l) * w
        Where:
        - θ_t: target (mixed) parameters
        - θ_l: local model parameters
        - θ_g: global model parameters
        - w: adaptive weights ∈ [0,1]
        When w=0: θ_t = θ_l (Fully local)
        When w=1: θ_t = θ_g (Fully global)
        Args:
            params_t: Target parameters to update
            params_l: Local model parameters
            params_g: Global model parameters
        """
        with torch.no_grad():
            for pt, pl, pg, w in zip(params_t, params_l, params_g, self.ala_weights):
                # Element-wise mixing: Interpolate between local and global
                pt.data.copy_(pl.data + (pg.data - pl.data) * w)

    def _ala_one_epoch_update(self, temp_model, params_t, params_l, params_g, ala_loader):
        """Perform one epoch of ALA weight learning using gradient descent.
        Algorithm (based on Eq. 5 in FedALA paper):
        1. Mix head parameters: θ_t = θ_l + (θ_g - θ_l) * w
        2. Forward pass and compute loss
        3. Backward pass to get ∇θ_t
        4. Update weights: w <- clip(w - η * (∇θ_t ⊙ (θ_g - θ_l)), 0, 1)
        The gradient ∇θ_t ⊙ (θ_g - θ_l) indicates how much each weight should change
        to minimize loss on local data.
        Args:
            temp_model: Temporary model for computing gradients
            params_t: Temporary model head parameters
            params_l: Local model head parameters
            params_g: Global model head parameters
            ala_loader: DataLoader with ALA adaptation graphs 
        Returns:
            float: Average loss over the epoch"""
        
        temp_model.eval()  # Eval mode (No batch norm updates, no dropout randomness)
        total_loss = 0.0
        n_batches = 0

        loss_start = None  # Loss at first batch
        loss_end = None  # Loss at last batch
        grad_norm_acc = 0.0  # Accumulated gradient norms
        delta_norm_acc = 0.0  # Accumulated (θ_g - θ_l) norms
        n_terms = 0  # Number of parameter tensors

        for batch in ala_loader:
            batch = batch.to(self.device)

            # 1-Apply current mixing weights to head parameters
            self._apply_head_mix_(params_t, params_l, params_g)

            # 2-Forward pass and compute loss
            temp_model.zero_grad(set_to_none=True)  # Clear gradients
            _, logits = temp_model(batch)  # Get predictions
            loss = self.task.default_loss_fn(logits, batch.y)  # Classification loss

            # 3-Backward pass to compute ∇θ_t
            loss.backward()

            # Track loss progression
            if loss_start is None:
                loss_start = float(loss.item())
            loss_end = float(loss.item())

            # 4-Update ALA weights using gradient information
            with torch.no_grad():
                for pt, pl, pg, w in zip(params_t, params_l, params_g, self.ala_weights):
                    if pt.grad is None:  # Skip if no gradient
                        continue
                    
                    # Accumulate telemetry
                    grad_norm_acc += float(pt.grad.detach().norm().item())
                    delta_norm_acc += float((pg.data - pl.data).detach().norm().item())
                    n_terms += 1

                    # Compute weight gradient: ∇w = ∇θ_t ⊙ (θ_g - θ_l)
                    grad_w = pt.grad * (pg.data - pl.data)
                    
                    # Normalize gradient to prevent large updates
                    grad_norm = grad_w.norm()
                    if grad_norm > 1e-6:  # Avoid division by zero
                        grad_w = grad_w / grad_norm
                    
                    # Gradient descent: w <- w - η * ∇w
                    w.data.sub_(self.ala_eta * grad_w)
                    
                    # Clip to [0, 1] range (w represents mixing coefficient)
                    w.data.clamp_(0, 1)

            total_loss += float(loss.item())
            n_batches += 1

        # Compute average loss and collect statistics
        avg_loss = total_loss / max(1, n_batches)
        stats = self._w_stats()  # Get weight statistics
        stats.update({
            "ala_loss": float(avg_loss),
            "ala_loss_start": float(loss_start) if loss_start is not None else float("nan"),
            "ala_loss_end": float(loss_end) if loss_end is not None else float("nan"),
            "grad_norm_head": float(grad_norm_acc / max(1, n_terms)),  # Avg gradient norm
            "delta_norm": float(delta_norm_acc / max(1, n_terms)),  # Avg (θ_g - θ_l) norm
            "ala_batches": float(n_batches),
        })

        self._ala_epoch_stats = stats  # Store for logging
        return avg_loss
    
    def local_initialization(self, global_state_dict):
        """Perform local initialization with ALA before standard training.
        Three-phase strategy:
        - Round 0 (t=1): Skip ALA, just load global model
        - Round 1 (t=2): Learn ALA weights until convergence (Up to max_init_epochs)
        - Round 2+ (t≥3): One epoch of ALA weight refinement
        Process:
        1. Create copies of local, global, and temp models
        2. Extract head parameters from each
        3. Initialize or reuse ALA weights
        4. Learn weights via gradient descent on temp model
        5. Apply learned weights to mix local and global heads
        6. Load backbone from global, head from mixed version
        Args:
            global_state_dict: Global model parameters from server
        """
        round_id = int(self.message_pool.get("round", 0))  # Current communication round
        
        # Create model copies for ALA computation
        local_model = copy.deepcopy(self.task.model).to(self.device)  # Previous round's model
        global_model = copy.deepcopy(self.task.model).to(self.device)  # Current global model
        global_model.load_state_dict(global_state_dict, strict=True)

        # Round 0: No ALA needed (Local and global are identical)
        if round_id == 0:
            self.last_ala_stats = None
            self._ala_epoch_stats = None
            self.task.model.load_state_dict(global_state_dict, strict=True)
            return

        # Prepare temporary model for ALA weight learning
        temp_model = copy.deepcopy(global_model).to(self.device)
        
        # Extract head parameters from all three models
        params_l = self._head_params(local_model)  # Local head (From previous round)
        params_g = self._head_params(global_model)  # Global head (Current)
        params_t = self._head_params(temp_model)  # Temp head (To be mixed and updated)
        
        # Freeze backbone layers (We only adapt the head)
        for p in temp_model.convs.parameters():
            p.requires_grad = False
        for p in temp_model.batch_norms.parameters():
            p.requires_grad = False

        # Ensure head parameters are trainable
        for p in params_t:
            p.requires_grad = True
        
        # Create DataLoader with subset of training graphs for ALA
        ala_loader = self._build_ala_loader()

        # Initialize ALA weights (Ones = equal mixing of local and global)
        if self.ala_reset_each_round or (self.ala_weights is None):
            self.ala_weights = [torch.ones_like(p.data, device=self.device) for p in params_t]

        # Round 1: Learn weights until convergence
        if round_id == 1:
            stable_cnt = 0  # Counter for consecutive epochs meeting convergence criterion
            
            for _ in range(self.max_init_epochs):  # Up to 20 epochs
                w_prev = [w.detach().clone() for w in self.ala_weights]  # Save previous weights
                _ = self._ala_one_epoch_update(temp_model, params_t, params_l, params_g, ala_loader)

                # Check convergence: Max |Δw| across all weights
                with torch.no_grad():
                    max_delta = 0.0
                    for w, wp in zip(self.ala_weights, w_prev):
                        max_delta = max(max_delta, float((w - wp).abs().max().item()))

                # If weights changed less than threshold
                if max_delta < self.converge_eps:
                    stable_cnt += 1
                    if stable_cnt >= self.converge_patience:  # Converged for 2 consecutive epochs
                        break
                else:
                    stable_cnt = 0  # Reset counter if not converged
        else:
            # Round 2+: Just one epoch of weight refinement
            _ = self._ala_one_epoch_update(temp_model, params_t, params_l, params_g, ala_loader)

        # Save statistics for W&B logging
        self.last_ala_stats = copy.deepcopy(self._ala_epoch_stats)

        # Apply learned weights to create final personalized model
        # Backbone: Use global (similarity-weighted on server)
        # Head: Mix local and global using learned weights
        self.task.model.load_state_dict(global_state_dict, strict=True)  # Start with global
        
        # Apply ALA mixing to head parameters
        model_params_t = self._head_params(self.task.model)
        self._apply_head_mix_(model_params_t, params_l, params_g)  # θ_head = θ_l + (θ_g - θ_l) * w
        
        # Unfreeze backbone for subsequent local training
        for p in self.task.model.convs.parameters():
            p.requires_grad = True
        for p in self.task.model.batch_norms.parameters():
            p.requires_grad = True