import torch
import torch.nn as nn
import copy
import random
import numpy as np
from openfgl.flcore.base import BaseClient
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

class FedALAClient(BaseClient):
    # Device is where the tensor lives, keeps everything (model, gradients, tensors) on the same machine or GPU.
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        # Calling base client constructor
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

        # FedALA hyperparameters
        self.layer_idx = 1        # How many top layers of the model to adapt
        self.rand_percent = 50    # % of training nodes to sample for weight learning
        self.eta = 0.01           # Weight learning rate (small for stability)
        self.weights = None       # Stores the adaptive interpolation weights (layer-wise)
        self.start_phase = True   # Only perform full adaptation in first round

        print("FedALAClient initalized.", flush=True)

    def execute(self):
        """Perform local training and FedALA adaptive aggregation."""
        print("FedALAClient.execute() called.", flush=True)

        # Synchronize with global model from the server (server previously put a state_dict in message_pool["server"]["weight"])
        server_entry = self.message_pool.get("server")
        if server_entry and "weight" in server_entry:
            self.task.model.load_state_dict(server_entry["weight"])

        # Train local model normally on client data
        self.task.train()

        # Perform FedALA adaptive aggregation on top layers
        self.adaptive_local_aggregation()

    # Helper function for forward pass: Avoids code breaking depending on the architecture (GIN, GCN, GAT...) and implementation differences
    def forward_helper(self, model, data):
        """Forward pass wrapper for PyTorch Geometric (PyG) models.
        Different GNN implementations use different forward() signatures:
            1) forward(self, data)                     # Standard PyG style
            2) forward(self, x, edge_index)            # Explicit inputs
            3) forward(self, x, edge_index, batch)     # Explicit + batching """
        
        # Move tensors inside Data object to the correct device
        if hasattr(data, "x"):
            data.x = data.x.to(self.device)
        if hasattr(data, "edge_index"):
            data.edge_index = data.edge_index.to(self.device)
        if hasattr(data, "y"):
            data.y = data.y.to(self.device)
        if hasattr(data, "batch"):
            data.batch = data.batch.to(self.device)

        try:
            # Try the standard PyG forward
            return model(data)
        except Exception:
            # If standard one failed, try the explicit GNN signature which is model(x, edge_index, batch)
            try:
                return model(data.x, data.edge_index, getattr(data, "batch", None)) # Safe default if batch doesn't exist
            except Exception as e:
                # Raise original error so debugging is easier
                raise e

    def adaptive_local_aggregation(self):
        """Learn per-layer adaptive interpolation weights between global and local parameters.
        Only top layers (self.layer_idx) are adapted; lower layers are preserved."""
        print("Starting adaptive local aggregation...", flush=True)

        # Copies of global and local models for safe computation
        global_model = copy.deepcopy(self.task.model)
        local_model = copy.deepcopy(self.task.model)

        # Get list of training nodes
        # train_mask is a boolean mask: true means this node is for training so extract training node indices from the mask
        try:
            train_idx = self.task.train_mask.nonzero(as_tuple=False).squeeze().tolist()
        except Exception:
            # If train_mask does not exist or fails (some datasets may not have it) use all nodes as training nodes
            if hasattr(self.task.data, "x"):  # Check if node features exist
                # self.task.data.x is the feature matrix and size(0) = number of nodes
                # Use all nodes as training nodes
                train_idx = list(range(self.task.data.x.size(0)))
            else:
                # If no node features exist, we cannot proceed so exit
                return

        # Make sure train_idx is a list, even if it contains only one node
        if not isinstance(train_idx, list):
            train_idx = [train_idx]

        # If the list of training nodes is empty, there is nothing to train on so exit
        if len(train_idx) == 0:
            return

        # Sample a random subset of training nodes for adaptation
        rand_num = max(1, int(self.rand_percent / 100 * len(train_idx)))
        rand_nodes = random.sample(train_idx, rand_num)

        # Build induced subgraph for sampled nodes: Get the full edge list of the graph (all connections between nodes)
        edge_index_full = getattr(self.task.data, "edge_index", None)
        if edge_index_full is None:
            # If the graph has no edges, we cannot create a subgraph so exit
            return

        # Determine the total number of nodes in the graph, this is needed for subgraph construction
        if hasattr(self.task.data, "num_nodes") and self.task.data.num_nodes is not None:
            # If the dataset explicitly provides num_nodes, use it
            num_nodes = self.task.data.num_nodes
        else:
            try:
                # Otherwise, infer the number of nodes from edge_index:
                # The largest node index + 1 gives the total number of nodes
                num_nodes = int(edge_index_full.max()) + 1
            except Exception:
                # Fallback to using the number of node features
                if hasattr(self.task.data, "x"):
                    # Number of nodes = number of rows in feature matrix
                    num_nodes = self.task.data.x.size(0)
                else:
                    # If all else fails, set num_nodes to None
                    num_nodes = None

        # Call subgraph safely to get edges and mapping for sampled nodes (if num_nodes is None, omit argument) 
        if num_nodes is None:
            sub_edge_index, mapping = subgraph(rand_nodes, edge_index_full, relabel_nodes=True)
        else:
            sub_edge_index, mapping = subgraph(rand_nodes, edge_index_full, relabel_nodes=True, num_nodes=num_nodes)

        # Gather features (x) and labels (y) for the sampled nodes
        idx_tensor = torch.tensor(rand_nodes, dtype=torch.long) # Convert list of indices to tensor
        sub_x = self.task.data.x[idx_tensor].to(self.device) # Select features for sampled nodes and move to GPU/CPU
        sub_y = self.task.data.y[idx_tensor].to(self.device) # Select labels for sampled nodes and move to device

        # Build a PyG Data object for GNN forward pass
        sub_data = Data(x=sub_x, edge_index=sub_edge_index.to(self.device), y=sub_y)

        # Some GIN implementations expect a batch vector
        # If batch info does not exist, create a default batch of zeros (single graph)
        if not hasattr(sub_data, "batch"):
            sub_data.batch = torch.zeros(sub_data.x.size(0), dtype=torch.long).to(self.device)

        # Get model parameters
        params_g = list(global_model.parameters()) # Parameters from the global (server) model
        params_l = list(local_model.parameters()) # Parameters from the local (client) model

        # Skip adaptation if this is the first round (global and local models are identical)
        if len(params_g) == 0 or torch.allclose(params_g[0].data, params_l[0].data):
            return

        # Preserve lower layers, adapt only top layers
        if self.layer_idx < len(params_l):
            # Loop over lower layers and copy weights from global model to local model
            for p, pg in zip(params_l[:-self.layer_idx], params_g[:-self.layer_idx]):
                p.data = pg.data.clone()
        else:
            # If layer_idx >= total number of layers, do not preserve any layers (i.e., all layers are considered "top" layers for adaptation)
            pass

        # Prepare top layers for adaptation
        temp_model = copy.deepcopy(local_model) # Temporary model for gradient updates
        params_t = list(temp_model.parameters()) # All parameters of the temporary model
        params_p = params_l[-self.layer_idx:] # Top layers of local model
        params_gp = params_g[-self.layer_idx:] # Top layers of global model
        params_tp = params_t[-self.layer_idx:] # Top layers of temp model

        # Freeze lower layers of temporary model so only top layers get gradients
        for p in params_t[:-self.layer_idx]:
            p.requires_grad = False

        # Initialize adaptive weights if not already initialized
        # These weights control interpolation between local and global top layers
        if self.weights is None:
            self.weights = [torch.ones_like(p.data, device=self.device, dtype=p.data.dtype) for p in params_p] # Start with all ones (full influence from global model)

        # Track the loss values over iterations for stopping criteria
        losses = []
        max_iters = 20 # Safety cap to prevent infinite loops
        it = 0 # Iteration counter

        # Track consecutive loss increases to use in stopping rule
        consecutive_increase = 0
        last_loss = None

        # Start the adaptive weight learning loop
        while True:
            it += 1 # Increment iteration count

            # We manually update the adaptive weights
            # Before computing new gradients, we need to clear any existing gradients in top layers
            for pt in params_tp:
                if pt.grad is not None:
                    pt.grad.detach_() # Detach the gradient from computation graph to avoid accumulation
                    pt.grad.zero_() # Zero out the gradient tensor

            # Forward pass through the model
            # Use the forward pass helper function above to handle different GNN forward signatures
            # temp_model: the temporary model copy used for top-layer adaptation
            # sub_data: the sampled subgraph Data object
            try:
                output = self.forward_helper(temp_model, sub_data) # Returns predicted outputs
            except Exception:
                # If the forward pass fails (e.g., incompatible dataset or model) then stop
                break

            # Compute task-specific loss
            try:
                # self.task.loss could be  a cross-entropy loss or MSE depending on the task
                # output: predictions from the model for the sampled subgraph
                # sub_data.y: ground-truth labels for the sampled nodes
                loss_value = self.task.loss(output, sub_data.y)
            except Exception:
                # If self.task.loss is not compatible (e.g., shape mismatch), try simple cross_entropy
                loss_value = nn.functional.cross_entropy(output, sub_data.y)

            # Backpropagation to compute gradients
            # temp_model's top layer parameters (params_tp) will have their gradients stored in pt.grad
            # These gradients are used later to update the adaptive interpolation weights "w"
            loss_value.backward()

            # The goal here is to learn per-parameter interpolation weights "w" which control how much the top layer parameters should move
            # from the local model (p) toward the global model (pg)
            for pt, p, pg, w in zip(params_tp, params_p, params_gp, self.weights):
                if pt.grad is None: # Skip if no gradient 
                    continue

                # Ensure all tensors are on the same device and have the same dtype as the adaptive weight
                target_device = w.device
                target_dtype = w.dtype

                # Gradient for the top-layer parameter
                grad = pt.grad.to(device=target_device, dtype=target_dtype)
                # Difference between global and local top-layer parameters
                diff = (pg.data - p.data).to(device=target_device, dtype=target_dtype)

                # Update the adaptive weight elementwise with FedALA rule: w = clamp(w - eta * (grad * diff), 0, 1)
                # If gradient points in the same direction as diff, reduce weight (less global influence)
                # If gradient points opposite to diff, increase weight (more global influence)
                # Clamp ensures weights stay in [0,1] to prevent unstable updates
                w.data = torch.clamp(w.data - self.eta * (grad * diff), 0.0, 1.0)

            # Interpolate the top layers of the temporary model between local and global
            for pt, p, pg, w in zip(params_tp, params_p, params_gp, self.weights):
                # Ensure dtype/device consistency
                target_device = w.device
                target_dtype = w.dtype
                p_data = p.data.to(device=target_device, dtype=target_dtype)
                pg_data = pg.data.to(device=target_device, dtype=target_dtype)

                # Interpolation formula: pt = local + (global - local) * w
                # If w = 1 -> fully global
                # If w = 0 -> fully local
                # Values in between -> Mix based on learned weight
                pt.data = p_data + (pg_data - p_data) * w

            # Keep track of the current loss value so the loop can decide when to stop
            losses.append(float(loss_value.item()))

            # The adaptation loop needs a rule to know when to stop updating adaptive weights
            # FedALA uses three criteria:
            # 1) If the loss increases consecutively N times (here N=3)
            # 2) If recent losses are stable (low standard deviation)
            # 3) If maximum iterations are reached
            if last_loss is None:
                # Initialize last_loss
                last_loss = losses[-1]
                consecutive_increase = 0
            else:
                 # Check if current loss increased compared to the previous iteration
                if losses[-1] > last_loss + 1e-6:  # Slight tolerance to ignore tiny fluctuations
                    consecutive_increase += 1 # Track consecutive increases
                else:
                    consecutive_increase = 0 # Reset counter if loss didn't increase
                last_loss = losses[-1]

            # Check if the last 3 losses are very stable (small standard deviation)
            recent_std_ok = (len(losses) > 3 and np.std(losses[-3:]) < 1e-4)

            # Stop adaptation if any of these are true:
            # Loss increased 3 times consecutively
            # Recent losses are stable
            # Reached maximum number of iterations
            if (consecutive_increase >= 3) or recent_std_ok or (it >= max_iters and len(losses) > 0):
                break

        # Mark that first-phase adaptation is done (ensures full adaptation only happens in the first round)
        self.start_phase = False

        # After learning the interpolation weights and updating the temp_model,
        # we copy the top-layer parameters from temp_model back to the local_model
        for p, pt in zip(params_p, params_tp):
            p.data = pt.data.clone() # Ensures actual top-layer weights are updated

        # Write adapted local model back to the actual task model to send back to the server
        self.task.model.load_state_dict(local_model.state_dict())

    def send_message(self):
        """Send the locally trained/adapted model back to the server."""
        print("FedALAClient.send_message() is called.", flush=True)

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": getattr(self.task, "num_samples", 1),
            "weight": copy.deepcopy(self.task.model.state_dict())
        }
