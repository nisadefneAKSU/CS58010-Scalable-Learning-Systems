import numpy as np
import torch
from openfgl.flcore.base import BaseClient  # Base client class from OpenFGL
import random
from torch_geometric.loader import DataLoader  # PyTorch Geometric data loader for graph batches
import torch.nn as nn  # PyTorch neural network modules
import copy  # For deep copying model states

class FedALAClient(BaseClient):
    """Client-side implementation of FedALA (Federated Adaptive Local Aggregation).
    Key responsibilities:
    1. Receive global model from server
    2. Perform local initialization with adaptive weight learning (ALA)
    3. Train model on local data
    4. Send updated model back to server"""

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """Initialize FedALA client with configuration and ALA hyperparameters.
        Args:
            args: Configuration containing training hyperparameters
            client_id: Unique identifier for this client
            data: Client's local graph dataset
            data_dir: Directory for data storage
            message_pool: Shared dictionary for client-server communication
            device: torch.device for computations (CPU/GPU)"""
        
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        # ALA adaptation hyperparameters
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
        
    def execute(self):
        """Main client execution for one federated round.
        Steps:
        1. Get global model from server
        2. Perform local initialization (ALA adaptation)
        3. Train on local data"""
        
        global_state = self.message_pool["server"]["state"]  # Receive global model parameters
        self.local_initialization(global_state)  # ALA: adaptive mixing of global/local models
        self.task.train()  # Standard local training on local graphs


    def send_message(self):
        """Send updated model and sample count to server.
        Message contents:
        - num_samples: Number of training samples (for sample-weighted aggregation)
        - state: Model parameters after local training
        - ala: ALA statistics for telemetry/logging"""
        
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,  # Dataset size for weighted averaging
                "state": copy.deepcopy(self.task.model.state_dict()),  # Deep copy prevents shared refs
                "ala": copy.deepcopy(self.last_ala_stats)  # ALA diagnostics for W&B logging
            }

    ### Below are helper methods

    @staticmethod
    def _head_params(model):
        """Extract task-specific head layer parameters from GIN model.
        Head consists of:
        - lin1: First linear layer
        - batch_norm1: Batch normalization (learnable affine parameters)
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
        - w_frac_0: Fraction of weights ≈ 0 (local model dominance)
        - w_frac_1: Fraction of weights ≈ 1 (global model dominance)
        Returns:
            dict: Statistics for telemetry / W&B logging"""
        
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
        """Build DataLoader for ALA weight learning using subset of training graphs.
        Process:
        1. Get indices of training graphs
        2. Sample subset based on ala_data_ratio (80%)
        3. Create DataLoader for sampled graphs
        Returns:
            DataLoader: Batched loader for ALA optimization"""
        
        train_idx = self.task.train_mask.nonzero(as_tuple=False).view(-1)
        train_idx = train_idx.detach().cpu().tolist()

        # Sample subset of training graphs
        m = max(1, int(self.ala_data_ratio * len(train_idx)))
        sampled_idx = random.sample(train_idx, m) if m < len(train_idx) else train_idx

        # Create graph list from sampled indices
        sampled_graphs = [self.task.data[i] for i in sampled_idx]

        # Return DataLoader (smaller batch_size can be more stable for ALA)
        return DataLoader(sampled_graphs, batch_size=self.args.batch_size, shuffle=False)

    def _apply_head_mix_(self, params_t, params_l, params_g):
        """Apply adaptive layer aggregation (ALA) mixing to head parameters.
        Implements Eq(4): θ_t = θ_l + (θ_g - θ_l) * w
        where w is the learned mixing weight in [0,1].
        Args:
            params_t: Target parameters to overwrite
            params_l: Local model head parameters
            params_g: Global model head parameters"""
        
        with torch.no_grad():
            for pt, pl, pg, w in zip(params_t, params_l, params_g, self.ala_weights):
                pt.data.copy_(pl.data + (pg.data - pl.data) * w)

    def _ala_one_epoch_update(self, temp_model, params_t, params_l, params_g, ala_loader):
        """Perform one epoch of ALA weight learning.
        Implements Eq(5): w ← clip(w - η * (∇θ_t ⊙ (θ_g - θ_l)))
        Process:
        1. Apply current weights to mix head parameters
        2. Forward pass and compute loss
        3. Backward pass to get ∇θ_t
        4. Update weights using gradient information
        5. Clip weights to [0, 1]
        Args:
            temp_model: Temporary model for gradient computation
            params_t: Temp model head parameters
            params_l: Local model head parameters
            params_g: Global model head parameters
            ala_loader: DataLoader for ALA optimization
        Returns:
            float: Average loss over the epoch"""
        
        temp_model.eval()  # Evaluation mode (but we still compute gradients)
        total_loss = 0.0
        n_batches = 0

        # Telemetry variables
        loss_start = None
        loss_end = None
        grad_norm_acc = 0.0
        delta_norm_acc = 0.0
        n_terms = 0

        for batch in ala_loader:
            batch = batch.to(self.device)

            # 1-Apply current weights to reconstruct temp head
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
                    
                    # Gradient descent: w ← w - η * ∇w
                    w.data.sub_(self.ala_eta * grad_w)
                    
                    # Clip to [0, 1] range (w represents mixing coefficient)
                    w.data.clamp_(0, 1)

            total_loss += float(loss.item())
            n_batches += 1

        # Compute average loss and collect statistics
        avg_loss = total_loss / max(1, n_batches)
        stats: dict[str, float] = self._w_stats()  # Get weight statistics
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
        - Round 1 (t=2): Learn ALA weights until convergence (up to max_init_epochs)
        - Round 2+ (t≥3): One epoch of ALA weight refinement
        Process:
        1. Create copies of local, global, and temp models
        2. Extract head parameters from each
        3. Initialize or reuse ALA weights
        4. Learn weights via gradient descent on temp model
        5. Apply learned weights to mix local and global heads
        6. Load backbone from global, head from mixed version
        Args:
            global_state_dict: Global model parameters from server"""
        
        round_id = int(self.message_pool.get("round", 0))  # Current communication round
        
        # Create model copies for ALA computation
        local_model = copy.deepcopy(self.task.model).to(self.device)  # Previous round's model
        global_model = copy.deepcopy(self.task.model).to(self.device)  # Current global model
        global_model.load_state_dict(global_state_dict, strict=True)

        # Round 0: No ALA needed (local and global are identical)
        if round_id == 0:
            self.last_ala_stats = None
            self._ala_epoch_stats = None
            self.task.model.load_state_dict(global_state_dict, strict=True)
            return

        # Prepare temporary model for ALA weight learning
        temp_model = copy.deepcopy(global_model).to(self.device)
        
        # Extract head parameters from all three models
        params_l = self._head_params(local_model)  # Local head (from previous round)
        params_g = self._head_params(global_model)  # Global head (current)
        params_t = self._head_params(temp_model)  # Temp head (to be mixed and updated)
        
        # Freeze backbone layers (we only adapt the head)
        for p in temp_model.convs.parameters():
            p.requires_grad = False
        for p in temp_model.batch_norms.parameters():
            p.requires_grad = False

        # Ensure head parameters are trainable
        for p in params_t:
            p.requires_grad = True
        
        # Create DataLoader with subset of training graphs for ALA
        ala_loader = self._build_ala_loader()

        # Initialize ALA weights (ones = equal mixing of local and global)
        if self.ala_reset_each_round or (self.ala_weights is None):
            self.ala_weights = [torch.ones_like(p.data, device=self.device) for p in params_t]

        # Round 1: Learn weights until convergence
        if round_id == 1:
            stable_cnt = 0  # Counter for consecutive epochs meeting convergence criterion
            
            for _ in range(self.max_init_epochs):  # Up to 20 epochs
                w_prev = [w.detach().clone() for w in self.ala_weights]  # Save previous weights
                _ = self._ala_one_epoch_update(temp_model, params_t, params_l, params_g, ala_loader)

                # Check convergence: max |Δw| across all weights
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
        # Backbone: Use global
        # Head: Mix local and global using learned weights
        self.task.model.load_state_dict(global_state_dict, strict=True)  # Start with global
        
        # Apply ALA mixing to head parameters: θ_head = θ_l + (θ_g - θ_l) * w
        model_params_t = self._head_params(self.task.model)
        self._apply_head_mix_(model_params_t, params_l, params_g)
