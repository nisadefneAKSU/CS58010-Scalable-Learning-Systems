import copy  # For deep copying model state dictionaries
import numpy as np
import torch
import time
from openfgl.flcore.fedala.client import FedALAClient
from threading import Thread
from openfgl.flcore.base import BaseServer  # Base server class from OpenFGL
from collections import OrderedDict

class FedALAServer(BaseServer):
    """Server-side implementation of FedALA (Federated Adaptive Local Aggregation).
    Key responsibilities:
    1. Collect model updates from clients
    2. Aggregate models using sample-weighted averaging (FedAvg-style)
    3. Broadcast updated global model to all clients"""
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """Initialize FedALA server.
        Args:
            args: Configuration containing hyperparameters
            global_data: Global validation/test data (if any)
            data_dir: Directory for data storage
            message_pool: Shared dictionary for client-server communication
            device: torch.device for computations (CPU/GPU)"""
        
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)
   
    def execute(self):
        """Main server aggregation logic executed each round.
        Process:
        1. Collect model updates from sampled clients
        2. Compute sample-weighted average of model parameters
        3. Update global model with aggregated parameters
        
        Note: Uses FedAvg-style aggregation where each client's contribution
        is weighted by its number of training samples."""
        
        with torch.no_grad():  # No gradients needed for aggregation
            sampled = self.message_pool["sampled_clients"]  # List of client IDs participating this round
            
            # Compute total number of samples across all clients
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled)

            # Initialize aggregated state dictionary
            global_state = self.task.model.state_dict()  # Current global model parameters
            agg_state = OrderedDict()

            for k, v in global_state.items():
                if torch.is_floating_point(v):
                    agg_state[k] = torch.zeros_like(v)  # Initialize to zero for accumulation
                else:
                    # Do NOT average integers/bools (e.g., BatchNorm num_batches_tracked)
                    agg_state[k] = v.clone()

            # Weighted aggregation: Σ (n_i / n_total) * θ_i
            for cid in sampled:
                weight = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot  # Sample-based weight
                client_state = self.message_pool[f"client_{cid}"]["state"]

                for k in agg_state.keys():
                    if torch.is_floating_point(agg_state[k]):
                        # Ensure same device before accumulation
                        ck = client_state[k].to(agg_state[k].device)
                        agg_state[k].add_(ck, alpha=float(weight))  # Weighted sum
                    else:
                        # Keep integer/bool parameters as-is
                        pass
                    
            # Update global model with aggregated parameters
            self.task.model.load_state_dict(agg_state, strict=True)
        
    def send_message(self):
        """Send updated global model to clients for next round.
        Message contents:
        - state: Deep copy of aggregated global model parameters"""
        
        self.message_pool["server"] = {
            "state": copy.deepcopy(self.task.model.state_dict())  # Deep copy to prevent shared references
        }
