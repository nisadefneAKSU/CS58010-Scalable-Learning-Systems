import numpy as np
import torch
import ALA
from openfgl.flcore.base import BaseClient
import random
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from openfgl.data.simulation import get_subgraph_pyg_data
import torch.nn as nn
import copy
class FedALAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FedAvgClient.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between client and server.
            device (torch.device): Device to run the computations on.
        """
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.start_phase =  True
            
        
    def execute(self):
        
        global_param =self.message_pool["server"]["weight"]
        self.local_initialization(global_param)
        self.task.train()


    def send_message(self):
        """
        Sends a message to the server containing the model parameters after training
        and the number of samples in the client's dataset.
        """
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }

    def local_initialization(self, global_model):

        # first, load global model weights to the local model(s)
        # store global model copy
        # store local model copy
        # sample a random subset of training nodes for ALA
        # create a subgraph Data object for these nodes
        # get global and local model parameters
        # if first round, skip adaptation (at the first round, global and local models are identical)
        # if not first round: start ALA
        # create a temporary model copy # Aradığımız şey w’yi öğrenmek, local parametreleri güncellemek değil. bu yüzden temp model gerekli.
        # get the top layer parameters from global, local, and temp models
        # freeze lower layers of temp model
        # initialize adaptive weights with ones
        # forward pass on temp model with subgraph data
        # compute loss
        # backpropagate to get gradients (loss.backward())
        # update adaptive weights with Eq(5) --> weight = weight - eta * (param_t.grad * (param_g - param_l))
        # reconstruct the top layers of the temp model with --> param_l + (param_g - param_l) * weight
        # check convergence criteria: if converged, break the loop
        # at the end, get the top layers of the temp_model and write it to self.task.model
        print("Starting adaptive local aggregation...", flush=True)

        for p, gp in zip(self.task.model.parameters(), global_model):
            p.data.copy_(gp.data)
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
        rand_num = max(1, int(0.8* len(train_idx))) # rand_percent
        rand_nodes = random.sample(train_idx, rand_num)

        sub_data = get_subgraph_pyg_data(self.task.data, rand_nodes)

        # Some GIN implementations expect a batch vector
        # If batch info does not exist, create a default batch of zeros (single graph)
        if not hasattr(sub_data, "batch"):
            print("Adding batch info to sub_data for GIN model...", flush=True)
            sub_data.batch = torch.zeros(sub_data.x.size(0), dtype=torch.long).to(self.device)

        # Get model parameters
        params_g = list(global_model.parameters()) # Parameters from the global (server) model
        params_l = list(local_model.parameters()) # Parameters from the local (client) model
        

        # Skip adaptation if this is the first round (global and local models are identical)
        if len(params_g) == 0 or torch.allclose(params_g[0].data, params_l[0].data):
            return # very important: if round = 0, skip adaptation

        # Preserve lower layers, adapt only top layers
        # p: how many of the TOP layers of the local model ALA is allowed to modify. 
        # num_layers - p layers are directly copied from the global model. --> consists of one's.

        # let's say we have total 5 layers and layer_idx=2, then: the weight matrix consists of 3 layers. initially all ones.
        # we will learn this weight matrix in each iteration. 

        layer_idx = 1

        # Prepare top layers for adaptation
        temp_model = copy.deepcopy(local_model) # Temporary model for gradient updates
        params_t = list(temp_model.parameters()) # All parameters of the temporary model
        params_p = params_l[-layer_idx:] # Top layers of local model
        params_gp = params_g[-layer_idx:] # Top layers of global model
        params_tp = params_t[-layer_idx:] # Top layers of temp model

        # Freeze lower layers of temporary model so only top layers get gradients
        for p in params_t[:-layer_idx]:
            p.requires_grad = False

        # Initialize adaptive weights if not already initialized
        # These weights control interpolation between local and global top layers
        if self.weights is None:
            self.weights = [torch.ones_like(p.data, device=self.device, dtype=p.data.dtype) for p in params_p] # Start with all ones (full influence from global model)

        # Track the loss values over iterations for stopping criteria
        losses = []
        max_iters = 20 # Safety cap to prevent infinite loops
        it = 0 # Iteration counter
        it_thr = 10
        # Track consecutive loss increases to use in stopping rule
        consecutive_increase = 0
        last_loss = None

        # Start the adaptive weight learning loop (while Wip does not converge)
        while True:
            it += 1 # Increment iteration count
            for pt in params_tp:
                if pt.grad is not None:
                    pt.grad.detach_() # Detach the gradient from computation graph to avoid accumulation
                    pt.grad.zero_() # Zero out the gradient tensor
            # 1) loop'a girmeden (veya her iterasyonun başında) temp katmanları karıştır: 
            # # Interpolation formula: pt = local + (global - local) * w
            # If w = 1 -> fully global
            # If w = 0 -> fully local
            # Values in between -> Mix based on learned weight
            #pt.data = p_data + (pg_data - p_data) * w
            for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                param_t.data = param_l.data + (param_g.data - param_l.data) * weight

            # 2) sonra forward:
            # Forward pass through the model
                # Use the forward pass helper function above to handle different GNN forward signatures
                # temp_model: the temporary model copy used for top-layer adaptation
                # sub_data: the sampled subgraph Data object
            try:
                embedding , logits = temp_model(sub_data) # Returns predicted outputs
            except Exception:
                # If the forward pass fails (e.g., incompatible dataset or model) then stop
                print("Forward pass failed during ALA adaptation. Stopping adaptation.", flush=True)
                break
            
            # sub_data.y: ground-truth labels for the sampled nodes
            loss_value = self.task.loss(logits, sub_data.y)

            # Backpropagation to compute gradients
            # temp_model's top layer parameters (params_tp) will have their gradients stored in pt.grad
            # These gradients are used later to update the adaptive interpolation weights "w"
            loss_value.backward()

            # 3) weightleri güncelle:
            # If gradient points in the same direction as diff, reduce weight (less global influence)
            # If gradient points opposite to diff, increase weight (more global influence)
            for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                grad_w = param_t.grad * (param_g.data - param_l.data)
                weight.data = torch.clamp(weight.data - 0.005 * grad_w, 0.0, 1.0)

            # 4) yeni weight’lerle param_t’yi tekrar güncelle (bir sonraki iterasyon için): Interpolate the top layers of the temporary model between local and global
            for param_t, param_l, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                param_t.data = param_l.data + (param_g.data - param_l.data) * weight

            # Keep track of the current loss value so the loop can decide when to stop
            losses.append(loss_value.item())

            if self.start_phase == False:
                break

            # The adaptation loop needs a rule to know when to stop updating adaptive weights
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

            # Check if the last it_thr losses are very stable (small standard deviation)


    
            recent_std_ok = (len(losses) > it_thr and np.std(losses[-it_thr:]) < 1e-3)

            # Stop adaptation if any of these are true:
            # Loss increased 3 times consecutively
            # Recent losses are stable
            # Reached maximum number of iterations
            if (consecutive_increase >= 4) or recent_std_ok or (it >= max_iters and len(losses) > 0):
                break

        # Mark that first-phase adaptation is done (ensures full adaptation only happens in the first round)
        self.start_phase = False

        # After learning the interpolation weights and updating the temp_model,
        # we copy the top-layer parameters from temp_model back to the local_model
        for p, pt in zip(params_p, params_tp):
            p.data = pt.data.clone() # Ensures actual top-layer weights are updated

        # Write adapted local model back to the actual task model to send back to the server
        self.task.model.load_state_dict(local_model.state_dict())