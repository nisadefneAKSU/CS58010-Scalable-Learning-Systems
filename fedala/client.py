import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from torch.utils.data import DataLoader
from openfgl.flcore.base import BaseClient


class FedALAClient(BaseClient):
    '''
    FedALAClient implements the client-side logic for the Federated Adaptive Layer Aggregation (FedALA) algorithm.

    FedALA extends FedAvg by introducing adaptive layer-wise interpolation weights between the
    global model and the local model. Each client learns weights that determine how much to trust
    global vs. local updates for each layer.

    Reference:
        "FedALA: Adaptive Local Aggregation for Personalized Federated Learning"
        (Zhang et al., 2023)
    '''

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

        self.layer_idx = 1        # Number of higher layers to adapt
        self.rand_percent = 20    # % of local data sampled for weight learning
        self.eta = 0.1            # Weight learning rate
        self.weights = None       # Layer-wise interpolation weights
        self.start_phase = True   # Only perform full adaptation in first round

    def execute(self):
        """
        Executes the local training phase of the FedALA client:
        1. Synchronizes with the global model.
        2. Trains locally.
        3. Performs adaptive local aggregation.
        """
        print(f"Client {self.client_id}: execute() called", flush=True)

        # Step 1: Synchronize with global model
        self.task.model.load_state_dict(self.message_pool["server"]["weight"])

        # Step 2: Train locally
        self.task.train()

        # Step 3: Adaptive local aggregation
        self.adaptive_local_aggregation()

    def adaptive_local_aggregation(self):
        """
        Learns per-layer adaptive interpolation weights between the global and local parameters.
        Compatible with NodeClsTask (uses train_mask to select training nodes).
        """
        print(f"Client {self.client_id}: adaptive_local_aggregation() called", flush=True)

        # -------------------------------
        # Step 1: Get global and local model copies
        # -------------------------------
        global_model = copy.deepcopy(self.task.model)
        local_model = copy.deepcopy(self.task.model)

        # -------------------------------
        # Step 2: Sample subset of training nodes
        # -------------------------------
        train_idx = self.task.train_mask.nonzero(as_tuple=False).squeeze().tolist()
        if not isinstance(train_idx, list):
            train_idx = [train_idx]

        rand_ratio = self.rand_percent / 100
        rand_num = max(1, int(rand_ratio * len(train_idx)))

        rand_nodes = random.sample(train_idx, rand_num)

        x_sample = self.task.data.x[rand_nodes].to(self.device)
        y_sample = self.task.data.y[rand_nodes].to(self.device)

        # -------------------------------
        # Step 3: Layer-wise parameter handling
        # -------------------------------
        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())

        # Skip adaptation if models are identical (first round)
        if torch.sum(params_g[0] - params_l[0]) == 0:
            return

        # Preserve lower layers
        for p, pg in zip(params_l[:-self.layer_idx], params_g[:-self.layer_idx]):
            p.data = pg.data.clone()

        # Temporary model for higher layers
        temp_model = copy.deepcopy(local_model)
        params_t = list(temp_model.parameters())
        params_p = params_l[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # Freeze lower layers
        for p in params_t[:-self.layer_idx]:
            p.requires_grad = False

        # Initialize adaptive weights if not already
        if self.weights is None:
            self.weights = [torch.ones_like(p.data) for p in params_p]

        optimizer = torch.optim.SGD(params_tp, lr=0)

        # -------------------------------
        # Step 4: Weight learning loop
        # -------------------------------
        losses = []
        while True:
            optimizer.zero_grad()
            output = temp_model(x_sample)
            loss_value = self.task.loss(output, y_sample)
            loss_value.backward()

            # Update layer-wise weights
            for pt, p, pg, w in zip(params_tp, params_p, params_gp, self.weights):
                w.data = torch.clamp(w - self.eta * (pt.grad * (pg.data - p.data)), 0, 1)

            # Update temp model using new weights
            for pt, p, pg, w in zip(params_tp, params_p, params_gp, self.weights):
                pt.data = p.data + (pg.data - p.data) * w

            losses.append(loss_value.item())

            if not self.start_phase:
                break
            if len(losses) > 5 and np.std(losses[-5:]) < 1e-3:
                break

        self.start_phase = False

        # -------------------------------
        # Step 5: Apply adapted layers back to local model
        # -------------------------------
        for p, pt in zip(params_p, params_tp):
            p.data = pt.data.clone()

        # Update local model
        self.task.model.load_state_dict(local_model.state_dict())


    def send_message(self):
        """
        Sends the updated model weights and sample count to the server.
        """
        print(f"Client {self.client_id}: send_message() called", flush=True)
        
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters())
        }
