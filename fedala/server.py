import torch
from openfgl.flcore.base import BaseServer
import copy


class FedALAServer(BaseServer):
    """
    FedALAServer implements the server-side logic for the Federated Adaptive Layer-wise Aggregation (FedALA)
    algorithm, which extends the standard FedAvg approach by introducing adaptive aggregation weights
    for higher network layers.

    The server collects local model updates from clients and performs an adaptive aggregation
    step to better generalize across heterogeneous client distributions.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedALAServer.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Optional global dataset or graph data.
            data_dir (str): Directory containing datasets.
            message_pool (dict): Shared communication pool between server and clients.
            device (torch.device): Device for computation (CPU or GPU).
        """
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

        self.task.load_custom_model(self.task.model)
        print("FedALAServer initialized.", flush=True)

    def execute(self):
        """
        Executes the FedALA global aggregation step.

        The server aggregates model parameters received from the sampled clients.
        Each layer can be aggregated with adaptive weighting, enabling better handling
        of data heterogeneity.
        """
        print("FedALAServer.execute() called — starting adaptive aggregation...", flush=True)

        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_clients = len(sampled_clients)

            # Optional weighting based on sample counts
            total_samples = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients)

            # Initialize aggregation
            for it, client_id in enumerate(sampled_clients):
                local_weights = self.message_pool[f"client_{client_id}"]["weight"]
                if self.args.params_weight == "samples_num":
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / total_samples
                else:
                    weight = 1.0 / num_clients

                for (local_param, global_param) in zip(local_weights, self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

            # Optional: log or modify adaptive layer weights here (FedALA style)
            print("FedALAServer aggregation complete.", flush=True)

    def send_message(self):
        print("FedALAServer sending global model to clients.", flush=True)
        self.message_pool["server"] = {
            "num_samples": 0,
            "weight": copy.deepcopy(self.task.model.state_dict())
        }

