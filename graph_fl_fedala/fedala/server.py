import torch
import copy
from openfgl.flcore.base import BaseServer

class FedALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        # Call the basic server setup from BaseServer
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

        # Some loaders might fail to set self.task (bug or problem in some datasets), if task isn't set, aggregation cannot work
        # We must make sure the server has a task (the model and data it will work on)
        if getattr(self, "task", None) is None:
            raise RuntimeError("Server task not initialized. Ensure dataset and args.model are valid so BaseServer initializes self.task.")
        
        # Some special FedALA models might want to load a custom model instead of the default
        if hasattr(self.task, "load_custom_model") and callable(getattr(self.task, "load_custom_model")):
            try:
                # Try to replace the default model with the custom one
                self.task.load_custom_model(self.task.model)
            except Exception:
                # If it fails, just ignore and keep the default model
                pass
        
        print("FedALAServer initialized.", flush=True)

    def execute(self):
        """Do one round of FedALA aggregation: Combine client models into the global model."""
        print("FedALAServer.execute() called — starting adaptive aggregation...", flush=True)

        # We don't need gradients here because we are just combining weights (safe and faster)
        with torch.no_grad():
            # Get the list of clients selected in this round by the trainer
            sampled_clients = self.message_pool.get("sampled_clients", [])
            if len(sampled_clients) == 0:
                # If no clients were selected, nothing to do
                print("No sampled clients this round.", flush=True)
                return
            
            # We make a copy of the server's current model parameters so we can update safely (global model before aggregation)
            server_dict = copy.deepcopy(self.task.model.state_dict())

            # Prepare a "container" or "accumulator" to store the combined weights
            acc = {} # This will hold the sums
            float_keys = [] # These are the parameters we can average (weights and biases)
            for k, v in server_dict.items():
                if torch.is_floating_point(v):
                     # For numbers (weights/biases), start with zeros
                    acc[k] = torch.zeros_like(v, dtype=torch.float32, device=v.device)
                    float_keys.append(k)
                else:
                    # For non-number stuff (like batch norm running mean/variance), just copy as-is
                    acc[k] = v.clone()

            # Compute total number of samples from selected clients
            # FedAvg-style weighting based on client dataset size
            # This is used to weight clients according to how much data they have
            total_samples = sum(self.message_pool.get(f"client_{cid}", {}).get("num_samples", 0) for cid in sampled_clients)
            if total_samples == 0:
                # If something is wrong, just treat all clients equally
                total_samples = len(sampled_clients)

            # FedALA’s core idea:
            #   Instead of simple averaging, FedALA clients send "adapted higher-layer weights"
            #   The server still aggregates them using weighted averaging, but the local weights contain client-specific interpolation.

            # Aggregate client models
            for cid in sampled_clients:
                # Get the client's message from the pool
                client_entry = self.message_pool.get(f"client_{cid}")
                if client_entry is None:
                    continue
                
                # Get the weights the client is sending
                local_dict = client_entry.get("weight")
                if local_dict is None:
                    continue

                # Calculate the "importance" of this client based on how many samples it has
                # Weight is proportional to number of samples (FedAvg logic)
                client_samples = client_entry.get("num_samples", 1)
                weight = float(client_samples) / float(total_samples)

                # Loop through all parameters of global model
                for k in server_dict.keys():
                    if k not in local_dict:
                        # If the client didn't send this parameter, skip it
                        continue
                    local_val = local_dict[k]

                    # If it's a number (weight/bias), add it to the accumulator and do weighted sum
                    if k in float_keys:
                        # Cast local to float for accumulation
                        if not torch.is_floating_point(local_val):
                            local_val = local_val.float()

                        # Weighted sum: acc[k] += weight * client's value
                        acc[k] += weight * local_val.to(acc[k].dtype).to(acc[k].device)
                    else:
                        # Non-float items (running_mean, running_var, etc.) remain untouched
                        pass

            # After summing all clients, make a new model state dict
            new_state = {}
            for k, orig_v in server_dict.items():
                if k in float_keys:
                    # Make sure numbers have the same type as the original server parameters
                    new_state[k] = acc[k].to(orig_v.dtype)
                else:
                    # Non-number parameters remain as-is again
                    new_state[k] = acc[k]

            # Load the new aggregated global model into the server
            self.task.model.load_state_dict(new_state)

        print("FedALAServer aggregation complete.", flush=True)

    def send_message(self):
        """ Send the updated global model to all clients
        by writing it into the message pool so clients can download it."""

        self.message_pool["server"] = {
            "num_samples": 0, # Server doesn't have its own data
            "weight": copy.deepcopy(self.task.model.state_dict()) # Deep copy to avoid clients accidentally modifying server params during training
        }