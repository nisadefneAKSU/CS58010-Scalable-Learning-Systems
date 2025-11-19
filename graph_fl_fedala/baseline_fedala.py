import os
import torch
from torch.optim import Adam
import openfgl.config as config # Import the OpenFGL configuration module
from openfgl.flcore.trainer import FGLTrainer # Import the federated learning trainer
from openfgl.data.distributed_dataset_loader import FGLDataset # For loading datasets in distributed FL
import io
import sys

"""This script runs a baseline federated graph learning experiment on a dataset and saves results to results.txt"""

# Set up experiment arguments
args = config.args

# Directory to store or load datasets
args.root = "./data"

# Choose dataset(s) for the experiment
args.dataset = ["DHFR"] # or DD

# Graph classification task
args.task = "graph_cls"

# Graph federated learning
args.scenario = "graph_fl"

# Specify simulation mode for non-iid data among clients
args.simulation_mode = "graph_fl_label_skew"  # Skewed label distribution among clients

# Dirichlet parameter controlling label skew (lower = more skewed)
args.skew_alpha = 0.5 

# Number of clients in the federated setting
args.num_clients = 10

# Use FedALA
args.fl_algorithm = "fedala"

# Model(s) to use on each client
args.model = ["gin"]  # GIN (Graph Isomorphism Network) for graph classification

# Metrics to report during evaluation
args.metrics = ["accuracy"]

# Training hyperparameters
args.num_rounds = 100  # Number of communication rounds
args.lr = 0.005  # Learning rate for local optimizer
args.weight_decay = 5e-4  # Weight decay for regularization
args.local_epochs = 3  # Number of local epochs per client before aggregation
args.seed = 42  # Random seed for reproducibility
args.params_weight = "samples_num"  # Weight aggregation based on number of samples per client

# Patch torch.load for PyTorch 2.6+ issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # Ensure backward compatibility with older model loading code
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load # Override torch.load with patched version

# Create a federated learning trainer using the args
trainer = FGLTrainer(args)

# Start federated training, evaluate the global model and save results
# Capture printed output from evaluation
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()
trainer.train()
trainer.evaluate()  # This prints metrics to console
sys.stdout = old_stdout

with open("results.txt", "w", encoding="utf-8") as f:
    f.write(mystdout.getvalue())

print("Evaluation results saved to results.txt")
