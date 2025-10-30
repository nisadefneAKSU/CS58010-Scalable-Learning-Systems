# This script runs a baseline federated graph learning experiment on a dataset like Cora and saves results to results.txt

import os
import torch
from torch.optim import Adam
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
from openfgl.data.distributed_dataset_loader import FGLDataset
import io
import sys

# -------------------------------
# Step 1: Set up experiment arguments
# -------------------------------
args = config.args
# Directory to store or load datasets
args.root = "./data"
args.dataset = ["Cora"]  # Can also use ["Citeseer"] or ["Pubmed"]

args.simulation_mode = "subgraph_fl_label_skew"
args.skew_alpha = 0.5
args.num_clients = 10
args.fl_algorithm = "fedala"
args.model = ["gcn"]
args.metrics = ["accuracy"]
args.num_rounds = 50
args.lr = 0.01
args.weight_decay = 5e-4
args.local_epochs = 1
args.seed = 42

args.params_weight = "samples_num"

# -------------------------------
# Step 2: Patch torch.load for PyTorch 2.6+ issue
# -------------------------------
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# -------------------------------
# Step 3: Initialize federated trainer
# -------------------------------
trainer = FGLTrainer(args)

# -------------------------------
# Step 5: Start federated training, evaluate the global model and save results
# -------------------------------
# Capture printed output from evaluation
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()

trainer.train()
trainer.evaluate()  # this prints metrics to console

sys.stdout = old_stdout  # restore stdout

'''# Save printed results to file
with open("results.txt", "w") as f:
    f.write(mystdout.getvalue())'''

with open("results.txt", "w", encoding="utf-8") as f:
    f.write(mystdout.getvalue())


print("Evaluation results saved to results.txt")

# -------------------------------
# Optional: Inspect one client's data
# -------------------------------
data = torch.load("data/distrib/subgraph_fl_label_skew_0.50_Cora_client_10/data_0.pt", weights_only=False)
print("Graph summary:", data)
print("Node features:", data.x.shape)
print("Labels:", data.y.shape)
print("Edge connections:", data.edge_index)


'''
OpenFGL-main/
└── data/
    └── distrib/
        └── subgraph_fl_label_skew_0.50_Cora_client_10/
            ├── data_0.pt
            ├── data_1.pt
            ├── data_2.pt
            ├── ...
            ├── data_9.pt
            ├── global_data.pt
            ├── pre_transform.pt
            ├── description.txt
            └── processed/
data_0.pt, data_1.pt, … → graph data for each client (10 clients in your case)
data.pt → combined global graph
description.txt → summary of how the dataset was split (dataset name, mode, etc.)
pre_transform.pt → preprocessing settings
processed/ → internal PyTorch Geometric cache '''