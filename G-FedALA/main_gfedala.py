import random 
import numpy as np
import torch
import openfgl.config as config  # Import OpenFGL configuration module
from openfgl.flcore.trainer import FGLTrainer  # Import the FGL trainer class

args = config.args  # Get the argument parser object from OpenFGL config
args.root = "./data"  # Root directory where datasets will be loaded
args.scenario = "graph_fl"  # FL scenario is Graph-FL
args.task = "graph_cls"  # Task type: Graph classification (each graph is a sample to classify)
args.dataset = ["PROTEINS"]  # Dataset(s) to use
args.simulation_mode = "graph_fl_label_skew"  # Simulation mode: Non-IID data with label distribution skew
args.skew_alpha = 1  # Alpha parameter for label skew (Lower -> More skewed distribution)
args.dirichlet_alpha = 1  # Dirichlet distribution alpha for non-IID data partitioning
args.num_clients = 10  # Number of clients
args.num_rounds = 100  # Number of federated communication rounds
args.lr = 0.001  # Learning rate for local client optimization
args.num_epochs = 1  # Number of local training epochs per round
args.batch_size = 128  # Batch size for local training on each client
args.weight_decay = 5e-4  # Weight decay coefficient
args.dropout = 0.5  # Dropout rate for regularization in GNN layers
args.optim = "adam"  # Optimizer: Adam optimizer for gradient descent
args.use_wandb = True  # Enable W&B logging for experiment tracking
args.wandb_project = "pFGL"  # W&B project name for organizing experiments
args.wandb_entity = "scalable-group2"  # W&B entity/team name

args.lambda_graph = 0.5  # λ: Trade-off weight between graph distance and parameter distance
args.gala_temperature = 0.5  # τ: Softmax temperature for aggregation weights
args.gala_warmup_rounds = 10  # Number of warm-up rounds using graph+param distances

if True:  # Flag to switch between different FL algorithms
    args.fl_algorithm = "fedala"  # G-FedALA with graph awareness
    args.model = ["gin"]  # GNN architecture: Graph Isomorphism Network (GIN)
else:
    args.fl_algorithm = "fedproto"  # Alternative is FedProto algorithm
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"]  # Multiple models for model heterogeneity

args.metrics = ["accuracy"]  # Metrics to track is classification accuracy

# Patch torch.load for PyTorch 2.6+ security changes
original_torch_load = torch.load  # Save reference to original torch.load function
def patched_torch_load(*args, **kwargs):
    """
    Wrapper function to handle PyTorch 2.6+ weights_only parameter.
    PyTorch 2.6+ requires explicit weights_only flag for security.
    This ensures backward compatibility with older OpenFGL code.
    """
    # Ensure backward compatibility with older model loading code
    if "weights_only" not in kwargs:  # If weights_only not specified
        kwargs["weights_only"] = False  # Set to False to allow loading all objects (legacy behavior)
    return original_torch_load(*args, **kwargs)  # Call original function with patched kwargs
torch.load = patched_torch_load  # Override torch.load globally with patched version

trainer = FGLTrainer(args)  # Create trainer object above arguments
trainer.train()  # Start federated training process (100 rounds)