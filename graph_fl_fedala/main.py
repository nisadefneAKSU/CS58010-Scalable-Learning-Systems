import random
import numpy as np
import torch
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "/home/ceren/Desktop/scalable/OpenFGL-main/data"
args.scenario = "graph_fl"
args.task = "graph_cls"

args.dataset = ["DD"]
args.simulation_mode = "graph_fl_label_skew"
args.skew_alpha = 1
args.num_clients = 10
args.lr = 0.001
args.num_epochs = 1
args.num_rounds = 100
args.batch_size = 128
args.weight_decay = 5e-4
args.dropout = 0.5
args.optim = "adam"
args.seed = 42
if True:
    args.fl_algorithm = "fedala"
    args.model = ["gin"]
else:
    args.fl_algorithm = "fedproto"
    args.model = ["gcn", "gat", "sgc", "mlp", "graphsage"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]

# Patch torch.load for PyTorch 2.6+ issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # Ensure backward compatibility with older model loading code
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load # Override torch.load with patched version

trainer = FGLTrainer(args)

trainer.train()
