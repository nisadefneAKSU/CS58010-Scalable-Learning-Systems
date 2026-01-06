<h1 align="center">G-FedALA: Graph-Aware Federated Adaptive Local Aggregation</h1>

<p align="center">
  <b>Extending FedALA for Federated Graph Learning with Structure-Aware Aggregation</b>
</p>

<p align="center">
  <a href="#highlights">Highlights</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#method">Method</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenFGL-Compatible-blue" alt="OpenFGL Compatible"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="Python"/>
</p>

---

## Overview

**G-FedALA** (Graph-Aware Federated Adaptive Local Aggregation) is a novel federated graph learning algorithm that extends [FedALA](https://arxiv.org/abs/2212.01197) (AAAI 2023) to the graph domain. Built on the [OpenFGL](https://github.com/xkLi-Allen/OpenFGL) framework, G-FedALA addresses the unique challenges of federated learning on graph-structured data by integrating graph-structure awareness into the server-side aggregation process.

### Key Idea

In standard federated learning, statistical heterogeneity across clients degrades the performance of the global model. While FedALA addresses this through adaptive local aggregation on the client side, G-FedALA goes further by:

1. **Client-Side**: FedALA-based adaptive local aggregation using loss-driven, element-wise mixing of local and global model parameters. Clients compute graph embeddings locally and share them with the server.
2. **Server-Side**: Structure-aware aggregation that combines parameter similarity and graph-embedding similarity to weight client contributions.

---

## Highlights

| Feature | Description |
|---------|-------------|
| 🔷 **Graph-Aware Aggregation** | Utilizes client graph embeddings for structure-aware server aggregation |
| 🔷 **Adaptive Local Aggregation** | Learns per-parameter mixing weights for personalized initialization |
| 🔷 **Split Aggregation Strategy** | Backbone: Similarity-weighted, Head: Sample-size–weighted |
| 🔷 **Warm-up Mechanism** | Initial rounds use graph+param distances, later rounds use only param distances |
| 🔷 **OpenFGL Compatible** | Seamlessly integrates with the OpenFGL benchmark framework |
| 🔷 **Graph-FL Ready** | Designed for graph classification tasks in the Graph-FL scenario |

---

## Installation

### Prerequisites

```bash
# Python 3.8+
# PyTorch 2.0+
# PyTorch Geometric
# Anaconda
# Git
```

### Install OpenFGL

Install from source:

```bash
git clone https://github.com/xkLi-Allen/OpenFGL.git
cd OpenFGL
pip install -e .
```
Or download the ZIP file from [OpenFGL GitHub](https://github.com/xkLi-Allen/OpenFGL) and extract it.

### Install Our Repository

```bash
git clone https://github.com/nisadefneAKSU/CS58010-Scalable-Learning-Systems.git
cd CS58010-Scalable-Learning-Systems
```

## Quick Start

Option A: Using Conda (Recommended for Windows)

```bash
# Create a new conda environment
conda create -n openfgl_env python=3.9
conda activate openfgl_env

# Install PyTorch (adjust CUDA version as needed)
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU (CUDA 11.8 example):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install other dependencies
pip install -r docs/requirements.txt
```
Option B: Using pip

```bash
# Create virtual environment
python -m venv openfgl_env
source openfgl_env/bin/activate  # On Windows: openfgl_env\Scripts\activate

# Install dependencies
pip install -r docs/requirements.txt
```

---

Add FedALA and G-FedALA Files

Place the following files in the OpenFGL repository structure:

### 3.1 Algorithm Implementation Folders
```
OpenFGL/
├── openfgl/
│   ├── flcore/
│   │   ├── fedala/              # ← Add this folder
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   └── server.py
│   │   └── gfedala/             # ← Add this folder
│   │       ├── __init__.py
│   │       ├── client.py
│   │       └── server.py
```

**What these contain:**
- `client.py`: Client-side training logic 
- `server.py`: Server-side aggregation logic
- `__init__.py`: Exports client and server classes

---

### 3.2 Configuration Files
```
OpenFGL/
├── openfgl/
│   ├── config.py      # ← Replace this file
│   │
│   └── utils/
│       └── basic_utils.py      # ← Replace this file
```

**Modified files:**
- `config.py`: Adds `"fedala"` and `"gfedala"` to supported algorithms
- `basic_utils.py`: Adds FedALA/G-FedALA client and server loading logic

---

### Main Training Script
```
OpenFGL/
└── main.py                      # ← Add this file
```

**What it contains:**
- Dataset configuration
- Hyperparameter settings
- Training loop initialization

---

### Requirements File (Optional)
```
OpenFGL/
└── docs/
    └── requirements.txt         # ← Replace if using Windows-specific setup
```

---

## Verify File Structure

Your directory should look like this:
```
OpenFGL-main/
├── main.py                      # Your training script
├── openfgl/
│   ├── flcore/
│   │   ├── config.py            # Modified
│   │   ├── fedala/              # NEW
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   └── server.py
│   │   └── gfedala/             # NEW
│   │       ├── __init__.py
│   │       ├── client.py
│   │       └── server.py
│   └── utils/
│       └── basic_utils.py       # Modified
├── docs/
│   └── requirements.txt         # Updated (optional)
└── data/                        # Will be auto-generated
```

Edit main.py to set your experiment configuration (you can change other arguments to your liking):
```python
# Select algorithm
args.fl_algorithm = "fedala"  # or "gfedala"
```

## Method

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          G-FedALA Framework                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   Client 1   │     │   Client 2   │     │   Client N   │         │
│  │              │     │              │     │              │         │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │         │
│  │ │   ALA    │ │     │ │   ALA    │ │     │ │   ALA    │ │         │
│  │ │ Module   │ │     │ │ Module   │ │     │ │ Module   │ │         │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │         │
│  │      │       │     │      │       │     │      │       │         │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │         │
│  │ │  Local   │ │     │ │  Local   │ │     │ │  Local   │ │         │
│  │ │ Training │ │     │ │ Training │ │     │ │ Training │ │         │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │         │
│  │      │       │     │      │       │     │      │       │         │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │         │
│  │ │  Graph   │ │     │ │  Graph   │ │     │ │  Graph   │ │         │
│  │ │Embedding │ │     │ │Embedding │ │     │ │Embedding │ │         │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │         │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘         │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
│                    ┌──────────────────┐                             │
│                    │      Server      │                             │
│                    │                  │                             │
│                    │ ┌──────────────┐ │                             │
│                    │ │  Compute     │ │                             │
│                    │ │  Distances   │ │                             │
│                    │ │ (Param+Graph)│ │                             │
│                    │ └──────────────┘ │                             │
│                    │        │         │                             │
│                    │ ┌──────────────┐ │                             │
│                    │ │   Split      │ │                             │
│                    │ │ Aggregation  │ │                             │ 
│                    │ └──────────────┘ │                             │
│                    └──────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Client-Side: Adaptive Local Aggregation (ALA)

The ALA module learns element-wise mixing weights `w ∈ [0,1]` for head parameters:

```
θ_head = θ_local + (θ_global - θ_local) ⊙ w
```

**Weight Learning Process:**
1. Initialize weights to 1 (full global)
2. Forward pass on local data with mixed parameters
3. Compute gradients: `∇w = ∇θ ⊙ (θ_global - θ_local)`
4. Update: `w ← clip(w - η · ∇w, 0, 1)`

**Three-Phase Strategy:**
- **Round 0**: Skip ALA (global = local)
- **Round 1**: Learn weights until convergence (up to 20 epochs)
- **Round 2+**: Single epoch refinement

### Server-Side: Graph-Aware Aggregation

**Distance Computation:**

1. **Parameter Distance** (using head parameters):
```
d_param(i) = √(mean_k[(||θ_i^k - θ_g^k|| / ||θ_g^k||)²])
```

2. **Graph Distance** (during warm-up):
```
d_graph(i) = ||h_i - h_global|| / ||h_global||
```

**Aggregation Weight Calculation:**

```
logit_i = -[λ · d_param(i) + (1-λ) · d_graph(i)]  # During warm-up
logit_i = -d_param(i)                              # After warm-up

α_i = softmax(logit / τ)
```

**Split Aggregation:**
- **Backbone**: Similarity-weighted aggregation using `α_i`
- **Head**: Sample-weighted aggregation (FedAvg-style)


> ⚠️ **Current Limitation:** This implementation is hardcoded for the **GIN (Graph Isomorphism Network)** architecture. The backbone-neck-head split logic can be extended to other GNN models by modifying the layer detection functions.


### Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `lambda_graph` | λ | 0.5 | Balance between param (λ) and graph (1-λ) distances |
| `gala_temperature` | τ | 1.0 | Softmax temperature for aggregation weights |
| `gala_warmup_rounds` | - | 5 | Rounds to incorporate graph similarity |
| `ala_eta` | η | 1.0 | Learning rate for ALA weight updates |
| `ala_data_ratio` | - | 0.8 | Fraction of local data used for ALA |

---

## Experiments

### Supported GNN Models

Currently, G-FedALA supports:

- **GIN** (Graph Isomorphism Network) ✓

> **Note:** The current implementation uses hardcoded layer detection for the GIN architecture (`convs`, `batch_norms` for backbone; `lin1`, `batch_norm1`, `lin2` for head). While the backbone-neck-head split strategy is conceptually applicable to other GNN architectures, extending support requires modifying the `_is_backbone()`, `_is_neck_head()`, and `_head_params()` functions to match the target model's layer naming conventions.

### Extending to Other GNN Models

To add support for a new GNN architecture (e.g., GCN, GAT), modify the following functions:

**In `server.py`:**
```python
def _is_backbone(k: str) -> bool:
    """Modify to match your model's backbone layer names."""
    # Example for GCN:
    # return k.startswith("conv_layers.") or k.startswith("bn_layers.")
    return k.startswith("convs.") or k.startswith("batch_norms.")

def _is_neck_head(k: str) -> bool:
    """Modify to match your model's head layer names."""
    # Example for GCN:
    # return k.startswith("fc1.") or k.startswith("fc2.")
    return k.startswith("lin1.") or k.startswith("batch_norm1.") or k.startswith("lin2.")
```

**In `client.py`:**
```python
@staticmethod
def _head_params(model):
    """Modify to return head parameters for your model."""
    # Example for GCN:
    # return list(model.fc1.parameters()) + list(model.fc2.parameters())
    return (
        list(model.lin1.parameters()) +
        list(model.batch_norm1.parameters()) +
        list(model.lin2.parameters())
    )
```

### Running Experiments

#### Data Simulation

G-FedALA uses the **Label Skew** simulation mode from OpenFGL to create non-IID data distribution across clients:

```python
args.scenario = "graph_fl"
args.task = "graph_cls"
args.simulation_mode = "graph_fl_label_skew"  # Label-based non-IID partition
args.skew_alpha = 1.0 # a parameter not mentioned in OpenFGl/config.py but required to be passed to run the project. We predict that it should be same as the value of args.dirichlet_alpha
args.dirichlet_alpha = 1.0  # Controls heterogeneity (lower = more heterogeneous)
```

> **Note:** The `dirichlet_alpha` parameter controls the degree of label distribution skew. Lower values create more heterogeneous (non-IID) data distributions across clients.

#### Example Commands

```bash
# PROTEINS dataset with 10 clients
python main.py --dataset PROTEINS --num_clients 10 --fl_algorithm gfedala

# ENZYMES dataset with label skew partition
python main.py --dataset ENZYMES --num_clients 5 --simulation_mode graph_fl_label_skew

# COLLAB dataset (social network)
python main.py --dataset COLLAB --num_clients 10 --fl_algorithm gfedala

# Ablation: Without graph similarity (λ=1.0)
python main.py --dataset MUTAG --fl_algorithm gfedala --lambda_graph 1.0
```

---

## File Structure

```
G-FedALA/
├── gfedala/
│   ├── __init__.py
│   ├── client.py          # GFedALAClient implementation
│   └── server.py          # GFedALAServer implementation
├── fedala/
│   ├── __init__.py
│   ├── client.py          # FedALAClient (baseline)
│   └── server.py          # FedALAServer (baseline)
├── experiments/
│   ├── gfedala_grid_search.py
│   ├── get_best_params.py
│   └── ablation.py
├── assets/
│   └── gfedala_banner.png
├── README.md
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{gfedala2024,
  title={G-FedALA: Graph-Aware Federated Adaptive Local Aggregation for Federated Graph Learning},
  author={Aksu N. D., Arkac C.},
  year={2024},
  note={Built on OpenFGL framework}
}
```

Please also cite the original FedALA paper:

```bibtex
@inproceedings{zhang2023fedala,
  title={FedALA: Adaptive Local Aggregation for Personalized Federated Learning},
  author={Zhang, Jianqing and Hua, Yang and Wang, Hao and Song, Tao and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={11237--11244},
  year={2023}
}
```

And the OpenFGL benchmark:

```bibtex
@misc{li2024openfgl,
  title={OpenFGL: A Comprehensive Benchmarks for Federated Graph Learning},
  author={Li, Xunkai and Zhu, Yinlin and Pang, Boyang and Yan, Guochen and Yan, Yeyu and Li, Zening and Wu, Zhengyu and Zhang, Wentao and Li, Rong-Hua and Wang, Guoren},
  year={2024},
  eprint={2408.16288},
  archivePrefix={arXiv}
}
```

---

## Acknowledgements

- [OpenFGL](https://github.com/xkLi-Allen/OpenFGL) - Comprehensive benchmark for federated graph learning
- [FedALA](https://github.com/TsingZ0/FedALA) - Original adaptive local aggregation method
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural network library

---

<p align="center">
  <i>Built with ❤️ for the Federated Graph Learning community</i>
</p>
