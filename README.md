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
```
Or download the ZIP file from [OpenFGL GitHub](https://github.com/xkLi-Allen/OpenFGL) and extract it.

### Install G-FedALA

```bash
git clone https://github.com/nisadefneAKSU/CS58010-Scalable-Learning-Systems.git
cd CS58010-Scalable-Learning-Systems

# Copy algorithm files to OpenFGL
cp gfedala/client.py /path/to/openfgl/flcore/gfedala/
cp gfedala/server.py /path/to/openfgl/flcore/gfedala/
```

---

## Quick Start

### Basic Usage

```python
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

args = config.args

# Data configuration
args.root = "your_data_root"
args.dataset = ["PROTEINS"]  # or ["MUTAG", "NCI1", "PTC_MR", ...]
args.num_clients = 10

# G-FedALA specific hyperparameters
args.fl_algorithm = "gfedala"
args.model = ["gin"]  # Graph Isomorphism Network

# G-FedALA hyperparameters
args.lambda_graph = 0.5        # Balance between param and graph distances
args.gala_temperature = 1.0    # Softmax temperature for aggregation weights
args.gala_warmup_rounds = 5    # Rounds to use graph+param similarity

# Training configuration
args.num_rounds = 100
args.num_epochs = 1
args.batch_size = 32
args.lr = 0.01

# Initialize and train
trainer = FGLTrainer(args)
trainer.train()
```

### Command Line

```bash
python main.py \
    --dataset PROTEINS \
    --num_clients 10 \
    --fl_algorithm gfedala \
    --model gin \
    --lambda_graph 0.5 \
    --gala_temperature 1.0 \
    --gala_warmup_rounds 5 \
    --num_rounds 100
```

---

## Method

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          G-FedALA Framework                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   Client 1   │     │   Client 2   │     │   Client N   │        │
│  │              │     │              │     │              │        │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │        │
│  │ │   ALA    │ │     │ │   ALA    │ │     │ │   ALA    │ │        │
│  │ │ Module   │ │     │ │ Module   │ │     │ │ Module   │ │        │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │        │
│  │      │       │     │      │       │     │      │       │        │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │        │
│  │ │  Local   │ │     │ │  Local   │ │     │ │  Local   │ │        │
│  │ │ Training │ │     │ │ Training │ │     │ │ Training │ │        │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │        │
│  │      │       │     │      │       │     │      │       │        │
│  │ ┌──────────┐ │     │ ┌──────────┐ │     │ ┌──────────┐ │        │
│  │ │  Graph   │ │     │ │  Graph   │ │     │ │  Graph   │ │        │
│  │ │Embedding │ │     │ │Embedding │ │     │ │Embedding │ │        │
│  │ └──────────┘ │     │ └──────────┘ │     │ └──────────┘ │        │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘        │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
│                    ┌──────────────────┐                            │
│                    │      Server      │                            │
│                    │                  │                            │
│                    │ ┌──────────────┐ │                            │
│                    │ │  Compute     │ │                            │
│                    │ │  Distances   │ │                            │
│                    │ │ (Param+Graph)│ │                            │
│                    │ └──────────────┘ │                            │
│                    │        │         │                            │
│                    │ ┌──────────────┐ │                            │
│                    │ │   Split      │ │                            │
│                    │ │ Aggregation  │ │                            │
│                    │ └──────────────┘ │                            │
│                    └──────────────────┘                            │
│                                                                      │
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

### Supported Datasets

G-FedALA supports all Graph-FL datasets in OpenFGL:

| Dataset | Graphs | Avg. Nodes | Avg. Edges | Classes | Domain |
|---------|--------|------------|------------|---------|--------|
| MUTAG | 188 | 17.9 | 19.8 | 2 | Chemistry |
| PROTEINS | 1,113 | 39.1 | 72.8 | 2 | Biology |
| NCI1 | 4,110 | 29.9 | 32.3 | 2 | Chemistry |
| PTC_MR | 344 | 14.3 | 14.7 | 2 | Chemistry |
| IMDB-BINARY | 1,000 | 19.8 | 96.5 | 2 | Social |
| COLLAB | 5,000 | 74.5 | 2,457.8 | 3 | Social |

### Supported GNN Models

- **GIN** (Graph Isomorphism Network) - Default
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE
- And more...

### Running Experiments

```bash
# PROTEINS dataset with 10 clients
python main.py --dataset PROTEINS --num_clients 10 --fl_algorithm gfedala

# NCI1 dataset with label-based partition
python main.py --dataset NCI1 --num_clients 5 --simulation_mode graph_fl_label

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
│   ├── run_gfedala.py
│   └── configs/
├── assets/
│   └── gfedala_banner.png
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Comparison with Baseline Methods

| Method | Client Personalization | Server Aggregation | Graph-Aware |
|--------|----------------------|-------------------|-------------|
| FedAvg | ✗ | Sample-weighted | ✗ |
| FedProx | Proximal term | Sample-weighted | ✗ |
| FedALA | ALA module | Sample-weighted | ✗ |
| **G-FedALA** | **ALA module** | **Similarity-weighted (split)** | **✓** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{gfedala2024,
  title={G-FedALA: Graph-Aware Federated Adaptive Local Aggregation for Federated Graph Learning},
  author={Your Name},
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built with ❤️ for the Federated Graph Learning community</i>
