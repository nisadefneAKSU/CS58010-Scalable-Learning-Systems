"""
Grid Search for GFedALA Hyperparameters (No Fixed Seeds)

Hyperparameters to tune:
- lambda_graph: Weight for param distance vs graph distance (higher = more param-based)
- gala_temperature: Softmax temperature (lower = sharper weights)
- gala_warmup_rounds: Rounds to use graph similarity (0 = never use graph sim)

Each configuration is run multiple times WITHOUT fixed seeds (true randomness).
Results report mean ± std across runs.

Usage:
    python grid_search_gfedala.py
    python grid_search_gfedala.py --dataset MUTAG
    python grid_search_gfedala.py --dry_run
"""

import itertools
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

import torch
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer

# ============================
# Grid Search Configuration
# ============================
GRID = {
    "lambda_graph": [0.0, 0.3, 0.5, 0.7, 1.0],
    "gala_temperature": [0.05, 0.1, 0.5],
    "gala_warmup_rounds": [0, 5, 10],
}

NUM_RUNS = 3  # Runs per config (no fixed seeds)
DATASETS = ["IMDB-BINARY"]  # Default datasets

# Generate all combinations
CONFIGS = list(itertools.product(
    GRID["lambda_graph"],
    GRID["gala_temperature"],
    GRID["gala_warmup_rounds"]
))

print(f"Configurations: {len(CONFIGS)}")
print(f"Runs per config: {NUM_RUNS} (no fixed seeds)")
print(f"Total runs per dataset: {len(CONFIGS) * NUM_RUNS}")


# ============================
# Patch torch.load for PyTorch 2.6+
# ============================
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


def run_single_experiment(dataset, lambda_graph, gala_temperature, gala_warmup_rounds, run_id, dry_run=False):
    """
    Run a single GFedALA experiment.
    NO SEED IS SET - true randomness for each run.
    """
    exp_name = f"gfedala_lg{lambda_graph}_temp{gala_temperature}_warm{gala_warmup_rounds}_run{run_id}"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")
    
    if dry_run:
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "lambda_graph": lambda_graph,
            "gala_temperature": gala_temperature,
            "gala_warmup_rounds": gala_warmup_rounds,
            "run_id": run_id,
            "status": "dry_run",
        }
    
    try:
        # Get fresh args - deep copy to avoid state pollution between runs
        import copy
        args = copy.deepcopy(config.args)
        
        # Fixed settings
        args.root = "/home/ceren/Desktop/scalable/OpenFGL-main/data"
        args.scenario = "graph_fl"
        args.task = "graph_cls"
        args.dataset = [dataset]  # MUST be a list, e.g., ["PROTEINS"]
        print(f"Using dataset: {args.dataset}")  # Debug print
        args.simulation_mode = "graph_fl_label_skew"
        args.skew_alpha = 1
        args.dirichlet_alpha = 1
        args.num_clients = 10
        args.lr = 0.001
        args.num_epochs = 1
        args.num_rounds = 100
        args.batch_size = 128
        args.weight_decay = 5e-4
        args.dropout = 0.5
        args.optim = "adam"
        args.fl_algorithm = "gfedala"
        args.model = ["gin"]
        args.metrics = ["accuracy"]
        
        # W&B settings
        args.use_wandb = True
        args.wandb_project = "pFGL"
        args.wandb_entity = "scalable-group2"
        
        # Grid search hyperparameters
        args.lambda_graph = lambda_graph
        args.gala_temperature = gala_temperature
        args.gala_warmup_rounds = gala_warmup_rounds
        
        # NO SEED SET - each run is truly random
        # (removed: args.seed, torch.manual_seed, etc.)
        
        # Train
        trainer = FGLTrainer(args)
        trainer.train()
        
        # Extract final accuracy from trainer.evaluation_result
        # For graph_cls with local_model_on_local_data evaluation
        accuracy = None
        if hasattr(trainer, 'evaluation_result'):
            # Best test accuracy (selected by best validation)
            accuracy = trainer.evaluation_result.get("best_test_accuracy", None)
        
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "lambda_graph": lambda_graph,
            "gala_temperature": gala_temperature,
            "gala_warmup_rounds": gala_warmup_rounds,
            "run_id": run_id,
            "status": "success",
            "accuracy": accuracy,
            "best_val_accuracy": trainer.evaluation_result.get("best_val_accuracy", None),
            "best_round": trainer.evaluation_result.get("best_round", None),
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "lambda_graph": lambda_graph,
            "gala_temperature": gala_temperature,
            "gala_warmup_rounds": gala_warmup_rounds,
            "run_id": run_id,
            "status": "error",
            "error": str(e),
        }


def aggregate_results(results, dataset):
    """
    Aggregate results across runs for each configuration.
    Returns mean ± std for each config.
    """
    config_results = defaultdict(list)
    
    for r in results:
        if r.get("status") != "success" or r.get("accuracy") is None:
            continue
        
        config_key = (
            r["lambda_graph"],
            r["gala_temperature"],
            r["gala_warmup_rounds"]
        )
        config_results[config_key].append(r["accuracy"])
    
    summary = []
    for config_key, accuracies in config_results.items():
        lg, temp, warm = config_key
        acc_array = np.array(accuracies)
        
        summary.append({
            "lambda_graph": lg,
            "gala_temperature": temp,
            "gala_warmup_rounds": warm,
            "dataset": dataset,
            "mean_accuracy": float(acc_array.mean()),
            "std_accuracy": float(acc_array.std()),
            "n_runs": len(accuracies),
            "accuracies": accuracies,
        })
    
    summary.sort(key=lambda x: x["mean_accuracy"], reverse=True)
    return summary


def print_best_configs(summary, dataset, top_k=5):
    """Print the top-k best configurations."""
    print(f"\n{'='*70}")
    print(f"TOP {top_k} CONFIGURATIONS FOR {dataset}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'λ_graph':<10} {'Temp':<10} {'Warmup':<10} {'Accuracy':<20} {'Runs':<5}")
    print(f"{'-'*70}")
    
    for i, cfg in enumerate(summary[:top_k]):
        acc_str = f"{cfg['mean_accuracy']:.4f} ± {cfg['std_accuracy']:.4f}"
        print(f"{i+1:<5} {cfg['lambda_graph']:<10} {cfg['gala_temperature']:<10} "
              f"{cfg['gala_warmup_rounds']:<10} {acc_str:<20} {cfg['n_runs']:<5}")
    
    print(f"{'='*70}")


def run_grid_search(datasets, dry_run=False, output_dir="grid_search_results"):
    """Run full grid search."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset}")
        print(f"# Total runs: {len(CONFIGS) * NUM_RUNS} (no fixed seeds)")
        print(f"{'#'*70}")
        
        results_file = os.path.join(output_dir, f"grid_search_{dataset}_{timestamp}.json")
        all_results = []
        
        total_runs = len(CONFIGS) * NUM_RUNS
        run_idx = 0
        
        for i, (lg, temp, warm) in enumerate(CONFIGS):
            for run_id in range(1, NUM_RUNS + 1):
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] Config {i+1}/{len(CONFIGS)}, Run {run_id}/{NUM_RUNS}")
                
                result = run_single_experiment(
                    dataset=dataset,
                    lambda_graph=lg,
                    gala_temperature=temp,
                    gala_warmup_rounds=warm,
                    run_id=run_id,
                    dry_run=dry_run
                )
                all_results.append(result)
                
                # Save intermediate results
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"Status: {result['status']}")
        
        # Generate summary
        if not dry_run:
            summary = aggregate_results(all_results, dataset)
            summary_file = os.path.join(output_dir, f"summary_{dataset}_{timestamp}.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            print_best_configs(summary, dataset)
    
    print(f"\n{'='*60}")
    print(f"Grid search complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


def generate_latex_table(output_dir, datasets):
    """Generate LaTeX table from results."""
    print("\n% LaTeX Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{GFedALA Grid Search Results (mean $\\pm$ std over 3 runs, no fixed seeds)}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Dataset & $\\lambda_{graph}$ & Temperature & Warmup & Accuracy \\\\")
    print("\\midrule")
    
    for dataset in datasets:
        summary_files = [f for f in os.listdir(output_dir) if f.startswith(f"summary_{dataset}_")]
        if not summary_files:
            continue
        
        latest = sorted(summary_files)[-1]
        with open(os.path.join(output_dir, latest), "r") as f:
            summary = json.load(f)
        
        if summary:
            best = summary[0]
            print(f"{dataset} & {best['lambda_graph']} & {best['gala_temperature']} & "
                  f"{best['gala_warmup_rounds']} & "
                  f"${best['mean_accuracy']:.2f} \\pm {best['std_accuracy']:.2f}$ \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid search for GFedALA (no fixed seeds)")
    parser.add_argument("--dry_run", action="store_true", help="Preview without running")
    parser.add_argument("--dataset", type=str, default="PROTEINS",
                        help="Dataset(s), comma-separated (e.g., PROTEINS,MUTAG)")
    parser.add_argument("--output_dir", type=str, default="grid_search_results")
    parser.add_argument("--latex_table", action="store_true", help="Generate LaTeX table")
    parser.add_argument("--parse_results", type=str, help="Parse results file")
    
    args = parser.parse_args()
    datasets = [d.strip() for d in args.dataset.split(",")]
    
    if args.latex_table:
        generate_latex_table(args.output_dir, datasets)
    elif args.parse_results:
        with open(args.parse_results, "r") as f:
            results = json.load(f)
        dataset = results[0].get("dataset", "unknown") if results else "unknown"
        summary = aggregate_results(results, dataset)
        print_best_configs(summary, dataset, top_k=10)
    else:
        print(f"\n{'#'*70}")
        print(f"# GFedALA GRID SEARCH (No Fixed Seeds)")
        print(f"# Datasets: {datasets}")
        print(f"# Configs: {len(CONFIGS)}")
        print(f"# Runs per config: {NUM_RUNS}")
        print(f"# Total runs: {len(datasets) * len(CONFIGS) * NUM_RUNS}")
        print(f"{'#'*70}")
        
        run_grid_search(datasets=datasets, dry_run=args.dry_run, output_dir=args.output_dir)
