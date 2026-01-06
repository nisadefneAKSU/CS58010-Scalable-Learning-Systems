"""
Ablation Study for GFedALA

Measures the contribution of each component by removing one at a time
from the best configuration.

Ablation variants:
- Full GFedALA: Best tuned config
- w/o Graph Similarity: λ_graph=1.0 (only param distance)
- w/o Warmup: warmup=0 (no graph similarity in early rounds)
- w/o Sharp Weights: temperature=1.0 (nearly uniform aggregation)

Usage:
    python ablation_study.py --dataset PROTEINS
    python ablation_study.py --dataset PROTEINS,MUTAG,NCI1
    python ablation_study.py --dry_run
"""

import itertools
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import copy

import torch
import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


# ============================
# Configuration
# ============================
NUM_RUNS = 3  # Runs per config (no fixed seeds)

# Best config from grid search (UPDATE THESE with your tuned values)
BEST_CONFIG = {
    "lambda_graph": 0.5,
    "gala_temperature": 0.1,
    "gala_warmup_rounds": 5,
}

# Ablation variants
ABLATIONS = {
    "Full GFedALA": {
        "lambda_graph": BEST_CONFIG["lambda_graph"],
        "gala_temperature": BEST_CONFIG["gala_temperature"],
        "gala_warmup_rounds": BEST_CONFIG["gala_warmup_rounds"],
    },
    "w/o Graph Similarity": {
        "lambda_graph": 1.0,  # Only param distance
        "gala_temperature": BEST_CONFIG["gala_temperature"],
        "gala_warmup_rounds": 0,  # No warmup needed if no graph sim
    },
    "w/o Warmup": {
        "lambda_graph": BEST_CONFIG["lambda_graph"],
        "gala_temperature": BEST_CONFIG["gala_temperature"],
        "gala_warmup_rounds": 0,  # Disable warmup
    },
    "w/o Sharp Weights": {
        "lambda_graph": BEST_CONFIG["lambda_graph"],
        "gala_temperature": 1.0,  # Nearly uniform weights
        "gala_warmup_rounds": BEST_CONFIG["gala_warmup_rounds"],
    },
}

print(f"Ablation variants: {len(ABLATIONS)}")
print(f"Runs per variant: {NUM_RUNS}")
for name, cfg in ABLATIONS.items():
    print(f"  {name}: λ={cfg['lambda_graph']}, T={cfg['gala_temperature']}, W={cfg['gala_warmup_rounds']}")


# ============================
# Patch torch.load for PyTorch 2.6+
# ============================
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load


def run_single_experiment(dataset, ablation_name, ablation_config, run_id, dry_run=False):
    """
    Run a single ablation experiment.
    NO SEED IS SET - true randomness for each run.
    """
    exp_name = f"ablation_{ablation_name.replace(' ', '_').replace('/', '_')}_run{run_id}"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Dataset: {dataset}")
    print(f"Config: λ={ablation_config['lambda_graph']}, T={ablation_config['gala_temperature']}, W={ablation_config['gala_warmup_rounds']}")
    print(f"{'='*60}")
    
    if dry_run:
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "ablation_name": ablation_name,
            "lambda_graph": ablation_config["lambda_graph"],
            "gala_temperature": ablation_config["gala_temperature"],
            "gala_warmup_rounds": ablation_config["gala_warmup_rounds"],
            "run_id": run_id,
            "status": "dry_run",
        }
    
    try:
        # Get fresh args - deep copy to avoid state pollution
        args = copy.deepcopy(config.args)
        
        # Fixed settings
        args.root = "/home/ceren/Desktop/scalable/OpenFGL-main/data"
        args.scenario = "graph_fl"
        args.task = "graph_cls"
        args.dataset = [dataset]  # MUST be a list
        print(f"Using dataset: {args.dataset}")
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
        args.wandb_project = "pFGL-ablation"
        args.wandb_entity = "scalable-group2"
        
        # Ablation hyperparameters
        args.lambda_graph = ablation_config["lambda_graph"]
        args.gala_temperature = ablation_config["gala_temperature"]
        args.gala_warmup_rounds = ablation_config["gala_warmup_rounds"]
        
        # NO SEED SET - true randomness
        
        # Train
        trainer = FGLTrainer(args)
        trainer.train()
        
        # Extract accuracy
        accuracy = None
        if hasattr(trainer, 'evaluation_result'):
            accuracy = trainer.evaluation_result.get("best_test_accuracy", None)
        
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "ablation_name": ablation_name,
            "lambda_graph": ablation_config["lambda_graph"],
            "gala_temperature": ablation_config["gala_temperature"],
            "gala_warmup_rounds": ablation_config["gala_warmup_rounds"],
            "run_id": run_id,
            "status": "success",
            "accuracy": accuracy,
            "best_val_accuracy": trainer.evaluation_result.get("best_val_accuracy", None),
            "best_round": trainer.evaluation_result.get("best_round", None),
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "exp_name": exp_name,
            "dataset": dataset,
            "ablation_name": ablation_name,
            "lambda_graph": ablation_config["lambda_graph"],
            "gala_temperature": ablation_config["gala_temperature"],
            "gala_warmup_rounds": ablation_config["gala_warmup_rounds"],
            "run_id": run_id,
            "status": "error",
            "error": str(e),
        }


def aggregate_results(results):
    """Aggregate results across runs for each ablation variant."""
    # Group by (dataset, ablation_name)
    grouped = defaultdict(list)
    
    for r in results:
        if r.get("status") != "success" or r.get("accuracy") is None:
            continue
        key = (r["dataset"], r["ablation_name"])
        grouped[key].append(r["accuracy"])
    
    summary = []
    for (dataset, ablation_name), accuracies in grouped.items():
        acc_array = np.array(accuracies)
        summary.append({
            "dataset": dataset,
            "ablation_name": ablation_name,
            "mean_accuracy": float(acc_array.mean()),
            "std_accuracy": float(acc_array.std()),
            "n_runs": len(accuracies),
            "accuracies": accuracies,
        })
    
    return summary


def print_ablation_table(summary, datasets):
    """Print ablation results as a table."""
    print(f"\n{'='*90}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*90}")
    
    # Build table data
    table_data = defaultdict(dict)
    for item in summary:
        table_data[item["ablation_name"]][item["dataset"]] = item
    
    # Header
    header = f"{'Method':<25}"
    for ds in datasets:
        header += f" {ds:<15}"
    header += f" {'Avg':<15}"
    print(header)
    print("-"*90)
    
    # Rows
    for ablation_name in ABLATIONS.keys():
        row = f"{ablation_name:<25}"
        accs = []
        for ds in datasets:
            if ds in table_data.get(ablation_name, {}):
                item = table_data[ablation_name][ds]
                acc_str = f"{item['mean_accuracy']:.4f}±{item['std_accuracy']:.4f}"
                accs.append(item['mean_accuracy'])
            else:
                acc_str = "N/A"
            row += f" {acc_str:<15}"
        
        # Average
        if accs:
            avg_acc = np.mean(accs)
            row += f" {avg_acc:.4f}"
        else:
            row += " N/A"
        
        print(row)
    
    print(f"{'='*90}")


def generate_latex_table(summary, datasets):
    """Generate LaTeX table for the paper."""
    print("\n% LaTeX Ablation Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation study results (mean $\\pm$ std over 3 runs)}")
    
    # Column spec
    col_spec = "l" + "c" * len(datasets) + "c"
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    
    # Header
    header = "Method"
    for ds in datasets:
        header += f" & {ds}"
    header += " & Avg \\\\"
    print(header)
    print("\\midrule")
    
    # Build table data
    table_data = defaultdict(dict)
    for item in summary:
        table_data[item["ablation_name"]][item["dataset"]] = item
    
    # Rows
    for ablation_name in ABLATIONS.keys():
        row = ablation_name.replace("w/o", "w/o").replace("_", "\\_")
        accs = []
        for ds in datasets:
            if ds in table_data.get(ablation_name, {}):
                item = table_data[ablation_name][ds]
                acc_str = f"${item['mean_accuracy']*100:.2f} \\pm {item['std_accuracy']*100:.2f}$"
                accs.append(item['mean_accuracy'])
            else:
                acc_str = "N/A"
            row += f" & {acc_str}"
        
        # Average
        if accs:
            avg_acc = np.mean(accs) * 100
            row += f" & ${avg_acc:.2f}$"
        else:
            row += " & N/A"
        
        row += " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def run_ablation_study(datasets, dry_run=False, output_dir="ablation_results"):
    """Run full ablation study."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
    all_results = []
    
    total_runs = len(datasets) * len(ABLATIONS) * NUM_RUNS
    run_idx = 0
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*70}")
        
        for ablation_name, ablation_config in ABLATIONS.items():
            for run_id in range(1, NUM_RUNS + 1):
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {dataset} - {ablation_name} - Run {run_id}/{NUM_RUNS}")
                
                result = run_single_experiment(
                    dataset=dataset,
                    ablation_name=ablation_name,
                    ablation_config=ablation_config,
                    run_id=run_id,
                    dry_run=dry_run
                )
                all_results.append(result)
                
                # Save intermediate results
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"Status: {result['status']}")
            
            # Checkpoint after each ablation variant
            if not dry_run:
                summary = aggregate_results(all_results)
                summary_file = os.path.join(output_dir, f"ablation_summary_{timestamp}.json")
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)
    
    # Final summary
    if not dry_run:
        summary = aggregate_results(all_results)
        print_ablation_table(summary, datasets)
        generate_latex_table(summary, datasets)
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


def load_best_config_from_grid_search(grid_search_dir="grid_search_results"):
    """Load best config from grid search results using analyze logic."""
    from get_best_params import load_all_summaries
    
    summaries = load_all_summaries(grid_search_dir)
    if not summaries:
        print("No grid search results found. Using default BEST_CONFIG.")
        return BEST_CONFIG
    
    # Find global best by mean accuracy
    config_accs = defaultdict(list)
    for dataset, configs in summaries.items():
        for cfg in configs:
            key = (cfg["lambda_graph"], cfg["gala_temperature"], cfg["gala_warmup_rounds"])
            config_accs[key].append(cfg["mean_accuracy"])
    
    best_key = max(config_accs.keys(), key=lambda k: np.mean(config_accs[k]))
    
    best_config = {
        "lambda_graph": best_key[0],
        "gala_temperature": best_key[1],
        "gala_warmup_rounds": best_key[2],
    }
    
    print(f"Loaded best config from grid search: {best_config}")
    return best_config


# ============================
# Main
# ============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ablation study for GFedALA")
    parser.add_argument("--dry_run", action="store_true", help="Preview without running")
    parser.add_argument("--dataset", type=str, default="PROTEINS",
                        help="Dataset(s), comma-separated")
    parser.add_argument("--output_dir", type=str, default="ablation_results")
    parser.add_argument("--auto_config", action="store_true",
                        help="Auto-load best config from grid search results")
    parser.add_argument("--grid_search_dir", type=str, default="grid_search_results",
                        help="Directory with grid search results (for --auto_config)")
    parser.add_argument("--parse_results", type=str, help="Parse existing results file")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table from results")
    
    args = parser.parse_args()
    datasets = [d.strip() for d in args.dataset.split(",")]
    
    if args.parse_results:
        with open(args.parse_results, "r") as f:
            results = json.load(f)
        summary = aggregate_results(results)
        parsed_datasets = list(set(r["dataset"] for r in results if r.get("dataset")))
        print_ablation_table(summary, parsed_datasets)
        if args.latex:
            generate_latex_table(summary, parsed_datasets)
    else:
        # Auto-load best config if requested
        if args.auto_config:
            try:
                best = load_best_config_from_grid_search(args.grid_search_dir)
                ABLATIONS["Full GFedALA"] = best
                ABLATIONS["w/o Graph Similarity"]["gala_temperature"] = best["gala_temperature"]
                ABLATIONS["w/o Warmup"]["lambda_graph"] = best["lambda_graph"]
                ABLATIONS["w/o Warmup"]["gala_temperature"] = best["gala_temperature"]
                ABLATIONS["w/o Sharp Weights"]["lambda_graph"] = best["lambda_graph"]
                ABLATIONS["w/o Sharp Weights"]["gala_warmup_rounds"] = best["gala_warmup_rounds"]
            except Exception as e:
                print(f"Could not load grid search results: {e}")
                print("Using default BEST_CONFIG.")
        
        print(f"\n{'#'*70}")
        print(f"# GFedALA ABLATION STUDY")
        print(f"# Datasets: {datasets}")
        print(f"# Ablation variants: {len(ABLATIONS)}")
        print(f"# Runs per variant: {NUM_RUNS}")
        print(f"# Total runs: {len(datasets) * len(ABLATIONS) * NUM_RUNS}")
        print(f"{'#'*70}")
        
        print("\nAblation configurations:")
        for name, cfg in ABLATIONS.items():
            print(f"  {name}: λ={cfg['lambda_graph']}, T={cfg['gala_temperature']}, W={cfg['gala_warmup_rounds']}")
        
        run_ablation_study(datasets=datasets, dry_run=args.dry_run, output_dir=args.output_dir)