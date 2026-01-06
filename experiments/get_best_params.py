"""
Analyze Grid Search Results Across All Datasets

Reads all summary files and finds:
1. Best config per dataset
2. Global best config by mean rank
3. Global best config by mean accuracy

Usage:
    python analyze_grid_search.py
    python analyze_grid_search.py --input_dir grid_search_results
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np


def load_all_summaries(input_dir):
    """Load all summary files from the input directory."""
    summaries = {}
    
    for filename in os.listdir(input_dir):
        if filename.startswith("summary_") and filename.endswith(".json"):
            # Extract dataset name: summary_PROTEINS_20240101_120000.json -> PROTEINS
            parts = filename.replace("summary_", "").replace(".json", "").rsplit("_", 2)
            dataset = parts[0]
            
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # Keep only the latest summary per dataset
            if dataset not in summaries or filename > summaries[dataset]["filename"]:
                summaries[dataset] = {
                    "filename": filename,
                    "data": data
                }
    
    return {k: v["data"] for k, v in summaries.items()}


def get_config_key(config):
    """Create a hashable key from config."""
    return (config["lambda_graph"], config["gala_temperature"], config["gala_warmup_rounds"])


def analyze_per_dataset_best(summaries):
    """Find the best config for each dataset."""
    print("\n" + "="*70)
    print("PER-DATASET BEST CONFIGURATIONS")
    print("="*70)
    print(f"{'Dataset':<15} {'λ_graph':<10} {'Temp':<10} {'Warmup':<10} {'Accuracy':<20}")
    print("-"*70)
    
    best_configs = {}
    for dataset, configs in sorted(summaries.items()):
        if not configs:
            continue
        best = configs[0]  # Already sorted by accuracy descending
        best_configs[dataset] = best
        acc_str = f"{best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}"
        print(f"{dataset:<15} {best['lambda_graph']:<10} {best['gala_temperature']:<10} "
              f"{best['gala_warmup_rounds']:<10} {acc_str:<20}")
    
    print("="*70)
    return best_configs


def analyze_global_by_mean_rank(summaries):
    """Find the best global config by mean rank across datasets."""
    print("\n" + "="*70)
    print("GLOBAL BEST BY MEAN RANK")
    print("="*70)
    
    # Collect all unique configs
    all_configs = set()
    for dataset, configs in summaries.items():
        for cfg in configs:
            all_configs.add(get_config_key(cfg))
    
    # Calculate rank for each config in each dataset
    config_ranks = defaultdict(list)
    config_accs = defaultdict(dict)
    
    for dataset, configs in summaries.items():
        for rank, cfg in enumerate(configs, 1):
            key = get_config_key(cfg)
            config_ranks[key].append(rank)
            config_accs[key][dataset] = {
                "accuracy": cfg["mean_accuracy"],
                "std": cfg["std_accuracy"],
                "rank": rank
            }
    
    # Calculate mean rank
    config_mean_ranks = []
    for key, ranks in config_ranks.items():
        mean_rank = np.mean(ranks)
        config_mean_ranks.append((key, mean_rank, ranks))
    
    # Sort by mean rank (lower is better)
    config_mean_ranks.sort(key=lambda x: x[1])
    
    # Print top 5
    print(f"\n{'Rank':<5} {'λ_graph':<10} {'Temp':<10} {'Warmup':<10} {'Mean Rank':<12} {'Ranks per Dataset'}")
    print("-"*80)
    
    for i, (key, mean_rank, ranks) in enumerate(config_mean_ranks[:5], 1):
        lg, temp, warm = key
        ranks_str = str(ranks)
        print(f"{i:<5} {lg:<10} {temp:<10} {warm:<10} {mean_rank:<12.2f} {ranks_str}")
    
    # Show details of best config
    if config_mean_ranks:
        best_key = config_mean_ranks[0][0]
        lg, temp, warm = best_key
        print(f"\n>>> BEST GLOBAL CONFIG (by mean rank): λ={lg}, T={temp}, W={warm}")
        print(f"\nPerformance per dataset:")
        print(f"{'Dataset':<15} {'Accuracy':<20} {'Rank':<10}")
        print("-"*50)
        for dataset, info in sorted(config_accs[best_key].items()):
            acc_str = f"{info['accuracy']:.4f} ± {info['std']:.4f}"
            print(f"{dataset:<15} {acc_str:<20} {info['rank']:<10}")
    
    print("="*70)
    return config_mean_ranks[0] if config_mean_ranks else None


def analyze_global_by_mean_accuracy(summaries):
    """Find the best global config by mean accuracy across datasets."""
    print("\n" + "="*70)
    print("GLOBAL BEST BY MEAN ACCURACY")
    print("="*70)
    
    # Collect accuracies for each config across datasets
    config_accs = defaultdict(list)
    config_details = defaultdict(dict)
    
    for dataset, configs in summaries.items():
        for cfg in configs:
            key = get_config_key(cfg)
            config_accs[key].append(cfg["mean_accuracy"])
            config_details[key][dataset] = {
                "accuracy": cfg["mean_accuracy"],
                "std": cfg["std_accuracy"]
            }
    
    # Calculate mean accuracy across datasets
    config_mean_accs = []
    for key, accs in config_accs.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        config_mean_accs.append((key, mean_acc, std_acc, len(accs)))
    
    # Sort by mean accuracy (higher is better)
    config_mean_accs.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 5
    print(f"\n{'Rank':<5} {'λ_graph':<10} {'Temp':<10} {'Warmup':<10} {'Mean Acc':<15} {'Std':<10} {'N_datasets'}")
    print("-"*80)
    
    for i, (key, mean_acc, std_acc, n) in enumerate(config_mean_accs[:5], 1):
        lg, temp, warm = key
        print(f"{i:<5} {lg:<10} {temp:<10} {warm:<10} {mean_acc:<15.4f} {std_acc:<10.4f} {n}")
    
    # Show details of best config
    if config_mean_accs:
        best_key = config_mean_accs[0][0]
        lg, temp, warm = best_key
        print(f"\n>>> BEST GLOBAL CONFIG (by mean accuracy): λ={lg}, T={temp}, W={warm}")
        print(f"\nPerformance per dataset:")
        print(f"{'Dataset':<15} {'Accuracy':<20}")
        print("-"*40)
        for dataset, info in sorted(config_details[best_key].items()):
            acc_str = f"{info['accuracy']:.4f} ± {info['std']:.4f}"
            print(f"{dataset:<15} {acc_str:<20}")
    
    print("="*70)
    return config_mean_accs[0] if config_mean_accs else None


def generate_latex_tables(summaries, best_per_dataset, global_best_rank, global_best_acc):
    """Generate LaTeX tables for the paper."""
    print("\n" + "="*70)
    print("LATEX TABLES")
    print("="*70)
    
    # Table 1: Per-dataset best
    print("\n% Table 1: Per-dataset optimal hyperparameters")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Optimal hyperparameters per dataset}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Dataset & $\\lambda_{graph}$ & $\\tau$ & Warmup & Accuracy \\\\")
    print("\\midrule")
    
    for dataset, cfg in sorted(best_per_dataset.items()):
        acc_str = f"${cfg['mean_accuracy']*100:.2f} \\pm {cfg['std_accuracy']*100:.2f}$"
        print(f"{dataset} & {cfg['lambda_graph']} & {cfg['gala_temperature']} & "
              f"{cfg['gala_warmup_rounds']} & {acc_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Table 2: Global config comparison
    if global_best_rank and global_best_acc:
        print("\n% Table 2: Global hyperparameter performance")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Global hyperparameter configurations}")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("Selection Method & $\\lambda_{graph}$ & $\\tau$ & Warmup \\\\")
        print("\\midrule")
        
        lg, temp, warm = global_best_rank[0]
        print(f"Best by Mean Rank & {lg} & {temp} & {warm} \\\\")
        
        lg, temp, warm = global_best_acc[0]
        print(f"Best by Mean Accuracy & {lg} & {temp} & {warm} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--input_dir", type=str, default="grid_search_results",
                        help="Directory containing summary files")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    
    args = parser.parse_args()
    
    # Load all summaries
    print(f"Loading summaries from: {args.input_dir}")
    summaries = load_all_summaries(args.input_dir)
    
    if not summaries:
        print("No summary files found!")
        return
    
    print(f"Found {len(summaries)} datasets: {list(summaries.keys())}")
    
    # Analyze
    best_per_dataset = analyze_per_dataset_best(summaries)
    global_best_rank = analyze_global_by_mean_rank(summaries)
    global_best_acc = analyze_global_by_mean_accuracy(summaries)
    
    # LaTeX tables
    if args.latex:
        generate_latex_tables(summaries, best_per_dataset, global_best_rank, global_best_acc)


if __name__ == "__main__":
    main()