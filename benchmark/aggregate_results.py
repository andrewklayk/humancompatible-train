"""
Aggregate individual benchmark run results into summary CSVs.

This script collects results from parallel multirun sweeps and produces:
1. Summary statistics (mean/std across seeds) per hyperparameter combo
2. Best hyperparameters per algorithm
3. Comparative analysis across algorithms

Usage:
    python aggregate_results.py results/
    python aggregate_results.py results/ --output summary_report.csv
"""

import pandas as pd
import numpy as np
import json
import glob
import os
from pathlib import Path
import argparse
from collections import defaultdict


def load_run_results(results_dir: str):
    """Load all run results from results directory structure."""
    
    results_dir = Path(results_dir)
    runs_by_algo = defaultdict(list)
    
    # Iterate through algorithm subdirectories
    for algo_dir in results_dir.glob("*"):
        if not algo_dir.is_dir():
            continue
            
        algorithm = algo_dir.name
        
        # Load all runs for this algorithm
        config_files = list(algo_dir.glob("*_config.json"))
        
        for config_file in config_files:
            run_id = config_file.stem.replace("_config", "")
            val_file = algo_dir / f"{run_id}_val.csv"
            train_file = algo_dir / f"{run_id}_train.csv"
            
            if not val_file.exists():
                print(f"Warning: Missing validation file for {run_id}")
                continue
            
            try:
                with open(config_file, 'r') as f:
                    params = json.load(f)
                
                val_df = pd.read_csv(val_file, index_col=[0, 1])
                train_df = pd.read_csv(train_file, index_col=[0, 1])
                
                runs_by_algo[algorithm].append({
                    'run_id': run_id,
                    'params': params,
                    'val_df': val_df,
                    'train_df': train_df
                })
            except Exception as e:
                print(f"Error loading {run_id}: {e}")
                continue
    
    return runs_by_algo


def aggregate_by_params(runs):
    """Aggregate runs by hyperparameter combination."""
    
    param_groups = defaultdict(list)
    
    for run in runs:
        # Create param key (excluding seed)
        params = run['params'].copy()
        seed = params.pop('seed', None)
        param_key = json.dumps(params, sort_keys=True)
        
        param_groups[param_key].append({
            'seed': seed,
            'val_df': run['val_df'],
            'train_df': run['train_df']
        })
    
    return param_groups


def compute_summary_stats(param_groups):
    """Compute mean/std across seeds for each parameter combination."""
    
    summaries = []
    
    for param_key, group_runs in param_groups.items():
        params = json.loads(param_key)
        
        # Extract metrics across seeds
        final_losses = []
        min_losses = []
        best_epochs = []
        constraint_violations = []
        
        for run_data in group_runs:
            val_df = run_data['val_df']
            
            # Get final epoch metrics
            final_metrics = val_df.iloc[-1]
            final_losses.append(final_metrics['loss'])
            min_losses.append(val_df['loss'].min())
            best_epochs.append(val_df['loss'].argmin())
            
            # Get max constraint violation if available
            c_cols = [col for col in val_df.columns if col.startswith('c_')]
            if c_cols:
                max_c = val_df[c_cols].abs().max().max()
                constraint_violations.append(max_c)
        
        summary = params.copy()
        summary['n_seeds'] = len(group_runs)
        summary['final_loss_mean'] = np.mean(final_losses)
        summary['final_loss_std'] = np.std(final_losses)
        summary['min_loss_mean'] = np.mean(min_losses)
        summary['min_loss_std'] = np.std(min_losses)
        summary['best_epoch_mean'] = np.mean(best_epochs)
        summary['best_epoch_std'] = np.std(best_epochs)
        
        if constraint_violations:
            summary['max_constraint_mean'] = np.mean(constraint_violations)
            summary['max_constraint_std'] = np.std(constraint_violations)
        
        summaries.append(summary)
    
    return pd.DataFrame(summaries)


def process_algorithm(algorithm: str, runs: list, output_dir: str):
    """Process results for a single algorithm."""
    
    # Aggregate by parameters
    param_groups = aggregate_by_params(runs)
    
    # Compute summary statistics
    summary_df = compute_summary_stats(param_groups)
    
    # Sort by min_loss_mean
    summary_df = summary_df.sort_values('min_loss_mean')
    
    # Save summary
    output_file = os.path.join(output_dir, f"summary_{algorithm}.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary for {algorithm}: {output_file}")
    
    # Find and report best params
    best_idx = summary_df['min_loss_mean'].idxmin()
    best_params = summary_df.loc[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm}")
    print(f"{'='*60}")
    print(f"Best hyperparameters:")
    for col in summary_df.columns:
        if col not in ['n_seeds', 'best_epoch_mean', 'best_epoch_std']:
            print(f"  {col}: {best_params[col]}")
    print(f"\nBest min_loss: {best_params['min_loss_mean']:.6f} ± {best_params['min_loss_std']:.6f}")
    if 'max_constraint_mean' in best_params:
        print(f"Max constraint: {best_params['max_constraint_mean']:.6f} ± {best_params['max_constraint_std']:.6f}")
    print()
    
    return summary_df


def generate_comparison_report(summaries_by_algo: dict, output_dir: str):
    """Generate cross-algorithm comparison report."""
    
    comparison_data = []
    
    for algorithm, summary_df in summaries_by_algo.items():
        best_idx = summary_df['min_loss_mean'].idxmin()
        best_row = summary_df.loc[best_idx].to_dict()
        best_row['algorithm'] = algorithm
        comparison_data.append(best_row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = os.path.join(output_dir, 'algorithm_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\n{'='*60}")
    print("ALGORITHM COMPARISON (Best Results)")
    print(f"{'='*60}")
    print(comparison_df[['algorithm', 'min_loss_mean', 'min_loss_std', 'max_constraint_mean']].to_string(index=False))
    print(f"\nComparison report saved: {comparison_file}")
    
    return comparison_df


def main(results_dir: str, output_dir: str = None):
    """Main aggregation pipeline."""
    
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from {results_dir}...")
    runs_by_algo = load_run_results(results_dir)
    
    if not runs_by_algo:
        print("No results found!")
        return
    
    print(f"Found {sum(len(v) for v in runs_by_algo.values())} total runs across {len(runs_by_algo)} algorithms\n")
    
    # Process each algorithm
    summaries_by_algo = {}
    for algorithm, runs in runs_by_algo.items():
        summary_df = process_algorithm(algorithm, runs, output_dir)
        summaries_by_algo[algorithm] = summary_df
    
    # Generate comparison report
    generate_comparison_report(summaries_by_algo, output_dir)
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate parallel gridsearch results into summary CSVs"
    )
    parser.add_argument('results_dir', help='Directory containing algorithm subdirectories')
    parser.add_argument('--output', '-o', default=None, help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    main(args.results_dir, args.output)
