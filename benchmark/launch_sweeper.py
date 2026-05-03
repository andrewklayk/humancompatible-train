#!/usr/bin/env python3
"""
Hydra multirun launcher for parallel gridsearch benchmarks.

This script provides convenient commands for running hyperparameter sweeps
with different backends (joblib, Ray, SLURM).

Usage:
    # Local multirun with joblib (N_JOBS=-1 uses all cores)
    python launch_sweeper.sh alm_max --backend joblib --n-jobs -1

    # Single algorithm sweep (all defaults + specific seed)
    python launch_sweeper.sh alm_max --seeds 0,1,2

    # Multi-algorithm comparison
    python launch_sweeper.sh all --backend joblib --n-jobs 4

    # Ray cluster sweep
    python launch_sweeper.sh alm_max --backend ray --num-workers 16

    # SLURM cluster
    python launch_sweeper.sh pbm_adapt --backend slurm --partition gpu --time 2:00:00
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Parameter ranges for each algorithm
PARAM_RANGES = {
    'adam': {
        'seeds': '0,1,2',
        'params.lr': '0.001,0.005,0.01,0.05',
    },
    'alm_max': {
        'seeds': '0,1,2',
        'params.primal_lr': '0.001,0.005,0.01,0.05',
        'params.dual_lr': '0.001,0.005,0.01,0.05',
        'params.penalty': '0,1',
        'params.moreau_mu': '0,0.5,1,2,4',
    },
    'alm_slack': {
        'seeds': '0,1,2',
        'params.primal_lr': '0.001,0.005,0.01,0.05',
        'params.dual_lr': '0.001,0.005,0.01,0.05',
        'params.penalty': '0,1',
        'params.moreau_mu': '0,0.5,1,2,4',
    },
    'pbm_adapt': {
        'seeds': '0,1,2',
        'params.primal_lr': '0.001,0.005,0.01,0.05',
        'params.dual_penalty_mult': '0.9,0.99',
        'params.dual_gamma': '0.9,0.99',
        'params.moreau_mu': '0,1,2,4',
    },
    'pbm_dimin': {
        'seeds': '0,1,2',
        'params.primal_lr': '0.001,0.005,0.01,0.05',
        'params.dual_penalty_mult': '1.0,0.999,0.99',
        'params.dual_gamma': '0.9,0.99',
        'params.moreau_mu': '0,0.5,1,2,4',
    },
    'ssg': {
        'seeds': '0,1,2',
        'params.primal_lr': '0.001,0.005,0.01,0.05',
        'params.dual_lr': '0.001,0.005,0.01,0.05',
        'params.moreau_mu': '0,2',
    },
}

ALGORITHMS = list(PARAM_RANGES.keys())


def build_multirun_command(
    algorithms,
    seeds=None,
    backend='joblib',
    n_jobs=None,
    num_workers=None,
    partition='gpu',
    time_limit='4:00:00',
    task_config=None,
    data_config=None,
    custom_params=None
):
    """Build the Hydra multirun command."""
    
    cmd = ['python', 'run_single_experiment.py', '--multirun']
    
    # Add algorithms
    if algorithms == 'all':
        algo_list = ALGORITHMS
    else:
        algo_list = [algorithms] if isinstance(algorithms, str) else algorithms
    
    # Build sweep parameters
    sweep_params = []
    
    # Handle algorithms differently - need to use defaults system
    if len(algo_list) == 1:
        cmd.append(f"--config-name=algorithm/{algo_list[0]}")
    
    for algo in algo_list:
        params = PARAM_RANGES.get(algo, {})
        if seeds:
            params['seed'] = seeds
        sweep_params.append(params)
    
    # If multiple algorithms, merge all sweep params
    if len(algo_list) > 1:
        all_sweep_params = {}
        for algo in algo_list:
            params = PARAM_RANGES.get(algo, {})
            if seeds:
                params['seed'] = seeds
            all_sweep_params.update(params)
        sweep_params = [all_sweep_params]
    else:
        sweep_params = [PARAM_RANGES.get(algo_list[0], {})]
    
    # Add seed override if provided
    if seeds:
        cmd.append(f'seed={seeds}')
    
    # Add other parameters
    for algo in algo_list:
        params = PARAM_RANGES[algo]
        for param_name, param_values in params.items():
            if param_name != 'seeds':  # Skip seeds, handled separately
                cmd.append(f'{param_name}={param_values}')
    
    # Custom parameters
    if custom_params:
        for k, v in custom_params.items():
            cmd.append(f'{k}={v}')
    
    # Task and data config
    if task_config:
        cmd.append(f'task={task_config}')
    if data_config:
        cmd.append(f'data={data_config}')
    
    # Launcher configuration
    if backend == 'joblib':
        cmd.append('hydra/launcher=joblib')
        if n_jobs is not None:
            cmd.append(f'hydra.launcher.n_jobs={n_jobs}')
        else:
            cmd.append('hydra.launcher.n_jobs=-1')  # Use all cores by default
    
    elif backend == 'ray':
        cmd.append('hydra/launcher=ray')
        if num_workers:
            cmd.append(f'hydra.launcher.num_workers={num_workers}')
    
    elif backend == 'slurm':
        cmd.append('hydra/launcher=submitit_slurm')
        cmd.append(f'hydra.launcher.partition={partition}')
        cmd.append(f'hydra.launcher.timeout_min={int(float(time_limit.split(":")[0]) * 60)}')
    
    # Reduce logging verbosity
    cmd.append('hydra.job_logging=none')
    
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description='Launch parallel Hydra gridsearch benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Local joblib (all cores)
  python launch_sweeper.sh alm_max --backend joblib

  # Specific seeds
  python launch_sweeper.sh pbm_adapt --seeds 0,1,2

  # All algorithms
  python launch_sweeper.sh all --backend joblib --n-jobs 8

  # Ray cluster
  python launch_sweeper.sh alm_max --backend ray --num-workers 16

  # SLURM (requires submitit)
  python launch_sweeper.sh ssg --backend slurm --partition gpu --time 4:00:00

Available algorithms: {', '.join(ALGORITHMS)}
        """
    )
    
    parser.add_argument(
        'algorithm',
        choices=['all'] + ALGORITHMS,
        help='Algorithm to sweep or "all" for all algorithms'
    )
    parser.add_argument(
        '--backend',
        choices=['joblib', 'ray', 'slurm'],
        default='joblib',
        help='Launcher backend'
    )
    parser.add_argument(
        '--seeds',
        default=None,
        help='Override default seeds (e.g., "0,1,2,3,4")'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel jobs (joblib only, -1=all cores)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of workers (ray only)'
    )
    parser.add_argument(
        '--partition',
        default='gpu',
        help='SLURM partition (slurm only)'
    )
    parser.add_argument(
        '--time',
        default='4:00:00',
        help='Time limit HH:MM:SS (slurm only)'
    )
    parser.add_argument(
        '--task',
        default=None,
        help='Task config (e.g., "dutch_positive_rate_pair")'
    )
    parser.add_argument(
        '--data',
        default=None,
        help='Data config (e.g., "dutch")'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print command without executing'
    )
    
    args = parser.parse_args()
    
    # Build command
    cmd = build_multirun_command(
        algorithms=args.algorithm,
        seeds=args.seeds,
        backend=args.backend,
        n_jobs=args.n_jobs,
        num_workers=args.num_workers,
        partition=args.partition,
        time_limit=args.time,
        task_config=args.task,
        data_config=args.data,
    )
    
    # Print command
    print("Running command:")
    print(" \\\n  ".join(cmd))
    print()
    
    if args.dry_run:
        print("[DRY RUN - command not executed]")
        return 0
    
    # Execute
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return e.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
