#!/bin/bash

# Parameters
#SBATCH --array=0-99%100
#SBATCH --cpus-per-task=4
#SBATCH --error=/mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/multirun/cifar10/pbm/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=run
#SBATCH --mem-per-cpu=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/multirun/cifar10/pbm/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=h200fast
#SBATCH --signal=USR2@120
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/multirun/cifar10/pbm/.submitit/%A_%a/%A_%a_%t_log.out --error /mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/multirun/cifar10/pbm/.submitit/%A_%a/%A_%a_%t_log.err /mnt/appl/software/Python/3.13.5-GCCcore-14.3.0/bin/python3 -u -m submitit.core._submit /mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/multirun/cifar10/pbm/.submitit/%j
