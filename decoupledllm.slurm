#!/bin/bash
#SBATCH --job-name=distributed         # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8           # Number of MPI tasks per node (1 per GPU)
#SBATCH --gres=gpu:8                   # Number of GPUs per node
#SBATCH --cpus-per-task=6             # Number of CPUs per task (based on GPU ratio)
#SBATCH --hint=nomultithread          # Disable hyperthreading
#SBATCH --time=00:30:00               # Max execution time (HH:MM:SS)
#SBATCH --output=logs/job_%j.out      # Standard output log
#SBATCH --error=logs/job_%j.err       # Standard error log

# Clean environment
module purge

# Print each command before executing
set -x

# Launch training script
srun python -u main.py train=acco-ft data=alpaca model=llama3
