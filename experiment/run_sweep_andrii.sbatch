#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=1-126      # <<< set automatically to number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:a100:1      # Request 1 GPU per task
####SBATCH --constraint=40GB # Avoid the buggy 1080Ti on Andrii's cluster (14GB is enough for this) + remove 24GB GPUs because I am running too many jobs in parallel for them.
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1

nvidia-smi

# Create necessary directories
mkdir -p logs
mkdir -p results

# Activate venv environment
source ../.modula/bin/activate

# Number of jobs to run in parallel
jobs_in_parallel=12

export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "0.9/$jobs_in_parallel" | bc -l)

# Get the parameters for this job from the parameter grid
export i=$(($SLURM_ARRAY_TASK_ID-1))

echo "Running $jobs_in_parallel jobs in parallel:"

echo "Slurm job index: $SLURM_ARRAY_TASK_ID"
for j in $(seq 0 $(($jobs_in_parallel-1))); do
    params=$(python -c "
idx = $i * $jobs_in_parallel + $j
print(idx)
")
    echo "Job index $j: $params"
done

# Launch n_in_parallel jobs
for j in $(seq 0 $(($jobs_in_parallel-1))); do
    params=$(python -c "
import json
with open('sweep_configs/parameter_grid.json', 'r') as f:
    idx = $i * $jobs_in_parallel + $j
    params = json.load(f)[idx]
    print(
        ' '.join([
            f'--{key} {params[key]}' 
            for key in params    # iterate over all keys in the dictionary and pass them as arguments to the script
        ])
    )
")
    echo "Job $((j+1)), full index $(($i)) * $jobs_in_parallel + $j: $params"
    python train.py $params &
done

wait
