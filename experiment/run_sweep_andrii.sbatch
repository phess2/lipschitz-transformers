#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=1-576       # <<< set automatically to number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:1      # Request 1 GPU per task
#SBATCH --constraint=14GB # Avoid the buggy 1080Ti on Andrii's cluster
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1

nvidia-smi

# Create necessary directories
mkdir -p logs
mkdir -p results

# Activate venv environment
source ../.modula/bin/activate

# Number of jobs to run in parallel
jobs_in_parallel=1

# Get the parameters for this job from the parameter grid
export i=$SLURM_ARRAY_TASK_ID-1

echo "Running $jobs_in_parallel jobs in parallel:"

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
    echo "Job $((j+1)): $params"
    python train.py $params &
done

wait
