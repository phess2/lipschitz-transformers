#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-23     # <<< set automatically to number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:1      # Request 1 GPU per task
##SBATCH --nodelist=isola-h100-1
#SBATCH --nodelist=isola-2080ti-[1-4],isola-v100-[1-2],sitzmann-4090-1,oliva-titanrtx-[1,3],freeman-titanrtx-[1-2],agrawal-v100-1,agrawal-2080ti-1,freeman-v100-[1-2],isola-titanrtx-1,isola-ada6000-1,improbablex[001-009]
##SBATCH --nodelist=isola-2080ti-[1-4],isola-v100-[1-2],isola-titanrtx-1,isola-ada6000-1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
##SBATCH --partition=vision-phillipi
##SBATCH --qos=vision-phillipi-main
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free

export PYTHONUNBUFFERED=1

nvidia-smi

# Create necessary directories
mkdir -p logs
mkdir -p results

# Activate venv environment
source ../.modula/bin/activate

# Number of jobs to run in parallel
jobs_in_parallel=1
echo "Running $jobs_in_parallel jobs:"

# Get the parameters for this job from the parameter grid
export i=$SLURM_ARRAY_TASK_ID

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
    echo ""
    echo "--------------------------------"
    echo "Job $((j+1)): $params"
    python train.py $params &
    wait
done