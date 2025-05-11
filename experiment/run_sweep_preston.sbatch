#!/bin/bash
#SBATCH --job-name=sweep_preston
#SBATCH --output=outLogs/sweep_%A_%a.out
#SBATCH --error=outLogs/sweep_%A_%a.err
#SBATCH --array=0-2    # <<< set automatically to number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:a100-mcdermott:1      # Request 1 GPU per task
####SBATCH --constraint=40GB
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH -p mcdermott

export PYTHONUNBUFFERED=1

nvidia-smi

# Create necessary directories
mkdir -p logs
mkdir -p results

# Activate venv environment
source ../.modula/bin/activate

# Number of jobs to run in parallel
jobs_in_parallel=16

export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "0.9/$jobs_in_parallel" | bc -l)

# Get the parameters for this job from the parameter grid
export i=$(($SLURM_ARRAY_TASK_ID))

echo "Running $jobs_in_parallel jobs in parallel:"

for j in $(seq 0 $(($jobs_in_parallel-1))); do
    idx=$((i * jobs_in_parallel + j))
    echo "python train.py --job_idx $idx"
    python train.py --job_idx $idx --sweep_config_path sweep_configs/sweep_specham_2/parameter_grid.json &
    sleep 1
done
wait
