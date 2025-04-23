#!/bin/bash
#SBATCH --job-name=modula
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-159    # <<< set automatically to number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:1      # Request 1 GPU per task
##SBATCH --nodelist=isola-2080ti-[1-4],isola-v100-[1-2],isola-titanrtx-1,isola-ada6000-1
#SBATCH --nodelist=isola-2080ti-[1-4],isola-v100-[1-2],sitzmann-4090-1,oliva-titanrtx-[1,3],freeman-titanrtx-[1-2],agrawal-v100-1,agrawal-2080ti-1,freeman-v100-[1-2],isola-titanrtx-1,isola-ada6000-1,improbablex00[1-9],huang-l40s-[1-2],beery-l40s-[1-4],sitzmann-l40s-1#,agrawal-l40s-1,beery-a100-[1-2]
##SBATCH --nodelist=huang-l40s-[1-2],beery-l40s-[1-4],sitzmann-l40s-1,agrawal-l40s-1,beery-a100-[1-2]
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --account=vision-phillipi
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
    idx=$((i * jobs_in_parallel + j))
    echo "python train.py --job_idx $idx"
    python train.py --job_idx $idx &
    sleep 1
done
wait