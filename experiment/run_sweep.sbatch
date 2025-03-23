#!/bin/bash
#SBATCH --job-name=manifold
#SBATCH --output=/data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/logs/sweep_%A_%a.out
#SBATCH --error=/data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/logs/sweep_%A_%a.err
#SBATCH --array=0-47      # <<< set this to the number of parameter combinations from generate_sweeps.py
#SBATCH --gres=gpu:1       # Request 1 GPU per task
#SBATCH --nodelist=isola-2080ti-[1-4],isola-v100-[1-2],sitzmann-4090-1,oliva-titanrtx-[1,3],freeman-titanrtx-[1-2],agrawal-v100-1,agrawal-2080ti-1,freeman-v100-[1-2],isola-titanrtx-1,isola-ada6000-1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free

export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p /data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/logs
mkdir -p /data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/results

# Activate venv environment
source /data/vision/phillipi/vector/duality/spring2025/.spring/bin/activate

# Get the parameters for this job from the parameter grid
export i=$SLURM_ARRAY_TASK_ID
params=$(python -c "
import json
with open('/data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/sweep_configs/parameter_grid.json', 'r') as f:
    params = json.load(f)
    print(
        f'--d_embed {params[$i][\"d_embed\"]} '
        f'--lr {params[$i][\"lr\"]} '
        f'--wd {params[$i][\"wd\"]} '
        f'--blocks {params[$i][\"blocks\"]} '
        f'--seq_len {params[$i][\"seq_len\"]} '
        f'--num_heads {params[$i][\"num_heads\"]} '
        f'--softmax_scale {params[$i][\"softmax_scale\"]} '
        f'--final_scale {params[$i][\"final_scale\"]} '
        f'--optimizer {params[$i][\"optimizer\"]} '
        f'--pre_dualize {params[$i][\"pre_dualize\"]} '
        f'--post_dualize {params[$i][\"post_dualize\"]} '
        f'--beta1 {params[$i][\"beta1\"]} '
        f'--beta2 {params[$i][\"beta2\"]} '
        f'--batch_size {params[$i][\"batch_size\"]} '
        f'--project {params[$i][\"project\"]} '
        f'--manifold {params[$i][\"manifold\"]} '
        f'--steps {params[$i][\"steps\"]} '
        f'--data {params[$i][\"data\"]} '
        f'--output-dir {params[$i][\"output_dir\"]} '
    )
")

# Run the training script with these parameters
echo $params
python /data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment/train.py $params