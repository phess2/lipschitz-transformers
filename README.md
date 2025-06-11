# Modula Training Environment

This repository contains a modified version of the Modula library for training neural networks with various optimization techniques. This README will guide you through setting up the environment and running the training code.

## Setup Instructions

1. Create a new conda environment:
```bash
conda create -n modula python=3.9
conda activate modula
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Clone and install the modified Modula library:
```bash
git clone <your-repo-url>
cd modula-v2
pip install -e .
```

## Running Training Examples

The repository includes several training examples in the `experiment` directory. Here's how to run them:

### CIFAR-10 Training Example

```bash
python experiment/train.py \
    --data cifar \
    --batch_size 512 \
    --model_dtype float32 \
    --project_dtype float32 \
    --steps 1000 \
    --lr 0.001 \
    --w_max 1.0 \
    --wd 0.0 \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.999 \
    --schedule cosine \
    --accum_steps 1 \
    --log_interval 10 \
    --val_interval 100 \
    --val_iters 10 \
    --seed 0 \
    --job_idx 0
```

### Shakespeare Text Generation Example

```bash
python experiment/train.py \
    --data shakespeare \
    --seq_len 128 \
    --batch_size 64 \
    --model_dtype float32 \
    --project_dtype float32 \
    --steps 1000 \
    --lr 0.001 \
    --w_max 1.0 \
    --wd 0.0 \
    --optimizer adam \
    --beta1 0.9 \
    --beta2 0.999 \
    --schedule cosine \
    --accum_steps 1 \
    --log_interval 10 \
    --val_interval 100 \
    --val_iters 10 \
    --seed 0 \
    --job_idx 0
```

## Key Features

- Supports multiple datasets (CIFAR-10, Shakespeare, FineWeb)
- Various optimization techniques including:
  - Adam optimizer
  - Learning rate schedules (linear, cosine, sqrt)
  - Weight decay
  - Gradient accumulation
  - Spectral normalization
- Training monitoring with:
  - Loss tracking
  - Accuracy metrics
  - GPU memory usage
  - RAM usage
  - ETA calculations

## Notes

- The training code automatically handles data loading and preprocessing
- By default, the DataLoader drops incomplete batches at the end of each epoch
- Training progress is logged with timestamps and resource usage statistics
- Checkpoints are saved periodically during training

For more detailed information about the training parameters and their effects, refer to the argument parser in `experiment/train.py`.