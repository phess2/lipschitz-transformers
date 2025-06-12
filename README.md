# Lipschitz enforced transformer training!

Is weight decay the best we can do? This repo contains the code for the paper "Training Transformers with Enforced Lipschitz Constants." There remains lots to do to train models faster and with regularization guarantees like a Lipschitz constant. The repo has a simple setup to train your own Lipschitz-constrained models, or reproduce our results.

## Setup

1. Create environment:
```bash
conda create -n modula python=3.9
conda activate modula
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Clone repo:
```bash
git clone <your-repo-name>
cd <your-repo-name>
pip install -e .
```

## Train a Lipschitz-enforced transformer

### Warmup: MLP on CIFAR-10

```bash
python experiment/train.py \
    --data cifar \
    --batch_size 512 \
    --model_dtype float32 \
    --project_dtype float32 \
    --steps 4000 \
    --lr 0.1 \
    --w_max 2.0 \
    --wd 0.0 \
    --optimizer muon \
    --project '{"default": "soft_cap"}' \
    --beta1 0.9 \
    --beta2 0.95 \
    --schedule linear \
    --accum_steps 1 \
    --log_interval 10 \
    --val_interval 100 \
    --val_iters 10 \
    --seed 0 \
    --job_idx 0
```

### 2M parameter Shakespeare transformer

```bash
python experiment/train.py \
    --data shakespeare \
    --seq_len 128 \
    --batch_size 64 \
    --model_dtype float32 \
    --project_dtype float32 \
    --steps 2000 \
    --lr 0.001 \
    --w_max 1.6 \
    --wd 0.0 \
    --optimizer muon \
    --project '{"default": "soft_cap"}' \
    --beta1 0.9 \
    --beta2 0.95 \
    --schedule linear \
    --accum_steps 1 \
    --log_interval 10 \
    --val_interval 100 \
    --val_iters 10 \
    --seed 0 \
    --job_idx 0
```

### 145M parameter NanoGPT

The [modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt) repo by Keller Jordan has a wonderful script that trains a GPT-2 small scale transformer in under 3 minutes on an 8xH100. We modified the script to enforce Lipschitz constraints. The script is `/nanogpt/run.py`. The setup instructions are identical to modded NanoGPT. There are some options, like which enforcement method you want to use (spectral normalize, spectral cap). Try it out! Shakespeare transformers could train faster with Lipschitz constraints, and we would love to hear if you can train at NanoGPT scale faster using strong weight constraints.

## Acknowledgments

Thank you to Lambda Labs and Rami Seid for supporting the work with compute credits.

## Citation

```bibtex
@article{newhouse2025lipschitztransformers,
  title={Training Transformers with Enforced Lipschitz Constants},
  author={Laker Newhouse, R. Preston Hess, Franz Cesista, Andrii Zahorodnii, Jeremy Bernstein, Phillip Isola},
  journal={arXiv:???????????},
  year={2025}
}
```