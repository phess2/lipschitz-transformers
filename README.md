# Lipschitz transformers using Muon + weight constraints!

What if large scale transformer training could be free of loss spikes? Is there a better way than weight decay?

This repo contains the code for "Training Transformers with Enforced Lipschitz Constants." Lipschitz constants are a bound on the model's sensitivity to input or weight changes; by controlling Lipschitz constants, we can stabilize training by preventing exploding attention logits, and the model artifact at the end can have higher adversarial robustness. We compare existing methods and our proposed _spectral cap_ and _spectral hammer_ methods. There is a lot of work left to train models faster and with regularization guarantees. This repo has a simple setup to train your own Lipschitz-constrained models, or reproduce our results.

## Setup

1. `git clone https://github.com/Arongil/lipschitz-transformers`
2. `conda create -n lipschitz python=3.9`
3. `conda activate lipschitz`
4. `pip install -e .`

## Train a Lipschitz-enforced transformer

### Warmup #1: MLP on CIFAR-10

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

### Warmup #2: Shakespeare transformer with 2M parameters

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

### The real deal: 145M parameter NanoGPT

The [modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt) repo by Keller Jordan has a wonderful script that trains a GPT-2 small scale transformer in under 3 minutes on an 8xH100. We modified the script to enforce Lipschitz constraints. You can run the script with `/nanogpt/run.sh` -- see the subdirectory's README for setup instructions.



. There are some options, like which enforcement method you want to use (spectral normalize, spectral cap). Try it out! Shakespeare transformers could train faster with Lipschitz constraints, and we would love to hear if you can train at NanoGPT scale faster using strong weight constraints.

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