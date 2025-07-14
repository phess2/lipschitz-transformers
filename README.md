# Training transformers with enforced Lipschitz constants

![Main method: 1) use Muon to constrain the weight update norm, 2) project weights to have a max singular value, 3) norm guarantee.](assets/main.png)

What if large scale transformer training could be free of loss spikes? Is there a better way than weight decay?

Lipschitz constants are a bound on the model's sensitivity to input or weight changes. By controlling them, we can stabilize training by preventing exploding attention logits, set adversarial robustness bounds in advance, and possibly create models more compatible with low precision inference. We compare pairs of (optimizer, weight constraint method) across AdamW / Muon and existing constraint methods / our proposed methods _spectral cap_ and _spectral hammer_. We find that Muon improves weight constraint methods across the board in the Lipschitz vs. performance tradeoff. And we show that it is possible to train a 145M parameter NanoGPT to competitive accuracy with entirely constrained weights.

As always, there is a lot of work left to train models faster and more scalably (e.g., with Lipschitz guarantees). This repo has a setup to reproduce our results, or train your own Lipschitz-constrained models. Our data is also available on [Huggingface](https://huggingface.co/phess2/lipschitz-transformers).

## Setup

1. `git clone https://github.com/Arongil/lipschitz-transformers`
2. `python -m venv lipschitz`
3. `source lipschitz/bin/activate`
4. `pip install -e .`

## Train a Lipschitz-enforced transformer

All three examples are available in `experiment.ipynb`. Just change the config you select, and run. Any of the examples in our paper are reproducible here, or you can try your own settings or your own constraint methods.

Warmup #1: MLP on CIFAR-10, unconstrained (baseline)

Warmup #2: MLP on CIFAR-10, constrained (ours)

Warmup #3: Shakespeare transformer with 2M parameters

To run the Shakespeare transformer from a checkpoint (or check out some of our checkpoints on [Huggingface](https://huggingface.co/phess2/lipschitz-transformers)), use `run_checkpoint.py`.

### The real deal: 145M parameter NanoGPT

The [modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt) repo by Keller Jordan has a wonderful script that trains a GPT-2 small scale transformer in under 3 minutes on an 8xH100. We modified the script to enforce Lipschitz constraints. You can run the script with `/nanogpt/run.sh` -- see the subdirectory's README for setup instructions. There's a default spectral cap example, plus a spectral normalization example.

## Acknowledgments

Thank you to Lambda Labs and Rami Seid for supporting the work with compute credits!

## Citation

```bibtex
@article{newhouse2025lipschitztransformers,
  title={Training Transformers with Enforced Lipschitz Constants},
  author={Laker Newhouse, R. Preston Hess, Franz Cesista, Andrii Zahorodnii, Jeremy Bernstein, Phillip Isola},
  journal={arXiv}
  year={2025}
}
```
