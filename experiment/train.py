import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

import glob
import json
import argparse
from pathlib import Path
from functools import partial
import numpy as np
import time
import jax
import jax.numpy as jnp
from modula.compound import GPT, OrthogonalGPT, LakerGPT, MLP, ManifoldMLP, LakerMLP
from modula.bond import Flatten
from modula.atom import Scalar

np.random.seed(0)

from data.shakespeare import load_shakespeare
from data.cifar10 import load_cifar10

def load_data(args):
    if args.data == "shakespeare":
        data = load_shakespeare(args.seq_len, args.batch_size)
    elif args.data == "cifar":
        data = load_cifar10(args.batch_size)
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    return data["train_loader"], data["test_loader"], data["loss"]

def create_model(args):
    if args.data == "shakespeare":
        kwargs = {"vocab_size": 65, "num_heads": args.num_heads, "d_embed": args.d_embed, "num_blocks": args.blocks, "softmax_scale": args.softmax_scale, "final_scale": args.final_scale, "residual_scale": args.residual_scale, "scales_learnable": args.scales_learnable, "d_query": args.d_embed // args.num_heads, "d_value": args.d_embed // args.num_heads, "zero_init": args.zero_init}
        if args.manifold:
            return OrthogonalGPT(**kwargs)
        elif args.project:
            return LakerGPT(**kwargs)
        else:
            return GPT(**kwargs)
    elif args.data == "cifar":
        kwargs = {"output_dim": 10, "input_dim": 32*32*3, "width": args.d_embed, "depth": args.blocks}
        if args.manifold:
            assert args.project, "Manifold models must be projected due to rectangular matrices in CIFAR"
            return Scalar(args.final_scale) @ MLP(**kwargs) @ Flatten()
        elif args.project:
            return Scalar(args.final_scale) @ LakerMLP(**kwargs) @ Flatten()
        else:
            return MLP(**kwargs) @ Flatten()
    else:
        raise ValueError(f"Unknown dataset: {args.data}")

def train(args):
    train_loader, val_loader, loss = load_data(args)

    model = create_model(args)
    model.jit()

    # loss takes (model, w, inputs, targets), so we wrap model in first
    loss_and_grad = jax.jit(jax.value_and_grad(partial(loss, model)))

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    w = model.initialize(subkey)
    log = {}

    losses = []
    val_losses = []
    accuracies = []
    num_params = sum(jnp.prod(jnp.array(p.shape)) for p in w).item()
    print(f"Training with {num_params} parameters")

    step = 0
    running_loss = 0.0
    buf1 = [0 * weight for weight in w]
    buf2 = [0 * weight for weight in w]
    schedule = {
        "linear": lambda step: (args.steps - step) / args.steps,
        "cosine": lambda step: 0.5 * (1 + jnp.cos(jnp.pi * step / args.steps)),
        "none": lambda step: 1
    }[args.schedule]

    for inputs, targets in train_loader:
        loss, grad_w = loss_and_grad(w, inputs, targets)
        # pre_dualize, update first moment, update second moment, possibly apply adam, post_dualize
        d_m = model.dualize(grad_w) if args.pre_dualize else grad_w
        buf1 = [args.beta1 * m + (1-args.beta1) * d_m    for m, d_m in zip(buf1, d_m)]
        buf2 = [args.beta2 * m + (1-args.beta2) * d_m**2 for m, d_m in zip(buf2, d_m)]
        d_w = [m1 / (jnp.sqrt(m2) + 1e-12) if args.optimizer == "adam" else m1 for m1, m2 in zip(buf1, buf2)]
        d_w = model.dualize(d_w) if args.post_dualize else d_w

        if args.wd_lr_power == 0: wd_step_size = schedule(step) # decoupled weight decay like in the original AdamW paper
        else: wd_step_size = (args.lr * schedule(step)) ** args.wd_lr_power # control the proportionality of weight decay to lr
        w = [(1 - args.wd * wd_step_size) * weight for weight in w]

        w = model.step(w, d_w, args.lr * schedule(step))
        if args.project:
            w = model.project(w)

        running_loss += loss.item()
        if step % args.log_interval == 0:
            interval_loss = running_loss if step == 0 else running_loss / args.log_interval
            print(f"Step {step}: loss {interval_loss}")
            log = model.log(w, grad_w)
            losses.append(float(interval_loss))
            running_loss = 0.0
        
        if step % args.val_interval == 0:
            accuracies_to_avg = []
            val_losses_to_avg = []
            for val_inputs, val_targets in val_loader:
                loss, _ = loss_and_grad(w, val_inputs, val_targets)
                val_losses_to_avg.append(float(loss))
                logits = model(val_inputs, w)
                preds = jnp.argmax(logits, axis=-1)
                accuracies_to_avg.append(jnp.mean(preds == val_targets))
                if len(val_losses_to_avg) >= args.val_iters:
                    break
            val_losses.append(float(sum(val_losses_to_avg)/len(val_losses_to_avg)))
            accuracies.append(float(sum(accuracies_to_avg)/len(accuracies_to_avg)))
            print(f"--> val loss {val_losses[-1]}")
            print(f"--> val accuracy {accuracies[-1]}")
        step += 1

        if step >= args.steps:
            break
    
    log["losses"] = losses
    log["val_losses"] = val_losses
    log["accuracies"] = accuracies
    return log

def save_results(results, args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timestamp_to_millisecond = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    filename = (
        f"{args.data}_"
        f"embed{args.d_embed}_lr{args.lr:.4f}_{args.optimizer}_"
        f"{'pre_' if args.pre_dualize else ''}"
        f"{'post_' if args.post_dualize else ''}"
        f"{'manifold_' if args.manifold else ''}"
        f"{'project_' if args.project else ''}"
        #f"fscale{args.final_scale}_sscale{args.softmax_scale}_"
        f"wd{args.wd:.4f}_steps{args.steps}_"
        f"{timestamp_to_millisecond}.json"
    )

    output = {
        'parameters': vars(args),
        'results': results,
        'code': code
    }

    def jax_to_numpy(d):
        if isinstance(d, dict):
            return {k: jax_to_numpy(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [jax_to_numpy(v) for v in d]
        elif isinstance(d, jnp.ndarray):
            return d.tolist()
        else:
            return d

    output = jax_to_numpy(output)
    output_path = Path(args.output_dir) / filename
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Modula train script for sweeps")
    parser.add_argument("--d_embed", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--wd_lr_power", type=float, default=0, help="Weight decay power of coupling to learning rate")
    parser.add_argument("--blocks", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--softmax_scale", type=float, default=1.0, help="Softmax scale")
    parser.add_argument("--final_scale", type=float, default=1.0, help="Final scale")
    parser.add_argument("--residual_scale", type=float, default=1.0, help="a, where x becomes (1 - a/depth) * x + (a/depth) * block(x)")
    parser.add_argument("--scales_learnable", type=lambda x: x.lower() == "true", default=False, help="Whether to learn the scales")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--pre_dualize", type=lambda x: x.lower() == "true", default=False, help="Whether to pre-dualize")
    parser.add_argument("--post_dualize", type=lambda x: x.lower() == "true", default=True, help="Whether to post-dualize")
    parser.add_argument("--beta1", type=float, default=0.95, help="Momentum buffer 1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Momentum buffer 2 coefficient")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--zero_init", type=lambda x: x.lower() == "true", default=True, help="Whether to zero-init the out projection in attention")
    parser.add_argument("--project", type=lambda x: x.lower() == "true", default=False, help="Whether to project the weights")
    parser.add_argument("--manifold", type=lambda x: x.lower() == "true", default=False, help="Whether to constrain to the manifold directly")
    parser.add_argument("--schedule", type=str, default="linear", help="Learning rate schedule")
    parser.add_argument("--steps", type=int, default=2001, help="Number of steps")
    parser.add_argument("--data", type=str, default="shakespeare", help="Which dataset to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    args.log_interval = 10 if args.data == "shakespeare" else 100
    args.val_interval = 100 if args.data == "shakespeare" else 500
    args.val_iters = 50
    
    results = train(args)    
    save_results(results, args)

if __name__ == "__main__":
    main()
