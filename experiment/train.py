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
import psutil
import jax
import jax.numpy as jnp
from modula.compound import GPT, MLP, OrthogonalGPT, ManifoldMLP
from modula.bond import Flatten
from modula.atom import Scalar, orthogonalize, laker_special_sauce, laker_pure_svd

np.random.seed(0)

from data.shakespeare import load_shakespeare
from data.cifar10 import load_cifar10
from data.fineweb import load_fineweb

max_log_priority = 1
def print_log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory = jax.device_get(jax.devices()[0].memory_stats()['peak_bytes_in_use']) / 1024**3 if jax.devices() else -1
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory:.1f}G ram {ram_usage:.1f}G] {' '*4*indent}{message}")

def load_data(args):
    if args.data == "fineweb":
        data = load_fineweb(args.seq_len, args.batch_size)
    elif args.data == "shakespeare":
        data = load_shakespeare(args.seq_len, args.batch_size)
    elif args.data == "cifar":
        data = load_cifar10(args.batch_size)
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    return data["train_loader"], data["test_loader"], data["loss"]

project_str_to_fn = {
    "none": lambda x: x,
    "orthogonal": orthogonalize,
    "laker": laker_special_sauce,
    "laker_pure_svd": laker_pure_svd,
}

def create_model(args):
    kwargs = args.copy()

    # set out the dictionary for which project function to apply for each layer
    project_dict = json.loads(args.project_dict)
    kwargs["project"] = {marker: project_str_to_fn[project] for marker, project in project_dict.items()}

    if args.data == "fineweb" or args.data == "shakespeare":
        return GPT(**kwargs) if not args.manifold else OrthogonalGPT(**kwargs)
    elif args.data == "cifar":
        kwargs["output_dim"] = 10
        kwargs["input_dim"] = 32*32*3
        model = MLP(**kwargs) if not args.manifold else ManifoldMLP(**kwargs)
        return Scalar(args.final_scale) @ model @ Flatten()
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
    print_log(f"Training with {num_params} parameters")

    step = 0
    accum_step = 0
    accum_loss = 0.0
    accum_grad = jax.tree.map(jnp.zeros_like, w)
    buf1 = jax.tree.map(jnp.zeros_like, w)
    buf2 = jax.tree.map(jnp.zeros_like, w) if args.optimizer == "adam" else None
    schedule = {
        "linear": lambda step: (args.steps - step) / args.steps,
        "cosine": lambda step: 0.5 * (1 + jnp.cos(jnp.pi * step / args.steps)),
        "none": lambda step: 1
    }[args.schedule]
    running_loss = 0.0

    for inputs, targets in train_loader:
        loss, grad_w = loss_and_grad(w, inputs, targets)
        accum_grad = jax.tree.map(jnp.add, accum_grad, grad_w)
        accum_loss += loss.item()
        accum_step += 1

        # only update the weights after accumulating enough gradients
        if accum_step % args.accum_steps != 0:
            continue

        # prepare for the next accumulation
        loss = accum_loss / args.accum_steps
        grad_w = jax.tree.map(lambda g: g / args.accum_steps, accum_grad)
        accum_grad = jax.tree.map(jnp.zeros_like, w)
        accum_loss = 0.0
        step = (accum_step - 1) // args.accum_steps

        # pre_dualize, update first moment, update second moment, possibly apply adam, post_dualize
        d_m = model.dualize(grad_w) if args.pre_dualize else grad_w
        buf1 = jax.tree.map(lambda m, d_m: args.beta1 * m + (1-args.beta1) * d_m, buf1, d_m)
        buf2 = jax.tree.map(lambda m, d_m: args.beta2 * m + (1-args.beta2) * d_m**2, buf2, d_m) if args.optimizer == "adam" else None
        d_w = jax.tree.map(lambda m1, m2: m1 / (jnp.sqrt(m2) + 1e-12), buf1, buf2) if args.optimizer == "adam" else buf1
        d_w = model.dualize(d_w) if args.post_dualize else d_w

        # Original coupling code (couples initial learning rate too)
        # if args.wd_lr_power == 0: wd_step_size = schedule(step) # decoupled weight decay like in the original AdamW paper
        # else: wd_step_size = (args.lr * schedule(step)) ** args.wd_lr_power # control the proportionality of weight decay to lr
        
        # Test coupling code (only couples the schedule step)
        wd_step_size = args.lr * schedule(step) ** args.wd_lr_power
        w = jax.tree.map(lambda weight: (1 - args.wd * wd_step_size) * weight, w)
        w = model.step(w, d_w, args.lr * schedule(step))
        w = model.project(w)
        print_log(f"Step:{step}/{args.steps} train_loss:{loss:.4f}")

        running_loss += loss
        if step % args.log_interval == 0:
            interval_loss = running_loss if step == 0 else running_loss / args.log_interval
            log = model.log(w, d_w)
            losses.append(float(interval_loss))
            running_loss = 0.0
        
        if step % args.val_interval == 0:
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_step = 0
            for val_inputs, val_targets in val_loader:
                loss, _ = loss_and_grad(w, val_inputs, val_targets)
                logits = model(val_inputs, w)
                val_loss_sum += float(loss)
                preds = jnp.argmax(logits, axis=-1)
                val_acc_sum += jnp.mean(preds == val_targets)
                val_step += 1
                if val_step >= args.val_iters:
                    break
            val_losses.append(float(val_loss_sum / val_step))
            accuracies.append(float(val_acc_sum / val_step))
            print_log(f"Step:{step}/{args.steps} val_loss:{val_losses[-1]:.4f} val_acc:{accuracies[-1]:.4f}", indent=1)

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
        f"{args.project + '_' if args.project != 'none' else ''}"
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
    
    print_log(f"Results saved to {output_path}")

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
    parser.add_argument("--accum_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--zero_init", type=lambda x: x.lower() == "true", default=True, help="Whether to zero-init the out projection in attention")
    parser.add_argument("--project_dict", type=str, default="none", help="The way to project the weights, with \"default\" and specific layer names each assigned project functions")
    parser.add_argument("--manifold", type=lambda x: x.lower() == "true", default=False, help="Whether to constrain to the manifold directly")
    parser.add_argument("--schedule", type=str, default="linear", help="Learning rate schedule")
    parser.add_argument("--steps", type=int, default=2001, help="Number of steps")
    parser.add_argument("--data", type=str, default="shakespeare", help="Which dataset to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--val_interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--val_iters", type=int, default=200, help="Validation iterations")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    results = train(args)    
    save_results(results, args)

if __name__ == "__main__":
    main()
