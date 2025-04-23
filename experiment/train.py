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
from modula.atom import Scalar, orthogonalize, laker_pure_svd, laker_special_sauce1, laker_special_sauce2, laker_special_sauce3, laker_special_sauce4, laker_special_sauce5

np.random.seed(0)

from data.shakespeare import load_shakespeare
from data.cifar10 import load_cifar10
#from data.fineweb import load_fineweb

max_log_priority = 1
def print_log(message, job_idx, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory = jax.device_get(jax.devices()[0].memory_stats()['peak_bytes_in_use']) / 1024**3 if jax.devices() else -1
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory:.1f}G ram {ram_usage:.1f}G idx {job_idx}] {' '*4*indent}{message}")

def load_data(args):
    if args.data == "fineweb":
        data = load_fineweb(args.seq_len, args.batch_size)
    elif args.data == "shakespeare":
        data = load_shakespeare(args.seq_len, args.batch_size)
    elif args.data == "cifar":
        data = load_cifar10(args.batch_size, randomize_labels=False)
    elif args.data == "cifar-random":
        data = load_cifar10(args.batch_size, randomize_labels=True)
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    return data["train_loader"], data["test_loader"], data["loss"]

project_str_to_fn = {
    "none": lambda x: x,
    "orthogonal": orthogonalize,
    "laker_pure_svd": laker_pure_svd,
    "laker_approximate1": laker_special_sauce1,
    "laker_approximate2": laker_special_sauce2,
    "laker_approximate3": laker_special_sauce3,
    "laker_approximate4": laker_special_sauce4,
    "laker_approximate5": laker_special_sauce5,
}

def create_model(args):
    kwargs = args.copy()

    # set out the dictionary for which project function to apply for each layer
    kwargs["project"] = {marker: project_str_to_fn[project] for marker, project in args.project.items()}

    if args.data == "fineweb" or args.data == "shakespeare":
        return GPT(**kwargs) if not args.manifold else OrthogonalGPT(**kwargs)
    elif args.data == "cifar":
        kwargs["output_dim"] = 10
        kwargs["input_dim"] = 32*32*3
        model = MLP(**kwargs) if not args.manifold else ManifoldMLP(**kwargs)
        return args.final_scale * model @ Flatten()
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
    print_log(f"Training with {num_params} parameters", args.job_idx)

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
    start_time = time.time()

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

        running_loss += loss
        if step % args.log_interval == 0:
            # Calculate and format ETA
            steps_done = (1 + step) * (1 + args.val_iters / args.val_interval)
            steps_remaining = (args.steps - step) * (1 + args.val_iters / args.val_interval)
            eta_seconds = (time.time() - start_time) * steps_remaining / steps_done
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            print_log(f"Step:{step}/{args.steps} train_loss:{loss:.4f} ETA:{eta_str}", args.job_idx)
                
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
            print_log(f"Step:{step}/{args.steps} val_loss:{val_losses[-1]:.4f} val_acc:{accuracies[-1]:.4f}", args.job_idx, indent=1)

        if step >= args.steps:
            break
    
    log["losses"] = losses
    log["val_losses"] = val_losses
    log["accuracies"] = accuracies
    log["total_time"] = time.time() - start_time
    return log

def save_results(results, args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timestamp_to_millisecond = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    args.timestamp = timestamp_to_millisecond

    uniqueness_hash = hash(json.dumps(args))
    filename = (
        f"{args.data}_{args.optimizer}_"
        f"embed{args.d_embed}_lr{args.lr:.4f}_"
        f"wd{args.wd:.4f}_steps{args.steps}_"
        f"{abs(uniqueness_hash):x}.json"
    )

    output = {
        'parameters': args,
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
    
    print_log(f"Results saved to {output_path}", args.job_idx)

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(e)

def main():
    # Create parser and add arguments
    parser = argparse.ArgumentParser(description="Modula train script for sweeps")
    parser.add_argument("--job_idx", type=int, default=-1, help="Index of the job")
    args = parser.parse_args()
    assert args.job_idx != -1, "job_idx must be set to the index of the job"
    
    with open('sweep_configs/parameter_grid.json', 'r') as f:
        job_idx = args.job_idx
        args = json.load(f)[job_idx]
        args["job_idx"] = job_idx
        args = DotDict(args)
    
    for key, value in args.items():
        print_log(f"{key}: {value}", args.job_idx)

    results = train(args)    
    save_results(results, args)

if __name__ == "__main__":
    main()
