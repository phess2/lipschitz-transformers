import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

import glob
import json
import argparse
from pathlib import Path
import numpy as np
import time
import jax
import jax.numpy as jnp
from modula.compound import OrthogonalGPT

np.random.seed(0)

from data.shakespeare import load_shakespeare

def load_data(args):
    data = load_shakespeare(args.seq_len, args.batch_size)

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    encode = data["encode"]
    decode = data["decode"]
    vocab_size = data["vocab_size"]

    return train_loader, val_loader, encode, decode, vocab_size

def train(args):
    train_loader, val_loader, encode, decode, vocab_size = load_data(args)

    log_interval = 10
    val_interval = 100
    val_iters = 50

    model = OrthogonalGPT(
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        d_embed=args.d_embed,
        num_blocks=args.blocks,
    )

    model.jit()

    def cross_entropy_loss(w, inputs, targets):
        logits = model(inputs, w)  # shape is [batch, seq_len, vocab_size]
        batch_indices = jnp.arange(logits.shape[0])[:, None]  # shape is [batch, 1]
        seq_indices = jnp.arange(logits.shape[1])[None, :]    # shape is [1, seq_len]
        losses = -logits[batch_indices, seq_indices, targets] + jax.nn.logsumexp(logits, axis=-1)  # shape is [batch, seq_len]
        return losses.mean()

    loss_and_grad = jax.jit(jax.value_and_grad(cross_entropy_loss))

    key = jax.random.PRNGKey(args.seed)
    w = model.initialize(key)
    log = {}

    losses = []
    val_losses = []
    num_params = sum(jnp.prod(jnp.array(p.shape)) for p in w).item()
    print(f"Training with {num_params} parameters")

    step = 0
    buf1 = [0 * weight for weight in w]
    buf2 = [0 * weight for weight in w]
    lr_schedule = lambda step: args.lr * (args.steps - step) / args.steps
    for inputs, targets in train_loader:
        loss, grad_w = loss_and_grad(w, inputs, targets)
        # pre_dualize, update first moment, update second moment, possibly apply adam, post_dualize
        d_m = model.dualize(grad_w) if args.pre_dualize else grad_w
        buf1 = [args.beta1 * m + (1-args.beta1) * d_m    for m, d_m in zip(buf1, d_m)]
        buf2 = [args.beta2 * m + (1-args.beta2) * d_m**2 for m, d_m in zip(buf2, d_m)]
        d_w = [m1 / (jnp.sqrt(m2) + 1e-12) if args.optimizer == "adam" else m1 for m1, m2 in zip(buf1, buf2)]
        d_w = model.dualize(d_w) if args.post_dualize else d_w
        wd_factor = 1 - args.wd * lr_schedule(step)
        w = [wd_factor * weight - lr_schedule(step) * d_weight for weight, d_weight in zip(w, d_w)]
        # w = model.project(w)
        losses.append(loss)

        if step % log_interval == 0:
            print(f"Step {step}: loss {loss}")
            log = model.log(w, grad_w)
        
        if step % val_interval == 0:
            val_losses = []
            for val_inputs, val_targets in val_loader:
                loss, _ = loss_and_grad(w, val_inputs, val_targets)
                val_losses.append(loss)
                if len(val_losses) >= val_iters:
                    break
            val_loss = sum(val_losses)/len(val_losses)
            print(f"--> val loss {val_loss}")
            val_losses.append(val_loss)
        step += 1

        if step >= args.steps:
            break
    
    # one big val set test at the end
    val_losses = []
    for val_inputs, val_targets in val_loader:
        loss, _ = loss_and_grad(w, val_inputs, val_targets)
        val_losses.append(loss)
        if len(val_losses) >= val_iters * 10:
            break
    print(f"--> val loss over {val_iters * 10} batches: {sum(val_losses)/len(val_losses)}")

    log["losses"] = losses
    log["val_losses"] = val_losses
    return log

def save_results(results, args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"embed{args.d_embed}_lr{args.lr:.4f}_{args.optimizer}_steps{args.steps}_{timestamp}.json"

    print(vars(args))
    output = {
        'parameters': vars(args),
        'results': results,
        'code': code
    }

    output_path = Path(args.output_dir) / filename
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Modula train script for sweeps")
    parser.add_argument("--d_embed", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--blocks", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--pre_dualize", type=lambda x: x.lower() == "true", default=False, help="Whether to pre-dualize")
    parser.add_argument("--post_dualize", type=lambda x: x.lower() == "true", default=True, help="Whether to post-dualize")
    parser.add_argument("--beta1", type=float, default=0.95, help="Momentum buffer 1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Momentum buffer 2 coefficient")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--project", type=lambda x: x.lower() == "true", default=True, help="Whether to project the weights")
    parser.add_argument("--steps", type=int, default=2001, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    results = train(args)    
    save_results(results, args)

if __name__ == "__main__":
    main()
