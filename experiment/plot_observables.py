"""
Takes results path and plots some observable over time, specified at the bottom of the script.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
import json
import dotenv
import os

dotenv.load_dotenv()
root_path = os.getenv('ROOT_PATH')
path = root_path + "experiment/results-8-lr-sweep-exp-scalar-lr-4xed/embed128_lr0.0160_muon_preFalse_postTrue_projectFalse_manifoldTrue_final_scale0.0_softmax_scale0.0_wd0.0000_steps1001_20250319_070044.json"

with open(path, "r") as f:
    data = json.load(f)
    log = data["results"]

# plot an observable over training: queries (q), keys (k), values (v), out_proj (w), mlp_in, or mlp_out
def plot_observable(tracker_start="q", observable="weight_norm"):
    trackers = [tracker for tracker in log.keys() if tracker.startswith(tracker_start)]
    
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    log_interval = 10
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for tracker in sorted(trackers):
        if observable in log[tracker]:
            observables = [float(w) if isinstance(w, jnp.ndarray) else w for w in log[tracker][observable]]
            steps = jnp.arange(len(observables)) * log_interval
            ax.plot(steps, observables, label=f"{tracker}", linewidth=2)
    
    ax.set_xlabel("Training steps", fontsize=16)
    ax.set_ylabel(observable, fontsize=16)
    ax.set_title(f"{tracker_start} {observable} over time (manifold constrained)", fontsize=18)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.2, linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, framealpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"scales_plots/{tracker_start}_{observable}.png", dpi=300)

#for t in ["q", "k", "v", "w", "mlp_in", "mlp_out", "mlp_final"]:
#    plot_observable(t, "weight_norm")

for t in ["softmax", "final_scale"]:
    plot_observable(t, "exp_scalar")