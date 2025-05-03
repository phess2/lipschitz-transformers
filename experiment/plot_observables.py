"""
Takes results paths and plots some observable over time, specified at the bottom of the script.
Allows comparing multiple experiments in the same plot.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib.pyplot as plt
import jax.numpy as jnp
import json
import dotenv
from pathlib import Path
import argparse

dotenv.load_dotenv()
root_path = os.getenv('ROOT_PATH') / Path("experiment")
parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=str, nargs='+', help="List of paths to experiment results")
args = parser.parse_args()

# Load all experiment data
experiments = {}
for path in args.paths:
    full_path = root_path / Path(path)
    exp_name = Path(path).stem  # Use filename without extension as experiment name
    with open(full_path, "r") as f:
        data = json.load(f)
        experiments[exp_name] = data["results"]

# plot an observable over training: queries (q), keys (k), values (v), out_proj (w), mlp_in, or mlp_out
def plot_observable(tracker_start="q", observable="weight_norm"):
    # Set matplotlib style for thesis
    plt.style.use('default')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 1.0
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False
    log_interval = 10
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Get all unique trackers across experiments
    all_trackers = set()
    for exp_data in experiments.values():
        trackers = [t for t in exp_data.keys() if t.startswith(tracker_start)]
        all_trackers.update(trackers)
    
    # Sort trackers for consistent ordering
    all_trackers = sorted(all_trackers)
    
    # Use different line styles for different trackers
    line_styles = ['-', '--', ':', '-.']
    
    for exp_name, exp_data in experiments.items():
        for tracker in all_trackers:
            if tracker in exp_data and observable in exp_data[tracker]:
                observables = [float(w) if isinstance(w, jnp.ndarray) else w for w in exp_data[tracker][observable]]
                steps = jnp.arange(len(observables)) * log_interval
                line_style = line_styles[all_trackers.index(tracker) % len(line_styles)]
                ax.plot(steps, observables, 
                       label=f"{exp_name}", 
                       linestyle=line_style)
    
    ax.set_xlabel("Training steps", fontsize=16)
    ax.set_ylabel(observable, fontsize=16)
    ax.set_title(f"{tracker_start} {observable} over time", fontsize=18)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.2, linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12, framealpha=0.7, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs("scales_plots", exist_ok=True)
    plt.savefig(f"scales_plots/{tracker_start}_{observable}.png", dpi=300, bbox_inches='tight')
    plt.close()

for t in ["mlp_in", "mlp_0", "mlp_out"]:
   plot_observable(t, "weight_norm")
   #plot_observable(t, "cos_angle_w_with_d_w")

# for t in ["softmax", "final_scale"]:
#     plot_observable(t, "exp_scalar")