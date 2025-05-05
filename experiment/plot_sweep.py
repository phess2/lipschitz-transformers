import json
import pickle
from pathlib import Path
import imageio
import os
import re

# Create a matplotlib plot of training curves
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator, FixedLocator
from collections import defaultdict


results_dir = Path('.')

# Set global font sizes
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Custom formatter to remove .0
def format_fn(x, p):
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"

# Cache the results to speed up plotting on the same data
cache_file = results_dir / 'cached_results.pkl'
def need_to_rebuild_cache():
    if not cache_file.exists():
        return True
    cache_mtime = cache_file.stat().st_mtime
    json_files = list(results_dir.glob('*.json'))
    return any(f.stat().st_mtime > cache_mtime for f in json_files)

if not cache_file.exists() or need_to_rebuild_cache():
    results = []
    for file in results_dir.glob('*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append({
                'learning_rate': data['parameters']['lr'],
                'final_scale': data['parameters']['final_scale'],
                'train_loss': min(data['results']['losses']),  # Get best train loss
                'test_loss': min(data['results']['val_losses']),  # Get best test loss
                'weight_decay': data['parameters']['wd'],
            })
    
    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
else:
    with open(cache_file, 'rb') as f:
        results = pickle.load(f)

# Group results by final scale
scale_results = defaultdict(list)
for r in results:
    scale_results[r['weight_decay']].append(r)

# Create the plot
plt.figure(figsize=(6, 4))

# Get scales and create color map
scales = sorted(scale_results.keys())
color_map = plt.cm.viridis(np.linspace(0, 1, len(scales)))
color_dict = {scale: color for scale, color in zip(scales, color_map)}

# Plot one curve for each final scale value
for scale in scales:
    scale_data = scale_results[scale]
    lrs = [r['learning_rate'] for r in scale_data]
    train_losses = [r['train_loss'] for r in scale_data]
    # Sort by learning rate for a smooth curve
    sorted_pairs = sorted(zip(lrs, train_losses))
    lrs_sorted, train_losses_sorted = zip(*sorted_pairs)
    
    # Plot with consistent styling and viridis colors
    color = color_dict[scale]
    plt.plot(lrs_sorted, train_losses_sorted, '-', linewidth=3, 
             color=color, label=f'Weight Decay={scale}')
    
    # Add red dot for minimum
    min_idx = np.argmin(train_losses_sorted)
    plt.plot(lrs_sorted[min_idx], train_losses_sorted[min_idx], 'ro', 
             markersize=6, zorder=10)

plt.xlabel('Learning Rate')
plt.ylabel('Train Loss')
plt.title('Train Loss vs. Learning Rate for Different Weight Decays', y=1.025)
plt.legend(title='Weight Decay', bbox_to_anchor=(1.01, 0.5), 
          loc='center left', borderaxespad=0)
plt.grid(False)
plt.xscale('log')
plt.yscale('log')

# Configure axis ticks
plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=4))
plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=4))
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_fn))

# Make axis lines thicker
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

# Save the plot
plt.savefig(results_dir / '_lrtransfer.png', bbox_inches='tight', dpi=300)
plt.savefig(results_dir / '_lrtransfer.svg', bbox_inches='tight', dpi=300)
plt.close()
