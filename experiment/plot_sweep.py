import json
import pickle
from pathlib import Path

# Create a matplotlib plot of training curves
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator
from collections import defaultdict

results_dir = Path('results')

# Set global font sizes
plt.rcParams.update({
    'font.size': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
})

# Custom formatter to remove .0
def format_fn(x, p):
    if data == 'cifar10':
        return f"{x:.1e}" # scientific notation
    else:
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"

# Cache the results to speed up plotting on the same data
cache_file = results_dir / 'cached_results.pkl'
def need_to_rebuild_cache():
    cache_mtime = cache_file.stat().st_mtime
    json_files = list(results_dir.glob('*.json'))
    return any(f.stat().st_mtime > cache_mtime for f in json_files)

if not cache_file.exists() or need_to_rebuild_cache():
    results = []
    for file in results_dir.glob('*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append({
                'd_embed': data['parameters']['d_embed'],
                'blocks': data['parameters']['blocks'],
                'learning_rate': data['parameters']['lr'],
                'batch_size': data['parameters']['batch_size'],
                'optimizer': data['parameters']['optimizer'],
                'weight_decay': data['parameters']['wd'],
                'data': data['parameters']['data'],
                'accuracy_history': data['results']['accuracies'],
                'train_loss_history': data['results']['losses'],
                'test_loss_history': data['results']['val_losses'],
                'seed': data['parameters']['seed'],
                'project': data['parameters']['project'],
                'manifold': data['parameters']['manifold'],
                'final_scale': data['parameters']['final_scale'],
                'softmax_scale': data['parameters']['softmax_scale'],
                'residual_scale': data['parameters']['residual_scale'],
            })
    
    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
else:
    with open(cache_file, 'rb') as f:
        results = pickle.load(f)

# Choose properties to make separate panels for, including an optional direct filter for all panels
panel_list = ['optimizer', 'project', 'weight_decay']
panel_filter = lambda x: x['project'] == True
panels = sorted(list(set(tuple(r[axis] for axis in panel_list) for r in results if panel_filter(r))))
# Choose what the color bar will sweep over
x_string = 'residual_scale'  # width, depth, batch_size
x_string_title = 'Residual Scale'  # Width, Depth, Batch Size
data = 'shakespeare'

use_test_loss = False
use_accuracy = True

# Get unique values for color bar
color_values = sorted(list(set(r[x_string] for r in results)))

# Create single row of subplots, one per panel
plot_size = 3.2621 * 2
fig = plt.figure(figsize=(plot_size, plot_size/4))#plot_size/3))
gs = gridspec.GridSpec(1, len(panels) + 1, 
                      width_ratios=[1]*len(panels) + [0.25])
gs.update(wspace=0.25)  # Adjust spacing between subplots

axes = []
for i in range(len(panels)):
    ax = fig.add_subplot(gs[0, i], sharey=axes[0] if len(axes) > 0 else None)
    axes.append(ax)
axes = np.array(axes)

loss_string = 'Test' if use_test_loss or use_accuracy else 'Training'
loss_string_unit = 'Accuracy' if use_accuracy else 'Loss'
model_string = 'MLP' if data == 'cifar10' else 'Transformer'
dataset_string = 'CIFAR-10' if data == 'cifar10' else 'Shakespeare'

# Store all lines and labels for the universal legend
all_lines = []
all_labels = []

# Store red dots to plot later
red_dots = []

panel_prefix = {
    'optimizer': lambda x: x.capitalize(),
    'project': lambda x: 'P' if x else 'NP',
    'weight_decay': lambda x: f'{x:.2f}',
}

fig.supxlabel('Learning Rate', y=0.02)
fig.supylabel(f'Final {loss_string} {loss_string_unit}', x=0.02)

for i, panel in enumerate(panels):
    ax = axes[i] if len(panels) > 1 else axes
    # Get unique values for color mapping
    x_values = sorted(list(set(r[x_string] for r in results 
                           if tuple(r[axis] for axis in panel_list) == panel and panel_filter(r))))
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_values)))
    
    # Plot sweep of final training loss vs learning rate for each color bar variable
    for x_value, color in zip(x_values, colors):
        # Get all results that match the current panel and color bar variable
        x_value_results = [r for r in results 
                         if r[x_string] == x_value 
                         and tuple(r[axis] for axis in panel_list) == panel
                         and panel_filter(r)]
        
        # Group results by learning rate and seed
        lr_results = defaultdict(list)
        for r in x_value_results:
            lr_results[r['learning_rate']].append(r)
        
        # Extract learning rates and compute statistics across seeds
        learning_rates = sorted(lr_results.keys())
        avg_losses = []
        std_losses = []
        
        for lr in learning_rates:
            fn = max if use_accuracy else min
            if use_accuracy:
                losses = [fn(r['accuracy_history']) * 100 for r in lr_results[lr]]  # convert to percent
            elif use_test_loss:
                losses = [fn(r['test_loss_history']) for r in lr_results[lr]]
            else:
                losses = [fn(r['train_loss_history']) for r in lr_results[lr]]
            
            avg_losses.append(np.mean(losses))
            std_losses.append(np.std(losses))
        
        # Convert to numpy arrays
        learning_rates = np.array(learning_rates)
        avg_losses = np.array(avg_losses)
        std_losses = np.array(std_losses)
        
        # Plot curve for this x_value with std deviation halo
        ax.fill_between(learning_rates, avg_losses - std_losses, avg_losses + std_losses, 
                        alpha=0.3, color=color)
        line, = ax.plot(learning_rates, avg_losses, '-', color=color, 
                       linewidth=3, markersize=4)
        
        # Only store lines/labels for legend from the first subplot
        if i == 0:
            all_lines.append(line)
            all_labels.append(f'{x_string} {x_value}')
        
        # Store red dot information for later plotting (minimum average loss)
        min_loss_idx = np.argmax(avg_losses) if use_accuracy else np.argmin(avg_losses)
        red_dots.append((ax, learning_rates[min_loss_idx], avg_losses[min_loss_idx]))

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(numticks=3))

    # Configure only the leftmost subplot's y-axis, since all are shared anyway
    if i == 0:
        if not use_accuracy:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(numticks=3))
        else:
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    else:
        # Hide y-axis labels for all but the leftmost subplot
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # Make axis lines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    title = ','.join(f'{panel_prefix[k](v)}' for k, v in zip(panel_list, panel))
    ax.set_title(title)
    ax.grid(True)
    
    # Set aspect ratio to be square
    ax.set_box_aspect(1)
    
# all red dots after all lines have been drawn
for ax, x, y in red_dots:
    ax.plot(x, y, 'ro', markersize=3, zorder=10)

# Add the universal legend to the right of all subplots
fig.legend(all_lines, all_labels, bbox_to_anchor=(1.01, 0.5), 
          loc='center left', borderaxespad=0)

plt.savefig('sweep_curves.png', bbox_inches='tight', dpi=300)
plt.close()