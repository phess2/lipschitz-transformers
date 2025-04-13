import json
import pickle
from pathlib import Path
import imageio
import os

# Create a matplotlib plot of training curves
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator
from collections import defaultdict


results_dir = Path('results')
USE_CACHE = True

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
    if data == 'cifar10' and False:
        return f"{x:.1e}" # scientific notation
    else:
        return f"{int(x)}" if x == int(x) else f"{x:.1f}"

# Cache the results to speed up plotting on the same data
cache_file = results_dir / 'cached_results.pkl'
def need_to_rebuild_cache():
    if not USE_CACHE:
        return True
    cache_mtime = cache_file.stat().st_mtime
    json_files = list(results_dir.glob('*.json'))
    return any(f.stat().st_mtime > cache_mtime for f in json_files)

if not cache_file.exists() or need_to_rebuild_cache():
    results = []
    for file in results_dir.glob('*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            key_renaming = {
                'wd': 'weight_decay',
                'lr': 'learning_rate',
            }
            result = {}
            for key in data['parameters']:
                result[key_renaming.get(key, key)] = data['parameters'][key]
            for key in data['results']:
                result[key] = data['results'][key]
            results.append(result)
    print(f"Loaded {len(results)} results")
    
    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
else:
    with open(cache_file, 'rb') as f:
        results = pickle.load(f)

# Choose properties to make separate panels for, including an optional direct filter for all panels
panel_list = ['optimizer', 'wd_lr_power']
panel_filter = lambda x: x['schedule'] == 'linear'
panels = sorted(list(set(tuple(r[axis] for axis in panel_list) for r in results if panel_filter(r))))

print("Panels:", panels)

# Choose what the color bar will sweep over
x_string = 'weight_decay'  # width, depth, batch_size         I AM MAKING PLOT GIF WORK
x_string_title = 'Weight Decay'  # Width, Depth, Batch Size
data = 'cifar10'

use_test_loss = True
use_accuracy = False
aggregator = lambda x: x[-1]   # this is the brutal honesty option
#aggregator = lambda x: max(x) if use_accuracy else min(x)   # this one papers over overfitting

GIF_MODE = False
FPS = 5
STEP_INCREMENT = 1 if use_test_loss or use_accuracy else 20  # we only record data every 100 steps anyway, unless it's training loss
OUTPUT_DIR = Path('gif_frames')

loss_string = 'Test' if use_test_loss or use_accuracy else 'Training'
loss_string_unit = 'Accuracy' if use_accuracy else 'Loss'
model_string = 'MLP' if data == 'cifar10' else 'Transformer'
dataset_string = 'CIFAR-10' if data == 'cifar10' else 'Shakespeare'
history_string = 'accuracies' if use_accuracy else ('val_losses' if use_test_loss else 'losses')

panel_prefix = {
    'optimizer': lambda x: x.capitalize(),
    'pre_dualize': lambda x: 'Pre' if x else '',
    'project': lambda x: 'Proj' if x else 'NoProj',
    'weight_decay': lambda x: f'wd{x:.2f}',
    'final_scale': lambda x: f'fs{int(x)}',
    'wd_lr_power': lambda x: f'alpha{x:.1f}',
}

ylims = {  # keys are (data, use_accuracy)
    ('shakespeare', False): (1, 3),
    ('shakespeare', True): (20, 70),
    ('cifar10', False): (1.15, 2.5),
    ('cifar10', True): (10, 60),
}

def plot_frame(cur_step=None, save_path=None):
    """Wrap the entire plotting code in one function for GIF mode."""

    # Create two rows of subplots, one for lr sweeps, one for training curves
    plot_size = 3.2621 * 2.5
    fig = plt.figure(figsize=(plot_size * 1.25, plot_size/1.5))
    gs = gridspec.GridSpec(2, len(panels) + 1,
                        width_ratios=[1.25]*len(panels) + [0],
                        height_ratios=[1, 1.5])
    gs.update(wspace=0.25, hspace=0.4)

    axes_top = []
    axes_bottom = []
    for i in range(len(panels)):
        ax_top = fig.add_subplot(gs[0, i], sharey=axes_top[0] if len(axes_top) > 0 else None)
        ax_bottom = fig.add_subplot(gs[1, i])  # Remove sharey for scatter plots
        axes_top.append(ax_top)
        axes_bottom.append(ax_bottom)
    axes_top = np.array(axes_top)
    axes_bottom = np.array(axes_bottom)

    # Map x_string values to viridis colors
    unique_x_values = sorted(list(set(r[x_string] for r in results)))
    color_values = {val: plt.cm.viridis(i/len(unique_x_values)) 
                   for i, val in enumerate(unique_x_values)}

    # Store lines for the legend and red dots for the extrema losses
    all_lines = []
    all_labels = []
    red_dots = []

    #fig.supxlabel('Learning Rate', y=0.02)
    #fig.supylabel(f'Final {loss_string} {loss_string_unit}', x=0.01)

    for i, panel in enumerate(panels):    
        # Store optimal configurations for training curves
        optimal_configs = []

        ax = axes_top[i]
        # Get unique values for color mapping
        x_values = sorted(list(set(r[x_string] for r in results 
                               if tuple(r[axis] for axis in panel_list) == panel 
                               and panel_filter(r))))
        colors = [color_values[x_value] for x_value in x_values]
        
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
                losses = [aggregator(r[history_string][:cur_step]) for r in lr_results[lr]]
                if use_accuracy:
                    losses = [100 * loss for loss in losses]
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
                           linewidth=2, markersize=4)
            
            # Only store lines/labels for unique lines
            if f'{x_string} {x_value}' not in all_labels:
                all_lines.append(line)
                all_labels.append(f'{x_string} {x_value}')
            
            # Store red dot information for later plotting (minimum average loss)
            min_loss_idx = np.argmax(avg_losses) if use_accuracy else np.argmin(avg_losses)
            optimal_lr = learning_rates[min_loss_idx]
            optimal_configs.append((panel, x_value, optimal_lr, color))
            red_dots.append((ax, optimal_lr, avg_losses[min_loss_idx]))

        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(numticks=3))
        # Force rotation for all x-axis labels and adjust their position
        ax.tick_params(axis='x', rotation=0, labelrotation=0)
        plt.setp(ax.get_xticklabels(), ha='center')  # Adjust horizontal alignment

        ax.set_xlabel('Learning Rate')

        # Configure only the leftmost subplot's y-axis, since all are shared anyway
        if i == 0:
            if not use_accuracy:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(numticks=3))
            else:
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
            ax.set_ylabel(f'Final {loss_string} {loss_string_unit}')
        else:
            # Hide y-axis labels for all but the leftmost subplot
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Make axis lines thicker
        # for spine in ax.spines.values():
        #     spine.set_linewidth(1.5)
        
        title = ','.join(f'{panel_prefix[k](v)}' for k, v in zip(panel_list, panel))
        ax.set_title(title)
        ax.grid(True)

        ax.set_ylim(*ylims[(data, use_accuracy)])
        
        # Set aspect ratio to be square
        # ax.set_box_aspect(1)
    
        # Instead of plotting training curves, create scatter plot of optimal points
        ax = axes_bottom[i]
        
        # Extract optimal points
        optimal_lrs = []
        optimal_x_values = []
        optimal_losses = []
        
        for x_value in x_values:
            # Get all results that match the current panel and x_value
            x_value_results = [r for r in results 
                             if r[x_string] == x_value 
                             and tuple(r[axis] for axis in panel_list) == panel 
                             and panel_filter(r)]
            
            # Group results by learning rate
            lr_results = defaultdict(list)
            for r in x_value_results:
                lr_results[r['learning_rate']].append(r)
            
            # Find optimal learning rate
            learning_rates = sorted(lr_results.keys())
            avg_losses = []
            
            for lr in learning_rates:
                losses = [aggregator(r[history_string][:cur_step]) for r in lr_results[lr]]
                if use_accuracy:
                    losses = [100 * loss for loss in losses]
                avg_losses.append(np.mean(losses))
            
            # Store optimal point
            min_loss_idx = np.argmax(avg_losses) if use_accuracy else np.argmin(avg_losses)
            optimal_lrs.append(learning_rates[min_loss_idx])
            optimal_x_values.append(x_value)
            optimal_losses.append(avg_losses[min_loss_idx])
        
        # Remove zero values since they can't be plotted on log scale
        optimal_x_values = np.array(optimal_x_values)
        optimal_lrs = np.array(optimal_lrs)
        optimal_losses = np.array(optimal_losses)
        nonzero_mask = optimal_x_values != 0
        optimal_x_values = optimal_x_values[nonzero_mask]
        optimal_lrs = optimal_lrs[nonzero_mask]
        optimal_losses = optimal_losses[nonzero_mask]
        
        # Create scatter plot without colors
        ax.scatter(optimal_lrs, optimal_x_values, color='red', s=50)
        
        # Fit line in log-log space
        log_lrs = np.log10(optimal_lrs)
        log_x_values = np.log10(optimal_x_values)
        slope, intercept = np.polyfit(log_lrs, log_x_values, 1)
        
        # Plot fitted line
        x_fit = np.logspace(min(log_lrs), max(log_lrs), 100)
        y_fit = 10**(slope * np.log10(x_fit) + intercept)
        ax.plot(x_fit, y_fit, 'r--', alpha=0.8, label=f'Slope: {slope:.2f}')
        ax.legend()
        
        # Match x-axis limits with top plot
        ax.set_xlim(axes_top[i].get_xlim())
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Learning Rate')
        if i == 0:  # Only leftmost plot needs y-label
            ax.set_ylabel(x_string_title)
        else:
            # Hide y-axis labels for all but the leftmost subplot
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.grid(True)

    # all red dots after all lines have been drawn
    for ax, x, y in red_dots:
        ax.plot(x, y, 'ro', markersize=3, zorder=10)

    # Add the universal legend to the right of all subplots
    fig.legend(all_lines, all_labels, bbox_to_anchor=(0.9, 0.5), 
              loc='center left', borderaxespad=0)
    
    if cur_step is not None:
        fig.suptitle(f'Step {cur_step}/{max_steps}')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig

if GIF_MODE:
    # Create output directory for frames
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find maximum number of steps across all histories
    max_steps = max([
        len(r['accuracy_history']) if use_accuracy else 
        len(r['test_loss_history']) if use_test_loss else 
        len(r['train_loss_history'])
        for r in results
    ])
    
    # Generate frames
    frames = []
    for step in range(1, max_steps // STEP_INCREMENT + 1):
        print(f'Frame {step} of {max_steps // STEP_INCREMENT}')
        save_path = OUTPUT_DIR / f'frame_{step:04d}.png'
        plot_frame(step * STEP_INCREMENT, save_path=save_path)
        frames.append(imageio.v2.imread(save_path))
    
    """
    # Find the maximum dimensions across all saved frames
    max_height = 0
    max_width = 0
    for frame_path in OUTPUT_DIR.glob('frame_*.png'):
        img = imageio.imread(frame_path)
        max_height = max(max_height, img.shape[0])
        max_width = max(max_width, img.shape[1])
    
    # Read and pad all frames to the same size
    frames = []
    for step in range(STEP_INCREMENT, min(max_steps, MAX_FRAMES * STEP_INCREMENT), STEP_INCREMENT):
        frame_path = OUTPUT_DIR / f'frame_{step:04d}.png'
        img = imageio.imread(frame_path)
        # Create a new white background image of the maximum size
        padded = np.full((max_height, max_width, 4), 255, dtype=np.uint8)
        # Center the current frame in the padded image
        y_offset = (max_height - img.shape[0]) // 2
        x_offset = (max_width - img.shape[1]) // 2
        padded[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        frames.append(padded)
    """
    
    # Save GIF
    imageio.mimsave('sweep_curves.gif', frames, fps=FPS)
    
    # Clean up frame files
    for frame_path in OUTPUT_DIR.glob('frame_*.png'):
        frame_path.unlink()
    OUTPUT_DIR.rmdir()
else:
    # Normal mode - just plot the final result
    fig = plot_frame(save_path='sweep_reddots.png')