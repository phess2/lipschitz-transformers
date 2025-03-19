"""
Takes results path and plots heatmap of validation loss over two axes: softmax scale and final scale.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib as mpl

# Directory with result files
path = Path('/data/vision/phillipi/vector/duality/spring2025/modula-v2/experiment')
results_path = path / 'results-7-sweep-lr-0.001'

# Find all json files
json_files = list(results_path.glob('*.json'))

# Set font sizes and style for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Extract data
data = []
for file_path in json_files:
    with open(file_path, 'r') as f:
        result = json.load(f)
        
        # Extract parameters and final validation loss
        final_scale = result['parameters']['final_scale']
        softmax_scale = result['parameters']['softmax_scale']
        
        # Get the last validation loss
        if 'val_losses' in result['results'] and len(result['results']['val_losses']) > 0:
            val_loss = result['results']['val_losses'][-1]
            data.append((final_scale, softmax_scale, val_loss))
        else:
            print(f"Warning: No validation loss found in {file_path.name}")

# Check if we have data
if not data:
    print("No data found!")
    exit(1)

# Get unique values for each axis
final_scales = sorted(set(item[0] for item in data))
softmax_scales = sorted(set(item[1] for item in data))

# Create a grid for the heatmap
loss_matrix = np.full((len(final_scales), len(softmax_scales)), np.nan)

# Fill the grid with validation losses
for final_scale, softmax_scale, val_loss in data:
    i = final_scales.index(final_scale)
    j = softmax_scales.index(softmax_scale)
    loss_matrix[i, j] = min(val_loss, 5)

# Create the heatmap with a larger figure size
plt.figure(figsize=(12, 10))
plt.xlabel('Softmax Scale', fontweight='bold')
plt.ylabel('Final Scale', fontweight='bold')

# Find min and max for better color scaling
valid_losses = loss_matrix[~np.isnan(loss_matrix)]
if len(valid_losses) > 0:
    vmin, vmax = np.min(valid_losses), np.max(valid_losses)
else:
    vmin, vmax = 0, 5

# Create the heatmap with a better colormap
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Create a copy of the matrix and replace NaNs with 5 for display purposes
loss_matrix_display = loss_matrix.copy()
loss_matrix_display[np.isnan(loss_matrix_display)] = 5

# Use the modified matrix for the heatmap
heatmap = plt.pcolormesh(loss_matrix_display, cmap=cmap, norm=norm)

# Add a more detailed colorbar
cbar = plt.colorbar(heatmap, label='Validation Loss', pad=0.02)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Validation Loss', fontweight='bold', fontsize=14)

# Set the x and y ticks
plt.xticks(np.arange(len(softmax_scales)) + 0.5, softmax_scales)
plt.yticks(np.arange(len(final_scales)) + 0.5, final_scales)

# Rotate x-axis labels if there are many values
if len(softmax_scales) > 6:
    plt.xticks(rotation=45, ha='right')

# Add text annotations with loss values - improved readability
for i in range(len(final_scales)):
    for j in range(len(softmax_scales)):
        # Set a background for the text to make it more readable
        val = loss_matrix[i, j]
        color_val = norm(val) if not np.isnan(val) else 5
        text_color = 'black' if color_val > 0.5 else 'white'
        plt.text(j + 0.5, i + 0.5, (f'{val:.3f}' if val < 5 else '>5') if not np.isnan(val) else 'nan',
                    ha='center', va='center', 
                    color=text_color,
                    fontweight='bold',
                    fontsize=11)

# Add a descriptive title with date
plt.suptitle('Val Loss (Scales Set, then Fixed)', fontsize=18, fontweight='bold', y=0.98)

# Save the plot with higher DPI for better quality
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
plt.savefig(path / 'validation_loss_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Heatmap saved as {path / 'validation_loss_heatmap.png'}") 