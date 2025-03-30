import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Custom scale function for compressing the 0-60 range, expanding the 60-100 range
def custom_scale(y):
    return np.where(y <= 60, y * 0.3 / 60, 0.3 + (y - 60) * 0.7 / 40)

# Inverse custom scale function - used for displaying scale labels
def inverse_custom_scale(y):
    return np.where(y <= 0.3, y * 60 / 0.3, 60 + (y - 0.3) * 40 / 0.7)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Logs directory
logs_dir = os.path.join(script_dir, '..', 'logs', 'training_time')
# Output path
output_dir = os.path.join(script_dir, '..', 'output')
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define dataset information (number of samples and number of categories)
dataset_info = {
    'visa': {'samples': 10281, 'categories': 12},
    'mvtec': {'samples': 5354, 'categories': 15},
    'mvtec3d': {'samples': 5312, 'categories': 10},
    'btech': {'samples': 2830, 'categories': 3},
    'kolektor': {'samples': 399, 'categories': 1}
}

# Sort datasets by sample count in descending order
sorted_datasets = sorted(dataset_info.keys(), key=lambda x: dataset_info[x]['samples'], reverse=True)

# Dataset CSV file paths
file_paths = {
    dataset: os.path.join(logs_dir, f"{dataset}.csv")
    for dataset in dataset_info.keys()
}

# Print file paths for debugging
for name, path in file_paths.items():
    print(f"Looking for {name} at: {os.path.abspath(path)}")

# Read CSV files
dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Get all unique model names
all_models = []
for df in dfs.values():
    all_models.extend(df['model'].tolist())
all_models = sorted(list(set(all_models)))

# Create data structure for plotting
auroc_values = {}  # Store AUROC values for each model on each dataset

# Process data
for model in all_models:
    auroc_values[model] = []
    
    for dataset in sorted_datasets:
        df = dfs[dataset]
        if model in df['model'].values:
            # Get AUROC value for the model on the current dataset
            model_row = df[df['model'] == model].iloc[0]
            auroc = float(model_row['image auroc value'])
            
            auroc_values[model].append(auroc)
        else:
            # If the model does not exist in the current dataset, fill with None
            auroc_values[model].append(None)

# Calculate average AUROC for each model
model_avg_auroc = {}
for model in all_models:
    valid_aurocs = [a for a in auroc_values[model] if a is not None]
    if valid_aurocs:
        model_avg_auroc[model] = sum(valid_aurocs) / len(valid_aurocs)
    else:
        model_avg_auroc[model] = 0

# Sort all models by average AUROC
sorted_models = sorted(model_avg_auroc.keys(), key=lambda m: model_avg_auroc[m], reverse=True)
print(f"Found {len(sorted_models)} models: {sorted_models}")

# Create a larger chart to accommodate all 15 models
fig = plt.figure(figsize=(18, 16))
ax = fig.add_subplot(111, polar=True)

# Determine angles
angles = np.linspace(0, 2*np.pi, len(sorted_datasets), endpoint=False).tolist()
# Close the figure
angles += angles[:1]

# Set radar chart direction, starting from above
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set labels for each point
ax.set_xticks(angles[:-1])
ax.set_xticklabels([dataset.upper() for dataset in sorted_datasets], fontsize=14, fontweight='bold')

# Apply custom scale
scaled_rgrids = [custom_scale(val) for val in [0, 20, 40, 60, 70, 80, 90, 95, 100]]
label_rgrids = [0, 20, 40, 60, 70, 80, 90, 95, 100]

# Set y-axis range, apply custom scale
ax.set_ylim(0, 1)  # Scaled range is 0-1
ax.set_rgrids(scaled_rgrids, labels=[f"{val}" for val in label_rgrids], fontsize=12)

# Draw special 60 circle, marked with dashed line
sixty_circle = plt.Circle((0, 0), custom_scale(60), transform=ax.transData._b, 
                        fill=False, edgecolor='gray', linestyle='--', linewidth=1)
ax.add_artist(sixty_circle)

# Mark 60 separator
plt.text(0, custom_scale(60) + 0.02, "60", transform=ax.transData._b, 
        horizontalalignment='center', verticalalignment='bottom', 
        fontsize=12, color='gray')

# Use HSV color space to create a series of clear distinguishable colors
hues = np.linspace(0, 1, len(sorted_models), endpoint=False)
colors = [plt.cm.hsv(h) for h in hues]

# Define line type and marker
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '8', '+', 'x', 'd', '|', '_', '.']

# Store graphic elements for each model, used to create legend
legend_elements = []

# Draw radar chart for each model
for i, model in enumerate(sorted_models):
    values = [v if v is not None else 0 for v in auroc_values[model]]
    # Apply custom scale
    scaled_values = [custom_scale(v) for v in values]
    # Close data
    scaled_values += scaled_values[:1]
    
    # Determine line width and line type
    line_width = 2.5
    line_style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i]
    
    # Draw line
    line = ax.plot(angles, scaled_values, 
            linewidth=line_width, 
            color=color, 
            linestyle=line_style, 
            marker=marker, 
            markersize=8,
            label=model)[0]
    
    # Fill area for all models, using same transparency
    ax.fill(angles, scaled_values, color=color, alpha=0.15)
    
    # Create legend element
    legend_elements.append(
        Line2D([0], [0], color=color, lw=line_width, linestyle=line_style, 
               marker=marker, markersize=8, 
               label=f"{model} (avg: {model_avg_auroc[model]:.1f})")
    )

# Create single legend, containing all models
ax.legend(handles=legend_elements, 
          loc='center left', 
          bbox_to_anchor=(1.05, 0.5),
          fontsize=13, 
          title="Models with Average AUROC", 
          title_fontsize=15)

# Add title
plt.title('AUROC Values for All Models Across Different Datasets', size=18, y=1.05)

# Compact layout
plt.tight_layout()

# Save chart
output_path = os.path.join(output_dir, 'auroc_radar_chart_all_models.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"All models radar chart saved to: {os.path.abspath(output_path)}")

# Save as SVG
svg_path = os.path.join(output_dir, 'auroc_radar_chart_all_models.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"All models radar chart also saved as SVG to: {os.path.abspath(svg_path)}")

plt.close()
