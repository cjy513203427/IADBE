import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Custom scale function for training time
def custom_scale(y):
    return np.where(y <= 10000, y / 10000 * 0.5, 0.5 + (y - 10000) / 340000 * 0.5)

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the logs directory
logs_dir = os.path.join(script_dir, '..', 'logs', 'training_time')
# Output path
output_dir = os.path.join(script_dir, '..', 'output')
# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define dataset information (number of samples and categories)
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

# Print the absolute paths for debugging
for name, path in file_paths.items():
    print(f"Looking for {name} at: {os.path.abspath(path)}")

# Read the CSV files
dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Get all unique model names
all_models = []
for df in dfs.values():
    all_models.extend(df['model'].tolist())
all_models = sorted(list(set(all_models)))

# Create data structure for plotting
training_times = {}  # Store training time for each model on each dataset

# Process data
for model in all_models:
    training_times[model] = []
    
    for dataset in sorted_datasets:
        df = dfs[dataset]
        if model in df['model'].values:
            # Get training time for the model on the current dataset
            model_row = df[df['model'] == model].iloc[0]
            training_time = int(model_row['training time'].rstrip('s'))
            
            training_times[model].append(training_time)
        else:
            # If the model does not exist in the current dataset, fill with None
            training_times[model].append(None)

# Plotting
fig, ax = plt.subplots(figsize=(14, 10))

# Each model corresponds to a different color
colors = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '8', '+', 'x']

# Create x-axis positions
x = np.arange(len(sorted_datasets))

# Draw line plots for each model
for i, model in enumerate(all_models):
    # Filter out None values
    valid_indices = [j for j, t in enumerate(training_times[model]) if t is not None]
    valid_x = [x[j] for j in valid_indices]
    valid_times = [training_times[model][j] for j in valid_indices]
    
    if not valid_times:
        continue
    
    # Draw line
    ax.plot(valid_x, valid_times, marker=markers[i % len(markers)], 
             markersize=10, color=colors[i], linewidth=2, label=model)

# Set x-axis label to dataset name and sample count
ax.set_xticks(x)
x_labels = [f"{dataset.upper()}\n({dataset_info[dataset]['samples']} samples)" for dataset in sorted_datasets]
ax.set_xticklabels(x_labels)

# Set y-axis to logarithmic scale for better display of different order of training time
ax.set_yscale('log')

# Add title and labels
ax.set_title('Training Time vs Datasets for Different Models', fontsize=16)
ax.set_xlabel('Datasets (Sorted by Sample Size)', fontsize=14)
ax.set_ylabel('Training Time (seconds, log scale)', fontsize=14)

# Add legend, placed outside the figure on the right
ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

# Compact layout
plt.tight_layout()

# Save plot as a PNG
output_path = os.path.join(output_dir, 'training_time_by_dataset.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: {os.path.abspath(output_path)}")

# Save as SVG
svg_path = os.path.join(output_dir, 'training_time_by_dataset.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Plot also saved as SVG to: {os.path.abspath(svg_path)}")

# Do not display the figure to avoid blocking
# plt.show()
