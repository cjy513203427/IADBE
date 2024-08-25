import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Custom scale function
def custom_scale(y):
    return np.where(y <= 10000, y / 10000 * 0.5, 0.5 + (y - 10000) / 340000 * 0.5)

# Function to check overlap between annotations based on training time
def is_overlapping(training_time, existing_annotations, threshold=15000):
    for ex in existing_annotations:
        if abs(training_time - ex) < threshold:
            return True
    return False

# Data for CSV files
file_paths = {
    'btech': '../logs/training_time/btech.csv',
    'kolektor': '../logs/training_time/kolektor.csv',
    'mvtec': '../logs/training_time/mvtec.csv',
    'mvtec3d': '../logs/training_time/mvtec3d.csv',
    'visa': '../logs/training_time/visa.csv'
}

# Read the CSV files
dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Plotting
plt.figure(figsize=(12, 8))

for label, df in dfs.items():
    # Strip 's' from training time and convert to int
    df['training time'] = df['training time'].str.rstrip('s').astype(int)

    # Apply custom scaling to y values
    scaled_training_time = custom_scale(df['training time'])
    plt.plot(df['model'], scaled_training_time, marker='o', label=label)

    # Annotating each point with its corresponding image auroc value
    existing_annotations = []
    for i, (txt, model, training_time, scaled_time) in enumerate(zip(df['image auroc value'], df['model'], df['training time'], scaled_training_time)):
        offset_x = -20 if i % 2 == 0 else 20
        offset_y = 10 if i % 2 == 0 else -10

        # Print debug information
        print(f"Checking model: {model}, training time: {training_time}, scaled time: {scaled_time:.2f}")

        # Check for overlap with existing annotations based on training time
        if is_overlapping(training_time, existing_annotations):
            print(f"Skipping annotation for {model} with training time {training_time} (scaled: {scaled_time:.2f}) due to overlap")
            continue

        # Add annotation
        plt.annotate(f'{txt}', (df['model'][i], scaled_time),
                     textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=8)

        # Record this annotation to check for future overlaps
        existing_annotations.append(training_time)

# Adjusting y-axis labels
y_ticks = [0, 5000, 10000, 50000, 100000, 200000, 300000, 350000]
plt.yticks(custom_scale(np.array(y_ticks)), y_ticks)

# Adding titles and labels
plt.title('Training Time and Image AUROC Value for Different Models')
plt.xlabel('Model Name')
plt.ylabel('Training Time (s)')
plt.xticks(rotation=90)
plt.legend(title='Datasets')
plt.tight_layout()

# Save plot as a PDF
# plt.savefig('training_time_vs_image_auroc.pdf', format='pdf')

# Show plot
plt.show()
