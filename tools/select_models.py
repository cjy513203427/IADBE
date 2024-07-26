import os
import random
import re

# select_models.py is used for keeping only one version model
# Specify the models directory
models_dir = "../results/models"

# Get all .ckpt files
ckpt_files = [f for f in os.listdir(models_dir) if f.endswith(".ckpt")]

# Use a dictionary to group files by model name
model_versions = {}

# Regular expression to match model name and version
pattern = re.compile(r"(.+)_v\d+_weights_lightning_model\.ckpt")

for file in ckpt_files:
    match = pattern.match(file)
    if match:
        model_name = match.group(1)
        if model_name not in model_versions:
            model_versions[model_name] = []
        model_versions[model_name].append(file)

# Keep one random version for each model
for model_name, versions in model_versions.items():
    file_to_keep = random.choice(versions)
    print(f"Keeping file: {file_to_keep}")

    for file in versions:
        if file != file_to_keep:
            file_path = os.path.join(models_dir, file)
            os.remove(file_path)
            print(f"Deleting file: {file}")

print("One .ckpt file retained per model, others deleted.")
