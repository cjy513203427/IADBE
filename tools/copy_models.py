import os
import shutil

# copy_models.py is used for copying model from results folder and rename it with the corresponding path
# Specify the root directory
root_dir = "../results/ReverseDistillation/"
results_dir = "../results"
models_dir = os.path.join(results_dir, "models")

# Create models directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Traverse the root directory to find .ckpt files
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".ckpt"):
            # Generate new filename by replacing '/' with '_'
            relative_path = os.path.relpath(os.path.join(subdir, file), results_dir)
            new_filename = relative_path.replace("/", "_")
            new_file_path = os.path.join(models_dir, new_filename)

            # Copy and rename the file
            shutil.copy(os.path.join(subdir, file), new_file_path)

print("All .ckpt files have been copied and renamed in the models directory.")
