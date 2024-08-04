import re

# processing raw logs, extract metric values and concatenate string.
with open('../logs/rawlogs/train_test_visa_reverse_distillation.log', 'r') as file:
    log_data = file.read()


pattern = re.compile(
    r"Start testing for dataset: (.*?)\n"  # Match dataset categories
    r"[\s\S]*?"  # Match any character (including line breaks) zero or more times
    r"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    r"┃        Test metric        ┃       DataLoader 0        ┃\n"
    r"┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
    r"│        image_AUROC        │\s+([0-9.]+)\s+│\n"  # Match image_AUROC
    r"│         image_PRO         │\s+([0-9.]+)\s+│\n"  # Match image_PRO
    r"│        pixel_AUROC        │\s+([0-9.]+)\s+│\n"  # Match pixel_AUROC
    r"│         pixel_PRO         │\s+([0-9.]+)\s+│\n"  # Match pixel_PRO
    r"└───────────────────────────┴───────────────────────────┘"
)

matches = pattern.findall(log_data)

dataset_names = []
image_AUROC_values = []
image_PRO_values = []
pixel_AUROC_values = []
pixel_PRO_values = []

for match in matches:
    dataset_name = match[0]
    image_AUROC = match[1]
    image_PRO = match[2]
    pixel_AUROC = match[3]
    pixel_PRO = match[4]

    dataset_names.append(dataset_name)
    image_AUROC_values.append(round(float(image_AUROC) * 100, 2))
    image_PRO_values.append(round(float(image_PRO) * 100, 2))
    pixel_AUROC_values.append(round(float(pixel_AUROC) * 100, 2))
    pixel_PRO_values.append(round(float(pixel_PRO) * 100, 2))


# Print the extracted values
for i in range(len(matches)):
    print()
    print(f"Dataset: {dataset_names[i]}")
    print(f"Test metric          Result")
    print(f"----------------------------------")
    print(f"image_AUROC          {image_AUROC_values[i]:.2f}")
    print(f"image_PRO            {image_PRO_values[i]:.2f}")
    print(f"pixel_AUROC          {pixel_AUROC_values[i]:.2f}")
    print(f"pixel_PRO            {pixel_PRO_values[i]:.2f}")

# Prepare LaTeX string
latex_values = " & ".join([f"{value:.2f}" for value in image_AUROC_values])
print(f"LaTeX string: {latex_values}")


# Print the counts
print(f"Number of matches: {len(matches)}")
print(f"Number of dataset names: {len(dataset_names)}")
print(f"Number of image_AUROC values: {len(image_AUROC_values)}")
print(f"Number of image_PRO values: {len(image_PRO_values)}")
print(f"Number of pixel_AUROC values: {len(pixel_AUROC_values)}")
print(f"Number of pixel_PRO values: {len(pixel_PRO_values)}")