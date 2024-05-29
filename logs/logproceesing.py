import re

# processing raw logs, extract metric values.
with open('rawlogs/train_test_mvtec_cfa.log', 'r') as file:
    log_data = file.read()

pattern = re.compile(
    r"Start testing for dataset: (.*?)\n"  # Match the dataset name
    r"[\s\S]*?"  # Match any character (including newlines) zero or more times
    r"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    r"┃        Test metric        ┃       DataLoader 0        ┃\n"
    r"┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
    r"│        image_AUROC        │\s+(.*?)\s+│\n"  # Match the value of image_AUROC
    r"│         image_PRO         │\s+(.*?)\s+│\n"  # Match the value of image_PRO
    r"│        pixel_AUROC        │\s+(.*?)\s+│\n"  # Match the value of pixel_AUROC
    r"│         pixel_PRO         │\s+(.*?)\s+│\n"  # Match the value of pixel_PRO
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
    image_AUROC_values.append(image_AUROC)
    image_PRO_values.append(image_PRO)
    pixel_AUROC_values.append(pixel_AUROC)
    pixel_PRO_values.append(pixel_PRO)

# Print the counts
print(f"Number of matches: {len(matches)}")
print(f"Number of dataset names: {len(dataset_names)}")
print(f"Number of image_AUROC values: {len(image_AUROC_values)}")
print(f"Number of image_PRO values: {len(image_PRO_values)}")
print(f"Number of pixel_AUROC values: {len(pixel_AUROC_values)}")
print(f"Number of pixel_PRO values: {len(pixel_PRO_values)}")

# Print the extracted values
for match in matches:
    dataset_name = match[0]
    image_AUROC = match[1]
    image_PRO = match[2]
    pixel_AUROC = match[3]
    pixel_PRO = match[4]

    print()
    print(f"Dataset: {dataset_name}")
    print(f"Test metric          Result")
    print(f"----------------------------------")
    print(f"image_AUROC          {image_AUROC}")
    print(f"image_PRO            {image_PRO}")
    print(f"pixel_AUROC          {pixel_AUROC}")
    print(f"pixel_PRO            {pixel_PRO}")
