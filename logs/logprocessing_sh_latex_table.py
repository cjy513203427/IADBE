import re

# processing raw logs, extract metric values and concatenate string.
with open('rawlogs/train_test_mvtec_rkde.log', 'r') as file:
    log_data = file.read()


pattern = re.compile(
    r"--data\.category (\w+) --config"  # 匹配数据集类别
    r"[\s\S]*?"  # 匹配任何字符（包括换行符）零次或多次
    r"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    r"┃        Test metric        ┃       DataLoader 0        ┃\n"
    r"┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩\n"
    r"│        image_AUROC        │\s+([0-9.]+)\s+│\n"  # 匹配 image_AUROC 值
    r"│         image_PRO         │\s+([0-9.]+)\s+│\n"  # 匹配 image_PRO 值
    r"│        pixel_AUROC        │\s+([0-9.]+)\s+│\n"  # 匹配 pixel_AUROC 值
    r"│         pixel_PRO         │\s+([0-9.]+)\s+│\n"  # 匹配 pixel_PRO 值
    r"└───────────────────────────┴───────────────────────────┘"
)

matches = pattern.findall(log_data)

dataset_categories = []
image_AUROC_values = []
image_PRO_values = []
pixel_AUROC_values = []
pixel_PRO_values = []

for match in matches:
    dataset_category = match[0]
    image_AUROC = match[1]
    image_PRO = match[2]
    pixel_AUROC = match[3]
    pixel_PRO = match[4]

    dataset_categories.append(dataset_category)
    image_AUROC_values.append(round(float(image_AUROC) * 100, 2))
    image_PRO_values.append(round(float(image_PRO) * 100, 2))
    pixel_AUROC_values.append(round(float(pixel_AUROC) * 100, 2))
    pixel_PRO_values.append(round(float(pixel_PRO) * 100, 2))


# Print the extracted values
for i in range(len(matches)):
    print()
    print(f"Dataset Category: {dataset_categories[i]}")
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
print(f"Number of dataset categories: {len(dataset_categories)}")
print(f"Number of image_AUROC values: {len(image_AUROC_values)}")
print(f"Number of image_PRO values: {len(image_PRO_values)}")
print(f"Number of pixel_AUROC values: {len(pixel_AUROC_values)}")
print(f"Number of pixel_PRO values: {len(pixel_PRO_values)}")