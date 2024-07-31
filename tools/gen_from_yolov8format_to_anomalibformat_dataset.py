import os
import shutil

# Base path for the dataset
base_path = '../datasets/Custom_Dataset/anomaly-detection.cardboard.yolov8'


# Function to classify and move images based on label content
def classify_images(images_path, labels_path, normal_dir, abnormal_dir):
    # Create target directories if they don't exist
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)

    # Iterate over all label files in the labels directory
    for label_file in os.listdir(labels_path):
        # Generate the corresponding image file name
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_path, image_file)

        # Path to the current label file
        label_file_path = os.path.join(labels_path, label_file)

        # Check if the label file has content
        if os.path.getsize(label_file_path) > 0:  # If label file has content, it's an abnormal image
            shutil.move(image_path, os.path.join(abnormal_dir, image_file))
        else:  # If label file is empty, it's a normal image
            shutil.move(image_path, os.path.join(normal_dir, image_file))


# Directories to process
base_dirs = ['train', 'valid', 'test']

for base_dir in base_dirs:
    images_path = os.path.join(base_path, base_dir, 'images')
    labels_path = os.path.join(base_path, base_dir, 'labels')
    normal_dir = os.path.join(base_path, base_dir, 'good')
    abnormal_dir = os.path.join(base_path, base_dir, 'defect')

    # Process each directory (train, valid, test)
    classify_images(images_path, labels_path, normal_dir, abnormal_dir)

    # Remove the original images and labels directories
    shutil.rmtree(images_path)
    shutil.rmtree(labels_path)

print("Dataset conversion and cleanup complete!")
