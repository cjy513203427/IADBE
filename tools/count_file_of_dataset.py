import os

def count_train_images(dataset_directory):
    total_image_count = 0

    # Supported image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Iterate over each subdirectory (e.g., bagel, cable_gland, etc.)
    for item in os.listdir(dataset_directory):
        item_path = os.path.join(dataset_directory, item)

        if os.path.isdir(item_path):
            # Locate the train directory
            train_dir = os.path.join(item_path, 'train')
            if os.path.exists(train_dir):
                # Iterate over all images in the train directory
                for root, dirs, files in os.walk(train_dir):
                    for file in files:
                        # If the file is an image, increment the count
                        if any(file.lower().endswith(ext) for ext in image_extensions):
                            total_image_count += 1

    return total_image_count

# Main function
if __name__ == "__main__":
    # Set the path to the MVTec 3D dataset
    mvtec3d_dataset_path = "../datasets/MVTec3D/"

    # Calculate the number of images
    total_images = count_train_images(mvtec3d_dataset_path)

    # Output the result
    print(f"Total number of images in the MVTec 3D dataset: {total_images}")
