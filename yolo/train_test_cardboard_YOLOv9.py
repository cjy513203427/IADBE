from ultralytics import YOLO
import os
os.environ["WANDB_MODE"] = "dryrun"

# Define the path to your dataset YAML configuration file
data_config_path = '/home/jinyao/PycharmProjects/IADBE/configs/data/custom_dataset_cardboard_YOLOv8.yaml'

# Load the YOLOv9 model
model = YOLO('yolov9s.pt')  # Replace with your model path if necessary

# Train the model
model.train(
    data=data_config_path,       # Path to your dataset YAML file
    epochs=50,                   # Number of epochs for training
    imgsz=800,                   # Image size for training
    batch=32,                    # Batch size
    workers=4,                   # Number of worker threads for data loading
    project='yolov8_project',    # Valid project name for wandb
    name='train_experiment',     # Experiment name for wandb
    save=True,                   # Whether to save the trained model
    save_period=-1,              # Save the model every N epochs, -1 means save only at the end
    cache=False,                 # Whether to cache images for faster training
    device=None,                 # Device to use for training (e.g., 'cpu' or 'cuda:0')
    amp=True                     # Use automatic mixed precision for faster training
)
