## General Usage
### CLI

YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, i.e. `imgsz=640`. See the YOLOv8 [CLI Docs](https://docs.ultralytics.com/usage/cli) for examples.

### Python

YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

## Custom Dataset
1. Set up .yaml file
```yaml
# use absolute path, relative path may have error report
path: /home/jinyao/PycharmProjects/IADBE/datasets/Custom_Dataset/anomaly-detection.v2i.yolov8
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['anomaly']
```

2. Train
<details>
<summary>Python</summary>

```python
from ultralytics import YOLO
import os
os.environ["WANDB_MODE"] = "dryrun"

# Define the path to your dataset YAML configuration file
data_config_path = 'path/to/your/custom_dataset_cardboard_YOLOv8.yaml'

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')  # Replace with your model path if necessary

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

```
</details>

<details>
<summary>CLI</summary>

```shell
yolo task=detect mode=train \
    data=path/to/your/custom_dataset_cardboard_YOLOv8.yaml \
    model=yolov8m.pt \
    epochs=50 \
    imgsz=800 \
    batch=32 \
    workers=4

```

</details>

3. Inference
<details>
<summary>Python</summary>

```python
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('/path/to/your/yolov8_project/train_experiment/weights/best.pt')

# Perform prediction on the specified image
results = model.predict(source='/home/jinyao/PycharmProjects/IADBE/datasets/Custom_Dataset/anomaly-detection.v2i.yolov8/valid/images/20230522124919_png.rf.2dcb75d340f86c8b412d55ce20bd3bc2.jpg')

# Iterate over the results and display or save them
for result in results:
    result.show()  # Display the image with predictions
    result.save(filename='inferenced_image.jpg')  # Save the results to an output directory

```
</details>

<details>
<summary>CLI</summary>

```shell
yolo task=detect mode=predict \
    model=./runs/detect/train13/weights/best.pt \
    source='./datasets/anomaly-detection.v2i.yolov8/valid/images/xxx.jpg'
```
</details>