# 📖 Introduction
It is worth noting that previous research accompanying open-source projects often takes
a lot of time due to problems with environment deployment, and some projects do not work due to versioning
problems. However, in this project the solution will be deployed using Docker or Conda. The primary goal is to provide
researchers with a ready-to-use industrial anomaly detection platform. This platform based on [anomalib](https://github.com/openvinotoolkit/anomalib) should be able to
reproduce and identify previous research, be bug free and easy to deploy.

# 📦 Installation 
Tested on Linux (Ubuntu22/20), Windows (Win11/10) ✅

IADBE offers two ways to install the library: Conda and Docker. Use Conda if you want to make changes to dependencies and work in dev mode. 
Use Docker if you want to copy our environment(python, torch...) exactly. ⚠️We assume that you have installed the nvidia driver and CUDA. Otherwise, you can train on CPU.

<details>
<summary>Install from Conda</summary>
Installing the library with Conda

```bash
# Use of virtual environment is highly recommended
# Using conda
conda create -n IADBE python=3.10
conda activate IADBE

# Clone the repository and install in editable mode
git clone https://github.com/cjy513203427/IADBE.git
cd IADBE

# Install anomalib
pip install anomalib

# Install the full package, this will install Anomalib CLI. Anomalib CLI is a command line interface for training, testing.
anomalib install

# Or using your favorite virtual environment
# ...

```
</details>
    
<details>
<summary>Install from Docker</summary>
Installing the library with Docker

```bash
# Clone the repository and install in editable mode
git clone https://github.com/cjy513203427/IADBE.git
cd IADBE

# Build docker image
docker build -t iadbe .
# Run docker container
docker run --gpus all -it --rm iadbe bash
```
</details>
You can either use it as a virtual machine with the same command to train, test and inference or set docker env as your external environment.

# 🧠 Training and Testing

IADBE supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use IADBE off-the-shelf.

<details>
<summary>Training and Testing via API</summary>
A train_test_xxx.py file looks like this. Run it with your IDE or <code>python train_test_xxx.py</code> to start training default with whole MVTec dataset.

```python
import logging
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Padim

# configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

datasets = ['screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor',
            'metal_nut', 'bottle', 'hazelnut', 'leather']

for dataset in datasets:
    logger.info(f"================== Processing dataset: {dataset} ==================")
    model = Padim()
    datamodule = MVTec(category=dataset, num_workers=0, train_batch_size=256,
                       eval_batch_size=256)
    engine = Engine(pixel_metrics=["AUROC", "PRO"], image_metrics=["AUROC", "PRO"], task=TaskType.SEGMENTATION)

    logger.info(f"================== Start training for dataset: {dataset} ==================")
    engine.fit(model=model, datamodule=datamodule)

    logger.info(f"================== Start testing for dataset: {dataset} ==================")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
    )
```

</details>

<details>
<summary>Training and Testing via CLI</summary>
A train_test_xxx.sh file looks like this. Run it with <code>bash train_test_xxx.sh</code> to start training default with whole MVTec dataset.

```bash
#!/bin/bash

datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
config_file="./configs/models/padim.yaml"

for dataset in "${datasets[@]}"
do
    command="anomalib train --data anomalib.data.MVTec --data.category $dataset --config $config_file"
    echo "Running command: $command"
    # Excute command
    $command
done

```
For the futher use of anomalib cli, you can retrieve [Training via CLI from Training](https://github.com/openvinotoolkit/anomalib?tab=readme-ov-file#-training)  

</details>

# 🤖 Inference

Anomalib includes multiple inferencing scripts, including Torch, Lightning, Gradio, and OpenVINO inferencers to perform inference using the trained/exported model. Here we show an inference example using the Lightning inferencer.

<details>
<summary>Inference via API</summary>

The following example demonstrates how to perform Lightning inference by loading a model from a checkpoint file.

```python
# Assuming the datamodule, model and engine is initialized from the previous step,
# a prediction via a checkpoint file can be performed as follows:
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```

</details>

<details>
<summary>Inference via CLI</summary>

```bash
# To get help about the arguments, run:
anomalib predict -h

# Predict by using the default values.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path <path/to/model.ckpt>

# Predict by overriding arguments.
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTec \
                 --ckpt_path <path/to/model.ckpt>
                 --return_predictions

# Predict by using a config file.
anomalib predict --config <path/to/config> --return_predictions
```

</details>


