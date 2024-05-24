# 📖 Introduction
It is worth noting that previous research accompanying open-source projects often takes
a lot of time due to problems with environment deployment, and some projects do not work due to versioning
problems. However, in this project the solution will be deployed using Docker or Conda. The primary goal is to provide
researchers with a ready-to-use industrial anomaly detection platform. The platform should be able to
reproduce and identify previous research, be bug free and easy to deploy.

# 📦 Installation on Linux (Ubuntu22.04)

IADBE offers two ways to install the library: Conda and Docker. Use Conda if you want to make changes to dependencies and work in dev mode. 
Use Docker if you want to copy our environment(python, torch...) exactly. We assume you have installed the nvidia driver and CUDA.

<details>
<summary>Install from Conda</summary>
Installing the library with Conda

```bash
# Use of virtual environment is highly recommended
# Using conda
yes | conda create -n IADBE python=3.10
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
Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

```
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
    # metrics is under "anomalib/metrics/"
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
