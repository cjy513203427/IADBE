# 📦 Installation

Anomalib provides two ways to install the library. The first is through PyPI, and the second is through a local installation. PyPI installation is recommended if you want to use the library without making any changes to the source code. If you want to make changes to the library, then a local installation is recommended.

<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with anomalib.

```bash
pip install anomalib
```

This will install Anomalib CLI. Anomalib CLI is a command line interface for training, inference, benchmarking, and hyperparameter optimization. If you want to use the library as a Python package, you can install the library with the following command:

```bash
# Get help for the installation arguments
anomalib install -h

# Install the full package
anomalib install

# Install with verbose output
anomalib install -v

# Install the core package option only to train and evaluate models via Torch and Lightning
anomalib install --option core

# Install with OpenVINO option only. This is useful for edge deployment as the wheel size is smaller.
anomalib install --option openvino
```

# 🧠 Training

Anomalib supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use anomalib off-the-shelf.

<details>
<summary>Training via API</summary>

```python
# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)
```

</details>

<details>
<summary>Training via CLI</summary>

```bash
# Get help about the training arguments, run:
anomalib train -h

# Train by using the default values.
anomalib train --model Patchcore --data anomalib.data.MVTec

# Train by overriding arguments.
anomalib train --model Patchcore --data anomalib.data.MVTec --data.category transistor

# Train by using a config file.
anomalib train --config <path/to/config>
```

</details>

# 🤖 Inference

Anomalib includes multiple inferencing scripts, including Torch, Lightning, Gradio, and OpenVINO inferencers to perform inference using the trained/exported model. Here we show an inference example using the Lightning inferencer. For other inferencers, please refer to the [Inference Documentation](https://anomalib.readthedocs.io).

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
