# 📖 Introduction
It is worth noting that previous research accompanying open-source projects often takes
a lot of time due to problems with environment deployment, and some projects do not work due to versioning
problems. However, in this project the solution will be deployed using Docker or Conda. The primary goal is to provide
researchers with a ready-to-use industrial anomaly detection platform. The platform should be able to
reproduce and identify previous research, be bug free and easy to deploy.

# 📦 Environment Setting

IADBE offers two ways to install the library: Conda and Docker. Use Conda if you want to make changes to dependencies and work in dev mode. 
Use Docker if you want to copy our environment(cuda, nvcc, python...) exactly.

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

# Install requirements.txt
pip install -r requirements.txt

# Or using your favorite virtual environment
# ...

```

This will install Anomalib CLI. Anomalib CLI is a command line interface for training, testing.
```bash
# Get help for the installation arguments
anomalib install -h

# Install the full package
anomalib install

# Install with verbose output
anomalib install -v
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
docker run -it --rm iadbe bash
```
</details>

# 🧠 Training and Testing

IADBE supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use IADBE off-the-shelf.

<details>
<summary>Training and Testing via API</summary>
A train_test_xxx.py file looks like this. Run it with your IDE or <code>python train_test_xxx.py</code> to start training.

```python
# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Padim()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

# Test the model
engine.test(datamodule=datamodule, model=model)
```

</details>

<details>
<summary>Training and Testing via CLI</summary>
A train_test_xxx.sh file looks like this. Run it with <code>bash train_test_xxx.sh</code> to start training.

```bash
anomalib train --data anomalib.data.MVTec --data.category transistor --config <path/to/config>
```
For the futher use of anomalib cli, you can retrieve [Training via CLI from Training](https://github.com/openvinotoolkit/anomalib?tab=readme-ov-file#-training)  

</details>
