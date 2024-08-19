# Use PyTorch official mirror with specific version and CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# set working directory
WORKDIR /app

# copy project files to docker container
COPY . .

# install libgl1-mesa-glx package
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# install libglib2.0-0 package
RUN apt-get install -y libglib2.0-0

# install anomalib
RUN pip install anomalib==1.1.0

# Install the full package
RUN anomalib install

# Install ultralytics
RUN pip install ultralytics

CMD ["bash"]
