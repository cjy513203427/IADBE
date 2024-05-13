# Use Python 3.10 official mirror
FROM python:3.10

# set working directory
WORKDIR /app

# copy project files to docker container
COPY . .

# intall libgl1-mesa-glx package
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# intall libglib2.0-0 package
RUN apt-get install -y libglib2.0-0

# install dependencies
RUN pip install -r requirements.txt

# Install the full package
RUN anomalib install

CMD ["bash"]