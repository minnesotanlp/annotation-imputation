# Set up using a base/parent image
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the requirements.txt file, install the necessary packages, and then remove the requirements.txt file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt