
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "Jean-Baptiste Cordonnier <jean-baptiste.cordonnier@epfl.ch>"

# Install lots of apt packages
RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  unzip \
  libssl-dev \
  libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade --no-cache-dir pip

# Python packages
RUN pip3 install --upgrade --no-cache-dir \
  numpy==1.16.2 \
  torch==1.0.1 \
  torchvision==0.2.2

# install auto_train package
COPY . /autoTrain
WORKDIR /autoTrain
RUN pip3 install --upgrade .

# move to workspace
RUN mkdir /submission
WORKDIR /submission

CMD train.py
