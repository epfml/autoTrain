
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "Jean-Baptiste Cordonnier <jean-baptiste.cordonnier@epfl.ch>"

# Install lots of apt packages
RUN apt-get update && apt-get install -y \
  cmake \
  curl \
  git \
  htop \
  locales \
  python3 \
  python3-pip \
  sudo \
  tmux \
  unzip \
  vim \
  wget \
  zsh \
  libssl-dev \
  libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# Set the locale to en_US
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

RUN pip3 install --upgrade pip

# Python packages
RUN pip3 install --upgrade \
  jupyter \
  matplotlib \
  numpy \
  pandas \
  scipy \
  seaborn

# Install pytorch
RUN pip3 install torch==1.0.1 torchvision==0.2.2

# install auto_train package
COPY . /autoTrain
WORKDIR /autoTrain
RUN pip3 install --upgrade .

# move to workspace
RUN mkdir /submission
WORKDIR /submission
