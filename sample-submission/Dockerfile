# Your Dockerfile must be based on autotrain/autotrain-env
FROM autotrain/autotrain-env:latest

# Here you can install external dependencies
# Dependencies need to be public and available when we build your image
# Example:
RUN pip3 install pygp

# Put your train.py in /submission directory
COPY ./train.py /submission/train.py
