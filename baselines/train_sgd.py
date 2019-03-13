#!/usr/bin/env python3

import os

import torch
from tqdm import tqdm
from task import Task


def train(task: Task):
    print("Target test loss: {:.3f}".format(task.target_test_loss))

    task = Task()
    batch_size = task.default_batch_size

    learning_rate = 0.1
    weight_decay = 1e-5
    n_epochs = 10

    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch))

        for batch in tqdm(task.train_iterator(batch_size=batch_size, shuffle=True)):
            # Get a batch gradient
            _, df = task.batchLossAndGradient(batch)

            # SGD Update
            for variable, grad in zip(task.state, df):
                variable.mul_(1 - weight_decay)
                variable.add_(-learning_rate, grad)

        test_loss = task.test(task.state)
        print("Test loss at epoch {}: {:.3f}".format(epoch, test_loss))


if __name__ == "__main__":
    task = Task()
    train(task)
