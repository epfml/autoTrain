#!/usr/bin/env python3

import math

import torch
from tqdm import tqdm as progress_bar

from auto_train import Task


def train(task: Task):
    print("Target test loss: {:.3f}".format(task.target_test_loss))

    batch_size = task.default_batch_size

    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    epsilon = 1e-08
    weight_decay = 1e-5
    n_epochs = 10

    # Adam Initialization
    first_moment = [torch.zeros_like(param) for param in task.state]
    second_moment = [torch.zeros_like(param) for param in task.state]
    t = 0

    for epoch in range(n_epochs):
        print("Epoch {}".format(epoch))

        for batch in progress_bar(task.train_iterator(batch_size=batch_size, shuffle=True)):
            # Get a batch gradient
            _, df = task.batchLossAndGradient(batch)

            # Adam Update
            t += 1
            lr = learning_rate * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
            for m1, m2, variable, grad in zip(first_moment, second_moment, task.state, df):
                m1 = beta1 * m1 + (1 - beta1) * grad
                m2 = beta2 * m2 + (1 - beta2) * grad * grad
                variable.mul_(1 - weight_decay)
                variable.add_(-lr, m1 / (torch.sqrt(m2 + epsilon)))

        test_loss = task.test(task.state)
        print("Test loss at epoch {}: {:.3f}".format(epoch, test_loss))


if __name__ == "__main__":
    task = Task()
    train(task)
