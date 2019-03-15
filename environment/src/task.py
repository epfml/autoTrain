from copy import deepcopy
from typing import Iterable, List

import torch
import torchvision

from .metrics import MeanAccumulator


class Batch:
    def __init__(self, x, y):
        self._x = x
        self._y = y


class Done(Exception):
    pass


class Task:
    """
    Interface for an optimizaiton task
    Interface:
        The following methods are exposed to the challenge participants:
            - `train_iterator`: returns an iterator of `Batch`es from the training set,
            - `batchLoss`: evaluate the function value of a `Batch`,
            - `batchLossAndGradient`: evaluate the function value of a `Batch` and compute the gradients,
            - `test`: compute the test loss of the model on the test set.
        The following attributes are exposed to the challenge participants:
            - `default_batch_size`
            - `target_test_loss`

        See documentation below for more information.

    Example:
        See train_sgd.py for an example of a Task in use.
    """

    def __init__(self):
        self.target_test_loss = None  # float
        self.default_batch_size = None  # integer

    def train_iterator(self, batch_size: int, shuffle: bool) -> Iterable[Batch]:
        """Create a dataloader serving `Batch`es from the training dataset.

        Example:
            >>> for batch in task.train_iterator(batch_size=32, shuffle=True):
            ...     batch_loss, gradients = task.batchLossAndGradient(batch)
        """
        raise NotImplementedError()

    def batchLoss(self, batch: Batch) -> float:
        """
        Evaluate the loss on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        """
        raise NotImplementedError()

    def batchLossAndGradient(self, batch: Batch) -> (float, List[torch.Tensor]):
        """
        Evaluate the loss and its gradients on a batch.
        If the model has batch normalization or dropout, this will run in training mode.

        Returns:
            - function value (float)
            - gradients (list of tensors in the same order as task.state())
        """
        raise NotImplementedError()

    def test(self, state) -> float:
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        raise NotImplementedError()
