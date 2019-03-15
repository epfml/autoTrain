from copy import deepcopy

import torch


class MeanAccumulator:
    """
    Running average of the values that are 'add'ed
    """

    def __init__(self):
        self.average = None
        self.counter = 0

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        if self.average is None:
            self.average = deepcopy(value)
        else:
            delta = value - self.average
            self.average += delta * weight / self.counter
            if isinstance(self.average, torch.Tensor):
                self.average.detach()

    def value(self):
        """Access the current running average"""
        return self.average
