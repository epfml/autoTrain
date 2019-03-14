from copy import deepcopy
from typing import Iterable, List

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from .metrics import MeanAccumulator


class Batch:
    def __init__(self, x, y):
        self._x = x
        self._y = y


class Done(Exception):
    pass


class Task:
    """
    Example implementation of an optimization task

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
        See train_sgd.py for an example
    """

    def __init__(
        self,
        target_test_loss: float,
        time_to_converge: float,
        default_batch_size: int = 128,
        test_batch_size: int = 100,
        num_workers: int = 2,
    ):
        self.target_test_loss = target_test_loss
        self.default_batch_size = default_batch_size

        self._time_to_converge = time_to_converge  # seconds
        self._test_batch_size = test_batch_size
        self._num_workers = num_workers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.random.manual_seed(42)
        self._model = self._create_model()
        self._model.to(self.device)
        self._model.train()

        self.state = [parameter.data for parameter in self._model.parameters()]

        self._train_set, self._test_set = self._get_dataset()

        self._test_loader = DataLoader(
            self._test_set,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        self._criterion = torch.nn.CrossEntropyLoss()

    def _create_model(self):
        raise NotImplementedError

    def _get_dataset(self):
        raise NotImplementedError

    def train_iterator(self, batch_size: int, shuffle: bool) -> Iterable[Batch]:
        """Create a dataloader serving `Batch`es from the training dataset.

        Example:
            >>> for batch in task.train_iterator(batch_size=32, shuffle=True):
            ...     batch_loss, gradients = task.batchLossAndGradient(batch)
        """
        train_loader = DataLoader(
            self._train_set, batch_size=batch_size, shuffle=shuffle, num_workers=self._num_workers
        )

        def batcher(datum):
            x, y = datum
            x = x.to(self.device)
            y = y.to(self.device)
            return Batch(x, y)

        class _Iterable:
            def __init__(self):
                pass

            def __len__(self):
                # useful to specify the length for tqdm
                return len(train_loader)

            def __iter__(self):
                return map(batcher, iter(train_loader))

        return _Iterable()

    def batchLoss(self, batch: Batch) -> float:
        """
        Evaluate the loss on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        """
        return self._criterion(self._model(batch._x), batch._y).item()

    def batchLossAndGradient(self, batch: Batch) -> (float, List[torch.Tensor]):
        """
        Evaluate the loss and its gradients on a batch.
        If the model has batch normalization or dropout, this will run in training mode.

        Returns:
            - function value (float)
            - gradients (list of tensors in the same order as task.state())
        """
        self._zero_grad()
        f = self._criterion(self._model(batch._x), batch._y)
        f.backward()
        df = [parameter.grad.data for parameter in self._model.parameters()]
        return f.item(), df

    def test(self, state):
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        test_model = self._build_test_model(state)
        mean_f = MeanAccumulator()
        for x, y in self._test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                f = self._criterion(test_model(x), y)
            mean_f.add(f)
        if mean_f.value() < self.target_test_loss:
            raise Done(mean_f.value())
        return mean_f.value()

    def _build_test_model(self, state):
        test_model = deepcopy(self._model)
        test_model.eval()
        for param, new_value in zip(test_model.parameters(), state):
            param.data = new_value.data
        return test_model

    def _zero_grad(self):
        for param in self._model.parameters():
            param.grad = None

