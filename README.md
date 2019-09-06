# AutoTrain

AutoTrain challenges you to submit optimizers that work reliably on any deep learning task without task-specific tuning.
It separates AutoML into (1) fully automatic training of a model and (2) model selection, and tackles the first aspect.

Your submissions will be benchmarked on a secret set of architecture/dataset pairs inspired by common deep learning tasks.
The optimizers need to achieve a target test loss as fast as possible. The fastest on average wins the competition.

The winning optimizers will be made publicly available as
open source and bring significant value to practitioners and researchers, by removing
the need of expensive hyperparameter tuning, and by providing fair benchmarking of
all optimizers.

## Timeline

| Date                |                                       |
|---------------------|---------------------------------------|
| October 1, 2019                  | Submission interface is stable.<br>We start accepting submissions |
| December 15, 2019  at 23:59 GMT  | Submission deadline      |
| January 25-29, 2020 | Applied Machine Learning Days<br>Publication of the results |


## Submission

You are required to submit a ZIP file before the deadline to [autotrain@groupes.epfl.ch](mailto:autotrain@groupes.epfl.ch) containing:

-   `README.md`: team name and team members,
-   `train.py`: code of your optimizer,
-   `report.pdf`: 4 pages (two-columns) report describing your submission.

You can refer to [train.py](./train.py) for a sample submission.


## Rules

### Evaluation

The participants are required to submit code for an AutoTrain optimizer which will be uploaded to the challenge platform. This code will be run on previously unseen architecture / dataset pairs. The submission is executed until the target test loss is reached, or it stops otherwise, or it consumes more than the maximum allowed resources (time or memory). Submissions are ranked on average time-to-accuracy (on the specified standard cloud instance), since this corresponds best to cost in real-world usecases. More precisely, the time-to-accuracy over the test cases will be normalized by the different baseline times for each test-case. In effect, with normalization, the final score is the harmonic mean of the time-to-accuracy speed-up for the different architecture / dataset pairs (speedup is defined as the time ratio compared to the baseline).

### Unseen architecture / dataset pairs

 The unseen architecture / dataset pairs on which they will be judged will be modifications of the sample architecture / dataset pairs provided beforehand to the participants. Hence, it is sufficient to ensure that the submitted code does not exceed maximum resources on the provided sample architecture / dataset. Most importantly, the number of weights on the unknown network will not exceed the one in the provided example models. Further, the range of the following high level characteristics of the unseen architecture / dataset pair will be of the same order of magnitude as that of their sample counterparts: i) number of parameters, ii) time required for forward pass, iii) time required for backprop, and iv) size of the training data. However, the exact values of each of these characteristics might not match that of any of the provided samples. Further, the architecture of the model, though similar, might not exactly match any of the provided samples.

### Task interface

The AutoTrain optimizer can access the (train) data via querying consecutive mini-batches of desired size. It is allowed to make as many calls as desired (within the resource limits) to the following oracles, which take a minibatch as input and output:

1. the loss value of the network on the corresponding mini-batch (i.e. inference),
2. the result of the backprop on the corresponding mini-batch.

The optimizer can update the weights as many times as desired. Access to the interface will be synchronous—multiple simultaneous queries are ignored. The interface is based on PyTorch.

The optimizer can query the test loss via `test_loss = task.test(task.state)`. The current test loss will be compared against the target __only__ when you call this function.

The optimizer also has access to the target test loss `task.target_test_loss`, and a default batch size `task.default_batch_size` which is guaranteed to not exceed memory limits for SGD and Adam.

### Additional rules

-   Each submission should be accompanied by an informative description (commented code, README, and writeup of the approach).
-   Source code of the submission must be provided. Your optimizer should be implemented in train.py and not use any external dependencies.
-   Use of external communication, pretraining, or manipulation of the provided oracles (such as backprop) is not allowed, only the use of the results (vectors) of the oracles is permitted.
-   We require the winning submission to be publicly released to ensure reproducibility and impact on the community.

### Environment

We will evaluate the submissions on a system with Ubuntu 18.04, Anaconda Python 3.7 and Cuda 10.
You can use the packages `torch`, `numpy`, `scipy` and any other package available in Anaconda Python.

### Optimizer

The submitted `train.py` file must define the function:

```python
def train(task: Task):
    """Train the (model, dataset) pair associated to the task.

    Args:
        task [Task]: task to optimize. Refer to `src/task.py` for available functions.
    """
```
An example is provided in [train.py](./train.py). 
Every time you evaluate the model on the test set (`task.test(task.state)`), you are compared againast the target loss and get a chance to win.

## Organizers

- Thijs Vogels, EPFL
- Sai Praneeth Karimireddy, EPFL
- Jean-Baptiste Cordonnier, EPFL
- Michael Tschannen, ETH Zürich
- Fabian Pedregosa, Google
- Sebastian U. Stich, EPFL
- Sharada Mohanty, EPFL
- Marcel Salathé, EPFL
- Martin Jaggi, EPFL

Contact: autotrain@groupes.epfl.ch
