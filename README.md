# AutoTrain

*NeurIPS 2019 - Proposed Challenge*

This AutoTrain competition has the goal to fully automatize the training
process of deep learning models. Submitted optimizers will be benchmarked on a set of
unknown architecture/dataset pairs, in a live competition at NeurIPS 2019 (following
an earlier trial round event). The winning optimizers will be made publicly available as
open source and bring significant value to practitioners and researchers, by removing
the need of expensive hyperparameter tuning, and by providing fair benchmarking of
all optimizers.



## Rules

### Evaluation

The participants are required to submit code for an AutoTrain optimizer which will be uploaded to the challenge platform. This code will be run on previously unseen architecture / data set pairs. The submission is executed until the target test accuracy is reached, or it stops otherwise, or it consumes more than the maximum allowed resources (time or memory). Submissions are ranked on average time-to-accuracy (on the specified standard cloud instance), since this corresponds best to cost in real-world usecases. More precisely, the time-to-accuracy over the test-cases will be normalized by the different baseline times for each test-case. In effect, with normalization, the final score is the harmonic mean of the time-to-accuracy speed-up for the different architecture / dataset pairs (speedup is defined as the time ratio compared to the baseline).

### Unseen architecture / dataset pairs.

 The unseen architecture / dataset pairs on which they will be judged will be modifications of the sample architecture / dataset pairs provided beforehand to the participants. Hence, it is sufficient to ensure that the submitted code does not exceed maximum resources on the provided sample architecture / dataset. Most importantly, the number of weights on the unknown network will not exceed the one in the provided example models. Further, the range of the following high level characteristics of the unseen architecture / dataset pair will be of the same order of magnitude as that of their sample counterparts: i) number of hyper-parameters, ii) time required for forward pass, iii) time required for backprop, and iv) size of the training data. However, the exact value of each of these characteristics might not match that of any of the provided samples. Further, the architecture of the model, though similar, might not exactly match any of the provided samples.

### Code Interface

The AutoTrain optimizer can access the (train) data via querying consecutive mini-batches of desired size. It is then allowed to make as many calls as desired (within the resource limits) to the following oracles, which take as input a subset of the data indices and output:

1. the loss value of the network on the corresponding mini-batch (i.e. inference),
2. the result of the backprop on the corresponding mini-batch.

The optimizer can update the weights as many times as desired. Note that one can utilize these oracles to internally perform train/validation split, select mini-batch size, perform restarts, etc. if desired. Access to the interface will be synchronous—multiple simultaneous queries are ignored. The interface will be be based on PyTorch.

### Additional rules.

- Each submission should be accompanied by an informative description (commented code, README, and writeup of the approach).
- Source code of the submission must be provided, along with a Dockerfile to build the container with all necessary dependencies. Code can leverage any external open source libraries as long as all such dependencies are well documented.
- Use of external communication, pretraining, or manipulation of the provided oracles (such as backprop) is not allowed, only the use of the results (vectors) of the oracles is permitted.
- We require the winning submission to be publicly released to ensure reproducibility and impact on the community.



## Implementation

You are required to submit a `train.py` file defining the function

```python
def train(task: Task):
	"""Train the (model, dataset) pair associated to the task.

	Args:
		task [Task]: task to optimize. Refer to `src/task.py` for available functions.
	"""
```

Examples of implementation of `SGD` and `Adam` are provided in `src/train_sgd.py` and `src/train_adam.py`.
