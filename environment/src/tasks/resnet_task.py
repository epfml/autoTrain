from .. import Task

import torchvision


class ResNetTask(Task):
    def __init__(self):
        super(ResNetTask, self).__init__(target_test_loss=0.6, time_to_converge=1000)

    def _create_model(self):
        return torchvision.models.resnet18()

    def _get_dataset(self, data_root="./data"):
        """Create train and test datasets"""
        dataset = torchvision.datasets.CIFAR10

        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)

        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
        test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

        return training_set, test_set
