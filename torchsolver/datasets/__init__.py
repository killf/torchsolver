from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

DATASETS = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "ImageFolder": ImageFolder
}


def register_dataset(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        DATASETS[name] = cls
        return cls

    return wrapper
