from torch import nn

LOSSES = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "BCELoss": nn.BCELoss
}


def register_loss(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        LOSSES[name] = cls
        return cls

    return wrapper
