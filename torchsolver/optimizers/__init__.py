OPTIMIZERS = {}


def register_optimizer(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        OPTIMIZERS[name] = cls
        return cls

    return wrapper
