DATASETS = {}


def register_dataset(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        DATASETS[name] = cls
        return cls

    return wrapper
