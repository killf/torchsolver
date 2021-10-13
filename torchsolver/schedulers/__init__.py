SCHEDULERS = {}


def register_scheduler(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        SCHEDULERS[name] = cls
        return cls

    return wrapper
