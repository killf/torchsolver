MODELS = {}


def register_model(name: str = None):
    def wrapper(cls):
        nonlocal name

        if name is None:
            name = cls.__name__

        MODELS[name] = cls
        return cls
    
    return wrapper
