import torch
import numpy as np

__ALL__ = ["Counter"]


class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        if key not in self.data:
            self.data[key] = []

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        self.data[key].append(value)

    def __getitem__(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)
