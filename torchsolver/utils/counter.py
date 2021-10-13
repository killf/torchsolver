import torch
import numpy as np

__ALL__ = ["Counter"]


class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            self.set(key, value)

    def set(self, key, value):
        if key not in self.data:
            self.data[key] = []

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        self.data[key].append(value)

    def get(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])

    def __getattr__(self, key):
        return self.get(key)
