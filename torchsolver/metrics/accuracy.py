import torch


def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = (y_pred == y_true).float().mean()
    return acc
