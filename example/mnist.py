import torch
from torch import nn

from torchsolver.models import register_model
from torchsolver.solver import Solver
from torchsolver.metrics import accuracy
from torchsolver.config import Config


@register_model()
class LeNet(nn.Module):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.act = nn.ReLU()

        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(512, classes_num)

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)

        x = torch.softmax(x, dim=-1)
        return x


class MnistSolver(Solver):
    def forward(self, img, label):
        pred = self.model(img)

        acc = accuracy(pred, label)
        if self.is_training:
            loss = self.loss(pred, label)
            return loss, {"loss": float(loss), "acc": float(acc)}
        else:
            return float(acc), {}


if __name__ == '__main__':
    from torchvision.transforms import *

    cfg = Config()

    cfg.train_data_name = "MNIST"
    cfg.train_data_args.root = "data"
    cfg.train_data_args.train = False
    cfg.train_data_args.transform = ToTensor()

    cfg.val_data_name = "MNIST"
    cfg.val_data_args.root = "data"
    cfg.val_data_args.train = False
    cfg.val_data_args.transform = ToTensor()

    cfg.model_name = "LeNet"
    cfg.model_args.classes_num = 10

    cfg.loss_name = "CrossEntropyLoss"
    cfg.optimizer_name = "Adam"

    MnistSolver(cfg).train()
