import torchsolver as ts
from torchvision.datasets import MNIST
from torchvision.transforms import *

train_data = MNIST("data", train=True, transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
ts.nets.DCGANNet(task_name="dcgan_MNIST", epochs=50, batch_size=128).fit(train_data=train_data)
