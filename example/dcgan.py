import torch
from torch import nn, optim
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size=100, num_feature=3136):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.num_feature = num_feature

        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU()
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


class MnistGANModule(ts.Module):
    def __init__(self, z_dimension=100, **kwargs):
        super(MnistGANModule, self).__init__(**kwargs)

        self.z_dimension = z_dimension

        self.g_net = Generator(z_dimension)
        self.d_net = Discriminator()

        self.loss = nn.BCELoss()

        self.g_optimizer = optim.Adam(self.g_net.parameters())
        self.d_optimizer = optim.Adam(self.d_net.parameters())

    def forward_d(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)

        # compute loss of real_img
        real_out = self.d_net(img)
        loss_real = self.loss(real_out, real_label)
        real_score = real_out

        # compute loss of fake_img
        z = torch.randn(N, self.z_dimension, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        loss_fake = self.loss(fake_out, fake_label)
        fake_score = fake_out

        d_loss = loss_real + loss_fake
        d_score = torch.cat([real_score, fake_score], dim=0)
        return d_loss, {"d_loss": float(d_loss), "d_score": float(d_score.mean())}

    def forward_g(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)

        # compute loss of fake_img
        z = torch.randn(N, self.z_dimension, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        g_loss = self.loss(fake_out, real_label)
        g_score = fake_out

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score.mean())}

    def train_step(self, *inputs):
        d_metrics = self.train_step_d(*inputs)
        g_metrics = self.train_step_g(*inputs)

        d_metrics.update(g_metrics)
        return d_metrics

    def train_step_d(self, *inputs):
        d_loss, d_metrics = self.forward_d(*inputs)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_metrics

    def train_step_g(self, *inputs):
        g_loss, g_metrics = self.forward_g(*inputs)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_metrics

    @torch.no_grad()
    def val_epoch(self, *inputs):
        z = torch.randn(32, self.z_dimension, device=self.device)
        img = self.g_net(z)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)

        self.logger.add_images("val/sample", img, global_step=self.global_step)
        self.logger.flush()


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision.transforms import *

    train_data = MNIST("data", train=True, transform=ToTensor())

    MnistGANModule(batch_size=128).fit(train_data=train_data)