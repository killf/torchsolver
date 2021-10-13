import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(10, 1000)

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * 28 * 28)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(Generator, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x


class ConditionalDCGAN(ts.Module):
    def __init__(self, z_dim=100, **kwargs):
        super(ConditionalDCGAN, self).__init__(**kwargs)

        self.z_dim = z_dim

        self.g_net = Generator(z_dim)
        self.d_net = Discriminator()

        self.loss = nn.BCELoss()

        self.g_optimizer = optim.SGD(self.g_net.parameters(), lr=0.01)
        self.d_optimizer = optim.SGD(self.d_net.parameters(), lr=0.01)

    def forward_d(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)

        # compute loss of real_img
        real_out = self.d_net(img, label)
        loss_real = self.loss(real_out, real_label)
        real_score = real_out

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z, label)
        fake_out = self.d_net(fake_img, label)
        loss_fake = self.loss(fake_out, fake_label)
        fake_score = fake_out

        d_loss = loss_real + loss_fake
        d_score = torch.cat([real_score, fake_score], dim=0)
        return d_loss, {"d_loss": float(d_loss), "d_score": float(d_score.mean())}

    def forward_g(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z, label)
        fake_out = self.d_net(fake_img, label)
        g_loss = self.loss(fake_out, real_label)
        g_score = fake_out

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score.mean())}

    def train_step(self, img, label):
        label = F.one_hot(label, num_classes=10).float()

        d_metrics = self.train_step_d(img, label)
        g_metrics = self.train_step_g(img, label)

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
    def val_epoch(self, *args):
        z = torch.randn(100, self.z_dim, device=self.device)
        label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype=torch.long, device=self.device)
        label = F.one_hot(label, num_classes=10).float()

        img = self.g_net(z, label)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)
        img = make_grid(img, nrow=10)

        self.logger.add_image("val/sample", img, global_step=self.global_step)
        self.logger.flush()


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision.transforms import *

    train_data = MNIST("data", train=True, transform=ToTensor())

    ConditionalDCGAN(epochs=50, batch_size=128).fit(train_data=train_data)
