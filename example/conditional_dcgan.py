import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self, in_c=1, c_dim=10):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02)
        )

        self.fc_c = nn.Sequential(
            nn.Linear(c_dim, 1000),
            nn.LeakyReLU(0.02)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 28 * 28 + 1000, 1024),
            nn.LeakyReLU(0.02)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)
        c = self.fc_c(c)
        x = torch.cat([x, c], 1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim=100, c_dim=10):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.fc_c = nn.Sequential(
            nn.Linear(c_dim, 1000),
            nn.LeakyReLU(0.02)
        )

        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)

        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.ConvTranspose2d(64, 32, 5, 1, 2)
        )

        self.deconv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.ConvTranspose2d(32, 1, 5, 1, 2)
        )

    def forward(self, z, c):
        c = self.fc_c(c)
        x = torch.cat([z, c], 1)

        x = self.fc(x)
        x = x.view(x.size(0), 64, 28, 28)

        x = self.deconv1(x)
        x = self.deconv2(x)

        x = F.sigmoid(x)
        return x


class ConditionalDCGAN(ts.GANModule):
    def __init__(self, z_dim=100, c_dim=10, in_c=1, **kwargs):
        super(ConditionalDCGAN, self).__init__(**kwargs)

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.g_net = Generator(z_dim=z_dim, c_dim=c_dim)
        self.d_net = Discriminator(in_c=in_c, c_dim=c_dim)

        self.loss = nn.BCELoss()

        self.g_optimizer = optim.SGD(self.g_net.parameters(), lr=0.01)
        self.d_optimizer = optim.SGD(self.d_net.parameters(), lr=0.01)

    def forward_d(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)
        label = F.one_hot(label, num_classes=self.c_dim).float()

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
        label = F.one_hot(label, num_classes=self.c_dim).float()

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z, label)
        fake_out = self.d_net(fake_img, label)
        g_loss = self.loss(fake_out, real_label)
        g_score = fake_out

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score.mean())}

    @torch.no_grad()
    def val_epoch(self, *args):
        z = torch.randn(100, self.z_dim, device=self.device)
        label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype=torch.long, device=self.device)
        label = F.one_hot(label, num_classes=self.c_dim).float()

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
