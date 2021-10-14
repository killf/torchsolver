import torch
from torch import nn, optim
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class GANModule(ts.GANModule):
    def __init__(self, z_dim=100, **kwargs):
        super(GANModule, self).__init__(**kwargs)

        self.z_dim = z_dim

        self.g_net = Generator(z_dim)
        self.d_net = Discriminator()

        self.loss = nn.BCELoss()

        self.g_optimizer = optim.Adam(self.g_net.parameters())
        self.d_optimizer = optim.Adam(self.d_net.parameters())

    def forward_d(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)

        # compute loss of real_img
        real_out = self.d_net(img.flatten(1))
        loss_real = self.loss(real_out, real_label)
        real_score = real_out

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
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
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        g_loss = self.loss(fake_out, real_label)
        g_score = fake_out

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score.mean())}

    @torch.no_grad()
    def val_epoch(self, *args):
        z = torch.randn(32, self.z_dim, device=self.device)
        img = self.g_net(z)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)
        img = img.view(img.size(0), 1, 28, 28)

        self.logger.add_images("val/sample", img, global_step=self.global_step)
        self.logger.flush()


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision.transforms import *

    train_data = MNIST("data", train=True, transform=ToTensor())

    GANModule(batch_size=128).fit(train_data=train_data)
