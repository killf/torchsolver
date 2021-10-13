import torch
from torch import nn

from torchsolver.models import register_model
from torchsolver.solver import GANSolver
from torchsolver.config import GANConfig


@register_model()
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


@register_model()
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


class MnistGANSolver(GANSolver):
    def forward_d(self, img, label):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)

        # compute loss of real_img
        real_out = self.d_net(img)
        loss_real = self.loss(real_out, real_label)
        real_score = real_out

        # compute loss of fake_img
        z = torch.randn(N, self.cfg.z_dimension, device=self.device)
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
        z = torch.randn(N, self.cfg.z_dimension, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        g_loss = self.loss(fake_out, real_label)
        g_score = fake_out

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score.mean())}

    @torch.no_grad()
    def val_epoch(self):
        z = torch.randn(32, self.cfg.z_dimension, device=self.device)
        img = self.g_net(z)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)

        self.logger.add_images("val/sample", img, global_step=self.global_step)
        self.logger.flush()


if __name__ == '__main__':
    from torchvision.transforms import *

    cfg = GANConfig(g_net_name="Generator", d_net_name="Discriminator", optimizer_name="Adam", loss_name="BCELoss")

    cfg.train_data_name = "MNIST"
    cfg.train_data_args.root = "data"
    cfg.train_data_args.download = True
    cfg.train_data_args.transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    cfg.z_dimension = 100
    cfg.g_net_args.input_size = cfg.z_dimension

    cfg.batch_size = 128
    cfg.epochs = 50

    MnistGANSolver(cfg).train()
