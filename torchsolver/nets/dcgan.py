import torch
from torch import nn, optim
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self, image_size=(1, 28, 28), channels=(32, 64), leaky=0.02):
        super(Discriminator, self).__init__()

        in_c, img_h, img_w = image_size

        layers = []
        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, 5, padding=2))
            layers.append(nn.LeakyReLU(leaky))
            layers.append(nn.AvgPool2d(2, stride=2))
            in_c, img_h, img_w = out_c, img_h // 2, img_w // 2
        self.layers = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_c * img_h * img_w, 1024),
            nn.LeakyReLU(leaky),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return self.final(x)


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=(50, 25), image_size=(1, 28, 28), leaky=0.002):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        in_c, init_h, init_w = channels[0], image_size[1], image_size[2]
        self.fc = nn.Sequential(
            nn.Linear(z_dim, init_h * init_w * in_c),
            ts.nn.View(in_c, init_h, init_w),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(leaky)
        )

        layers = []
        for i, out_c in enumerate(channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(leaky))
            in_c = out_c
        self.layers = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_c, image_size[0], 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.layers(x)
        x = self.final(x)
        return x


class DCGANNet(ts.GANModule):
    def __init__(self, z_dim=100, image_size=(1, 28, 28), d_channels=(32, 64), g_channels=(50, 25), g_net=None, d_net=None, **kwargs):
        super(DCGANNet, self).__init__(**kwargs)

        self.z_dim = z_dim

        self.g_net = g_net or Generator(z_dim=z_dim, image_size=image_size, channels=g_channels)
        self.d_net = d_net or Discriminator(image_size=image_size, channels=d_channels)

        self.loss = nn.BCELoss()

        self.g_optimizer = optim.Adam(self.g_net.parameters())
        self.d_optimizer = optim.Adam(self.d_net.parameters())

    def forward_d(self, img, *args):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)
        fake_label = torch.zeros(N, 1, device=self.device)

        # compute loss of real_img
        real_out = self.d_net(img)
        loss_real = self.loss(real_out, real_label)
        real_score = (real_out > 0.5).float()

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        loss_fake = self.loss(fake_out, fake_label)
        fake_score = (fake_out < 0.5).float()

        d_loss = loss_real + loss_fake
        d_score = torch.cat([real_score, fake_score], dim=0).mean()
        return d_loss, {"d_loss": float(d_loss), "d_score": float(d_score)}

    def forward_g(self, img, *args):
        N = img.size(0)
        real_label = torch.ones(N, 1, device=self.device)

        # compute loss of fake_img
        z = torch.randn(N, self.z_dim, device=self.device)
        fake_img = self.g_net(z)
        fake_out = self.d_net(fake_img)
        g_loss = self.loss(fake_out, real_label)
        g_score = (fake_out > 0.5).float().mean()

        return g_loss, {"g_loss": float(g_loss), "g_score": float(g_score)}

    @torch.no_grad()
    def val_epoch(self, epoch, *args):
        z = torch.randn(32, self.z_dim, device=self.device)
        img = self.g_net(z)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)

        self.logger.add_images("val/sample", img, global_step=epoch)
        self.logger.flush()
