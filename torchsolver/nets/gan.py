import torch
from torch import nn, optim
import torchsolver as ts


class Discriminator(nn.Module):
    def __init__(self, in_c=784, channels=(200,), leaky=0.02, layer_norm=nn.LayerNorm):
        super(Discriminator, self).__init__()

        layers = []
        for out_c in channels:
            layers.append(nn.Linear(in_c, out_c))
            layers.append(nn.LeakyReLU(leaky))
            if layer_norm is not None:
                layers.append(layer_norm(out_c))
            in_c = out_c
        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(in_c, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, z_dim=100, out_c=784, channels=(200,), leaky=0.02, layer_norm=nn.LayerNorm):
        super(Generator, self).__init__()

        layers = []
        for in_c in channels:
            layers.append(nn.Linear(z_dim, in_c))
            layers.append(nn.LeakyReLU(leaky))
            if layer_norm is not None:
                layers.append(layer_norm(in_c))
            z_dim = in_c
        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(z_dim, out_c)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return torch.tanh(x)


class GANNet(ts.GANModule):
    def __init__(self, z_dim=100, out_c=784, d_channels=(200,), g_channels=(200,), **kwargs):
        super(GANNet, self).__init__(**kwargs)

        self.z_dim = z_dim

        self.g_net = Generator(z_dim=z_dim, out_c=out_c, channels=g_channels)
        self.d_net = Discriminator(in_c=out_c, channels=d_channels)

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
    def sample(self, n, img_size=(1, 28, 28)):
        z = torch.randn(n, self.z_dim, device=self.device)
        img = self.g_net(z)

        img = (img + 1) / 2.
        img = torch.clamp(img, 0, 1)

        return img.view(img.size(0), *img_size)

    def val_epoch(self, *args):
        img = self.sample(32)

        self.logger.add_images("val/sample", img, global_step=self.global_step)
        self.logger.flush()
