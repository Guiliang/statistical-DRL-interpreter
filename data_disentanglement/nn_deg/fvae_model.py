"""model.py"""
import torch
import torch.nn as nn
import torch.nn.init as init

from utils.model_utils import get_same_padding


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""

    def __init__(self, z_dim=10):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2 * z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE2(nn.Module):

    def __init__(self, env_name, z_dim=10):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim

        encoder_depth = 6

        if env_name == 'flappybird':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, 2 * z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            # en_kernel_size = [4, 4, 4, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
        elif env_name == 'Assault-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, 2 * z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
        elif env_name == 'Breakout-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, 2 * z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
        elif env_name == 'SpaceInvaders-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, 2 * z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]


        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # padding_width = get_same_padding(input_image_width, kernel_sizes[cov_index], strides[cov_index], 1)
        self.encode = nn.Sequential(
            nn.Conv2d(en_in_channels[0], en_out_channels[0], en_kernel_size[0], stride[0], padding[0]),
            nn.ReLU(True),
            nn.Conv2d(en_in_channels[1], en_out_channels[1], en_kernel_size[1], stride[1], padding[1]),
            nn.ReLU(True),
            nn.Conv2d(en_in_channels[2], en_out_channels[2], en_kernel_size[2], stride[2], padding[2]),
            nn.ReLU(True),
            nn.Conv2d(en_in_channels[3], en_out_channels[3], en_kernel_size[3], stride[3], padding[3]),
            nn.ReLU(True),
            nn.Conv2d(en_in_channels[4], en_out_channels[4], en_kernel_size[4], stride[4], padding[4]),
            nn.ReLU(True),
            nn.Conv2d(en_in_channels[5], en_out_channels[5], en_kernel_size[5], stride[5], padding[5])
        )

        # ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding)
        self.decode = nn.Sequential(
            nn.Conv2d(int(en_out_channels[-1]/2), en_in_channels[-1], en_kernel_size[-1], stride[-1], padding[-1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(en_out_channels[-2], en_in_channels[-2], en_kernel_size[-2], stride[-2], padding[-2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(en_out_channels[-3], en_in_channels[-3], en_kernel_size[-3], stride[-3], padding[-3]),
            nn.ReLU(True),
            nn.ConvTranspose2d(en_out_channels[-4], en_in_channels[-4], en_kernel_size[-4], stride[-4], padding[-4]),
            nn.ReLU(True),
            nn.ConvTranspose2d(en_out_channels[-5], en_in_channels[-5], en_kernel_size[-5], stride[-5], padding[-5]),
            nn.ReLU(True),
            nn.ConvTranspose2d(en_out_channels[-6], en_in_channels[-6], en_kernel_size[-6], stride[-6], padding[-6]),
        )
        # dim_size = ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] + kernel_size[d])

        self.weight_init()


    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            # self.decoder_output_layer(x_recon, output_size=[80, 80])
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE3(nn.Module):
    """Encoder and Decoder architecture for 3D Faces data."""

    def __init__(self, z_dim=10):
        super(FactorVAE3, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2 * z_dim, 1)
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        else:
            raise ValueError('unknown initializer mode {0}'.format(mode))

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

if __name__ == "__main__":
    x = torch.ones([32, 3, 84, 84]).float()
    test = FactorVAE2(env_name='Assault-v0')
    stats = test.encode(x)
    mu = stats[:, :10]
    y = test.decode(mu)
    print('testing')
