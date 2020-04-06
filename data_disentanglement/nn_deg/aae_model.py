# source code mostly copied from https://github.com/neale/Adversarial-Autoencoder
import numpy as np
from torch import nn
from torch.nn import functional as F
from utils.model_utils import SpectralNorm as SN


class AAEGenerator(nn.Module):
    def __init__(self, dim):
        super(AAEGenerator, self).__init__()
        self.shape = (64, 64, 3)
        self.dim = dim
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 2* 4 * 4 * 4 * self.dim),
                nn.BatchNorm1d(2 * 4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 2, stride=2),
                nn.BatchNorm2d(4 * self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                )
        block3 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * 2 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        output = output.view(-1, 3, 64, 64)
        return output


class AAEDiscriminator(nn.Module):

    def __init__(self, dim):
        super(AAEDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class AAEEncoder(nn.Module):
    def __init__(self, dim):
        super(AAEEncoder, self).__init__()
        self.shape = (64, 64, 3)
        self.dim = dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(4 * self.dim, 8 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(8 * self.dim, 16 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 3, 64, 64)
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output.view(-1, self.dim)