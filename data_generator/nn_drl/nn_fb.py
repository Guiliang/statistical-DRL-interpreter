#!/usr/bin/env python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init


def compute_same_padding(input_size, kernel_size, stride, dilation):
    padding = 0.5 * ((input_size - 1) * stride - input_size + kernel_size + (kernel_size - 1)(dilation - 1))
    print('padding is {0}'.format(str(padding)))
    return padding


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


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class FlappyBirdDRLNN(nn.Module):

    def __init__(self, config):
        super(FlappyBirdDRLNN, self).__init__()

        in_channels = [4, 32, 64]
        out_channels = [32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        input_image_size = 80

        module = nn.Sequential()
        for cov_index in range(len(in_channels)):
            module.add_module(name='Conv2d-layer{0}'.format(str(cov_index)),
                              module=nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0],
                                               kernel_size=kernel_sizes[0], stride=strides[0],
                                               padding=compute_same_padding(input_image_size, kernel_sizes[0],
                                                                            strides[0], 1),
                                               dilation=1))
            module.add_module(name='Relu-layer{0}'.format(str(cov_index)), module=nn.ReLU(True))
            if cov_index < len(in_channels) - 1:
                module.add_module(name='MaxPool-layer{0}'.format(str(cov_index)),
                                  module=nn.MaxPool2d(kernel_size=2, stride=2))
        module.add_module(name='Flatten', module=Flatten())
        module.add_module(name='Linear', module=nn.Linear(1024, 512))
        module.add_module(name='Relu-Linear', module=nn.LeakyReLU(0.2, True))
        module.add_module(name='Linear', module=nn.Linear(512, config.Learn.actions))
        self.module = module

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x):
        return self.module(x).squeeze()


if __name__ == "__main__":
    from config.flappy_bird_config import FlappyBirdCongfig
    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    icehockey_cvrnn_config = FlappyBirdCongfig.load(flappybird_config_path)
    FlappyBirdDRLNN(config=icehockey_cvrnn_config)
