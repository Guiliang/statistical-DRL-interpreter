#!/usr/bin/env python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init

from utils.model_utils import get_same_padding, calculate_conv_output_dimension


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

        in_channels = [1, 4, 32]
        out_channels = [4, 32, 32]
        kernel_sizes = [8, 4, 4]
        strides = [2, 2, 1]
        input_image_size = config.DRL.Learn.input_image_size

        module = nn.Sequential()
        for cov_index in range(len(in_channels)):
            padding = get_same_padding(input_image_size, kernel_sizes[cov_index],
                                       strides[cov_index], 1)
            module.add_module(name='Conv2d-layer{0}'.format(str(cov_index)),
                              module=nn.Conv2d(in_channels=in_channels[cov_index], out_channels=out_channels[cov_index],
                                               kernel_size=kernel_sizes[cov_index], stride=strides[cov_index],
                                               padding=padding,
                                               # padding=0,
                                               dilation=1))
            module.add_module(name='Relu-layer{0}'.format(str(cov_index)), module=nn.ReLU(True))
            if cov_index < len(in_channels) - 1:
                module.add_module(name='MaxPool-layer{0}'.format(str(cov_index)),
                                  module=nn.MaxPool2d(kernel_size=kernel_sizes[cov_index],
                                                      stride=strides[cov_index], padding=1))
                input_image_size = calculate_conv_output_dimension(size=input_image_size,
                                                                   kernel_size=kernel_sizes[cov_index],
                                                                   stride=strides[cov_index], dilation=1, padding=1)
        input_image_size = input_image_size - 1

        module.add_module(name='Flatten', module=Flatten())
        module.add_module(name='Linear-1', module=nn.Linear(32 * input_image_size * input_image_size, 512))
        module.add_module(name='Relu-layer', module=nn.LeakyReLU(0.2, True))
        module.add_module(name='Linear-2', module=nn.Linear(512, config.DRL.Learn.actions))
        print(module)
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
        return self.module(x)


if __name__ == "__main__":
    from config.flappy_bird_config import FlappyBirdConfig

    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    icehockey_cvrnn_config = FlappyBirdConfig.load(flappybird_config_path)
    FlappyBirdDRLNN(config=icehockey_cvrnn_config)
