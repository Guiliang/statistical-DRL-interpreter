"""model.py"""
import torch
import torch.nn as nn
import torch.nn.init as init

from utils.model_utils import get_same_padding

class ConditionalFactorVAE3(nn.Module):

    def __init__(self, env_name, state_size, action_size, reward_size, z_dim=10):
        super(ConditionalFactorVAE3, self).__init__()
        self.z_dim = z_dim
        if env_name == 'icehockey':
            input_dims = [state_size, 256, 512, 512, 256, 256]
            output_dims = [256, 512, 512, 256, 256, 2*z_dim]
        else:
            raise ValueError("Unknown env name {0}".format(env_name))
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dims[0], output_dims[0], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(input_dims[1], output_dims[1], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(input_dims[2], output_dims[2], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(input_dims[3], output_dims[3], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(input_dims[4], output_dims[4], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(input_dims[5], output_dims[5], True),
        )
        condition_size = action_size+reward_size
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_size, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 20, True),
        )
        self.condition_prior = nn.Sequential(
            nn.Linear(condition_size, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 20, True),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(z_dim, input_dims[-1], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_dims[-2], input_dims[-2], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_dims[-3], input_dims[-3], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_dims[-4], input_dims[-4], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_dims[-5], input_dims[-5], True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(output_dims[-6], input_dims[-6], True),
        )

        self.conditional_q_nn = nn.Sequential(nn.Linear(20 + output_dims[-1], 64),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(64, 256),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(256, 64),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(64, 20))

        self.weight_init()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, condition, no_dec=False):
        stats_p = self.condition_prior(condition)
        mu_p = stats_p[:, :self.z_dim]
        logvar_p = stats_p[:, self.z_dim:]
        z_p = self.reparametrize(mu_p, logvar_p)

        encode = torch.cat((self.state_encoder(x).squeeze(), self.condition_encoder(condition)), 1)
        stats_q = self.conditional_q_nn(encode)
        mu_q = stats_q[:, :self.z_dim]
        logvar_q = stats_q[:, self.z_dim:]
        z_q = self.reparametrize(mu_q, logvar_q)

        if no_dec:
            return z_q.squeeze()
        else:
            x_recon = self.state_decoder(z_q)
            # self.decoder_output_layer(x_recon, output_size=[80, 80])
            return x_recon, mu_q, logvar_q, z_q, mu_p, logvar_p, z_p



class ConditionalFactorVAE2(nn.Module):

    def __init__(self, env_name, z_dim=10):
        super(ConditionalFactorVAE2, self).__init__()
        self.z_dim = z_dim

        encoder_depth = 6

        if env_name == 'flappybird':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            # en_kernel_size = [4, 4, 4, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
            condition_size = 3
        elif env_name == 'Enduro-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            # en_kernel_size = [4, 4, 4, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
            condition_size = 10
        elif env_name == 'Enduro-v1':
            en_in_channels = [3, 32, 32, 64, 64, 256, 256]
            en_out_channels = [32, 32, 64, 64, 256, 256, z_dim]
            en_kernel_size = [8, 8, 6, 6, 4, 4, 1]
            # en_kernel_size = [4, 4, 4, 4, 4, 1]
            # de_stride = [2, 2, 2, 2, 2, 1, 1]
            stride = [2, 2, 2, 2, 2, 1, 1]
            # de_padding = [1, 1, 1, 1, 1, 0, 0]
            padding = [1, 1, 1, 1, 1, 0, 0]
            condition_size = 10
        elif env_name == 'Assault-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
            condition_size = 8
        elif env_name == 'Breakout-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
            condition_size = 4
        elif env_name == 'SpaceInvaders-v0':
            en_in_channels = [3, 32, 32, 64, 64, 256]
            en_out_channels = [32, 32, 64, 64, 256, z_dim]
            en_kernel_size = [8, 8, 6, 4, 4, 1]
            stride = [2, 2, 2, 2, 1, 1]
            padding = [1, 1, 1, 1, 0, 0]
            condition_size = 7
        else:
            raise ValueError("unknown env name".format(env_name))


        # # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # # padding_width = get_same_padding(input_image_width, kernel_sizes[cov_index], strides[cov_index], 1)
        self.state_encoder = nn.Sequential(
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
            nn.Conv2d(en_in_channels[5], en_out_channels[5]*20, en_kernel_size[5], stride[5], padding[5])
        )
        #
        # # ConvTranspose2d (in_channels, out_channels, kernel_size, stride, padding)
        self.decode = nn.Sequential(
            nn.Conv2d(en_out_channels[-1], en_in_channels[-1], en_kernel_size[-1], stride[-1], padding[-1]),
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
        # # dim_size = ((input.size(d + 2) - 1) * stride[d] - 2 * padding[d] + kernel_size[d]
        # self.state_encoder = nn.Sequential()
        # for i in range(len(en_in_channels)):
        #     self.state_encoder.add_module('En_Conv{0}'.format(i), nn.Conv2d(en_in_channels[i], en_out_channels[i], en_kernel_size[i], stride[i], padding[i]))
        #     if i<len(en_in_channels)-1:
        #         self.state_encoder.add_module('En_Relu{0}'.format(i), nn.ReLU(True))
        #
        # self.decode = nn.Sequential()
        # for i in range(len(en_in_channels)):
        #     if i == 0:
        #         self.decode.add_module('De_Conv{0}'.format(i), nn.Conv2d(int(en_out_channels[-i-1]/2), en_in_channels[-i-1], en_kernel_size[-i-1], stride[-i-1], padding[-i-1]))
        #     else:
        #         self.decode.add_module('De_ConvTranspose2d{0}'.format(i),  nn.ConvTranspose2d(int(en_out_channels[-i-1]), en_in_channels[-i-1], en_kernel_size[-i-1], stride[-i-1], padding[-i-1]))
        #     if i<len(en_in_channels)-1:
        #         self.state_encoder.add_module('De_Relu{0}'.format(i), nn.ReLU(True))

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_size, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 20, True),
        )

        self.condition_prior = nn.Sequential(
            nn.Linear(condition_size, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 64, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 20, True),
        )

        self.conditional_q_nn = nn.Sequential(nn.Linear(en_out_channels[-1]*20 + en_out_channels[-1]*2, 256),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(256, 128),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(128, 64),
                                              nn.LeakyReLU(0.2, True),
                                              nn.Linear(64, en_out_channels[-1]*2))

        self.y_predict_nn = nn.Sequential(nn.Linear(en_out_channels[-1], 32),
                                              nn.LeakyReLU(0.2, True),
                                              # nn.Linear(32, 64),
                                              # nn.LeakyReLU(0.2, True),
                                              # nn.Linear(64, 32),
                                              # nn.LeakyReLU(0.2, True),
                                              nn.Linear(32, 1))
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

    def forward(self, x, condition, no_dec=False):
        stats_p = self.condition_prior(condition)
        mu_p = stats_p[:, :self.z_dim]
        logvar_p = stats_p[:, self.z_dim:]
        z_p = self.reparametrize(mu_p, logvar_p)

        # tmp1 = self.state_encoder(x).squeeze()
        # tmp2 = self.condition_encoder(condition)

        encode = torch.cat((self.state_encoder(x).squeeze(), self.condition_encoder(condition)), 1)
        stats_q = self.conditional_q_nn(encode)
        mu_q = stats_q[:, :self.z_dim]
        logvar_q = stats_q[:, self.z_dim:]
        z_q = self.reparametrize(mu_q, logvar_q)

        y_predict = self.y_predict_nn(mu_q)

        if no_dec:
            return z_q.squeeze()
        else:
            x_recon = self.decode(z_q.unsqueeze(-1).unsqueeze(-1))
            # self.decoder_output_layer(x_recon, output_size=[80, 80])
            return x_recon, mu_q, logvar_q, z_q, mu_p, logvar_p, z_p, y_predict


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
    cond = torch.ones([32, 4]).float()
    test = ConditionalFactorVAE2(env_name='flappybird')
    x_recon, mu_q, logvar_q, z_q, mu_p, logvar_p, z_p, y_predict = test.forward(x, cond[:, :3])
    print('testing')
