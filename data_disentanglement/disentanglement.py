"""solver.py"""

import os
# import visdom
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils.general_utils import DataGather, mkdirs, grid2gif, return_data
from utils.model_utils import recon_loss, kl_divergence, permute_dims, compute_latent_importance, calc_gradient_penalty, \
    prediction_loss
from data_disentanglement.nn_deg.fvae_model import FactorVAE2, Discriminator
from data_disentanglement.nn_deg.cvae_model import ConditionalFactorVAE2, ConditionalFactorVAE3
from data_disentanglement.nn_deg.aae_model import AAEGenerator, AAEDiscriminator, AAEEncoder


class Disentanglement(object):
    def __init__(self, config, deg_type, local_test_flag=False, global_model_data_path=''):

        self.global_model_data_path = global_model_data_path

        # Misc
        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = config.DEG.Learn.name
        self.max_iter = config.DEG.Learn.max_iter
        self.print_iter = config.DEG.Learn.print_iter
        self.global_iter = 0
        self.pbar = None

        # Data
        self.batch_size = config.DEG.Learn.batch_size
        self.data_loader = return_data(config.DEG.Learn, global_model_data_path, config.DRL.Learn.actions)
        self.action_size = config.DRL.Learn.actions

        # Dimension
        self.z_dim = config.DEG.Learn.z_dim
        self.image_length = config.DEG.Learn.image_length
        self.image_width = config.DEG.Learn.image_width

        self.nc = 3
        if deg_type == 'CVAE':
            self.gamma = config.DEG.CVAE.gamma
            if self.name == 'icehockey':
                self.CVAE = ConditionalFactorVAE3(env_name=self.name, state_size=config.DRL.Learn.state_size,
                                                  action_size=config.DRL.Learn.action_size,
                                                  reward_size=config.DRL.Learn.reward_size,
                                                  z_dim=self.z_dim).to(self.device)
            else:
                self.CVAE = ConditionalFactorVAE2(env_name=self.name, z_dim=self.z_dim).to(self.device)

            self.optim_CVAE = optim.Adam(self.CVAE.parameters(), lr=config.DEG.CVAE.lr_VAE,
                                        betas=(config.DEG.CVAE.beta1_VAE, config.DEG.CVAE.beta2_VAE))
            self.fvaeD = Discriminator(self.z_dim).to(self.device)
            self.optim_D = optim.Adam(self.fvaeD.parameters(), lr=config.DEG.CVAE.lr_D,
                                      betas=(config.DEG.CVAE.beta1_D, config.DEG.CVAE.beta2_D))
            self.nets = [self.CVAE, self.fvaeD]

            # Checkpoint
            self.ckpt_dir = os.path.join(self.global_model_data_path +config.DEG.CVAE.ckpt_dir, 'saved_model')
            self.ckpt_save_iter = config.DEG.CVAE.ckpt_save_iter
            if not local_test_flag:
                mkdirs(self.ckpt_dir)
            # if config.DEG.FVAE.ckpt_load:
            #     self.load_checkpoint(config.DEG.FVAE.ckpt_load)

            # Output(latent traverse GIF)
            self.output_dir = os.path.join(self.global_model_data_path+config.DEG.CVAE.output_dir, 'output')
            self.output_save = config.DEG.CVAE.output_save
            self.viz_ta_iter = config.DEG.CVAE.viz_ta_iter
            if not local_test_flag:
                mkdirs(self.output_dir)
        elif deg_type == 'FVAE':
            self.gamma = config.DEG.FVAE.gamma
            # self.lr_VAE = config.DEG.FVAE.lr_VAE
            # self.beta1_VAE = config.DEG.FVAE.beta1_VAE
            # self.beta2_VAE = config.DEG.FVAE.beta2_VAE
            # self.lr_D = config.DEG.FVAE.lr_D
            # self.beta1_D = config.DEG.FVAE.beta1_D
            # self.beta2_D = config.DEG.FVAE.beta2_D
            self.VAE = FactorVAE2(env_name=self.name, z_dim=self.z_dim).to(self.device)
            self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=config.DEG.FVAE.lr_VAE,
                                        betas=(config.DEG.FVAE.beta1_VAE, config.DEG.FVAE.beta2_VAE))
            self.fvaeD = Discriminator(self.z_dim).to(self.device)
            self.optim_D = optim.Adam(self.fvaeD.parameters(), lr=config.DEG.FVAE.lr_D,
                                      betas=(config.DEG.FVAE.beta1_D, config.DEG.FVAE.beta2_D))

            self.nets = [self.VAE, self.fvaeD]

            # Checkpoint
            self.ckpt_dir = os.path.join(self.global_model_data_path +config.DEG.FVAE.ckpt_dir, 'saved_model')
            self.ckpt_save_iter = config.DEG.FVAE.ckpt_save_iter
            if not local_test_flag:
                mkdirs(self.ckpt_dir)
            # if config.DEG.FVAE.ckpt_load:
            #     self.load_checkpoint(config.DEG.FVAE.ckpt_load)

            # Output(latent traverse GIF)
            self.output_dir = os.path.join(self.global_model_data_path+config.DEG.FVAE.output_dir, 'output')
            self.output_save = config.DEG.FVAE.output_save
            self.viz_ta_iter = config.DEG.FVAE.viz_ta_iter
            if not local_test_flag:
                mkdirs(self.output_dir)
        elif deg_type == 'VAE':
            self.gamma = config.DEG.VAE.gamma
            self.VAE = FactorVAE2(env_name=self.name, z_dim=self.z_dim).to(self.device)
            self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=config.DEG.VAE.lr_VAE,
                                        betas=(config.DEG.VAE.beta1_VAE, config.DEG.VAE.beta2_VAE))

            self.nets = [self.VAE]

            # Checkpoint
            self.ckpt_dir = os.path.join(self.global_model_data_path +config.DEG.VAE.ckpt_dir, 'saved_model')
            self.ckpt_save_iter = config.DEG.VAE.ckpt_save_iter
            if not local_test_flag:
                mkdirs(self.ckpt_dir)
            # if config.DEG.FVAE.ckpt_load:
            #     self.load_checkpoint(config.DEG.FVAE.ckpt_load)

            # Output(latent traverse GIF)
            self.output_dir = os.path.join(self.global_model_data_path+config.DEG.VAE.output_dir, 'output')
            self.output_save = config.DEG.VAE.output_save
            self.viz_ta_iter = config.DEG.VAE.viz_ta_iter
            if not local_test_flag:
                mkdirs(self.output_dir)
        elif deg_type == 'AAE':
            self.aeGnet= AAEGenerator(self.z_dim).to(self.device)
            self.aeDnet = AAEDiscriminator(self.z_dim).to(self.device)
            self.aeEnet = AAEEncoder(self.z_dim).to(self.device)
            self.nets = [self.aeGnet, self.aeDnet, self.aeEnet]
            self.optim_D = optim.Adam(self.aeDnet.parameters(), lr=config.DEG.AAE.lr_D,
                                      betas=(config.DEG.AAE.beta1_D, config.DEG.AAE.beta2_D))
            self.optim_G = optim.Adam(self.aeGnet.parameters(), lr=config.DEG.AAE.lr_G,
                                      betas=(config.DEG.AAE.beta1_G, config.DEG.AAE.beta2_G))
            self.optim_E = optim.Adam(self.aeEnet.parameters(), lr=config.DEG.AAE.lr_E,
                                      betas=(config.DEG.AAE.beta1_E, config.DEG.AAE.beta2_E))
            # self.optim_E_d = optim.Adam(self.aeEnet.parameters(), lr=config.DEG.AAE.lr_E/10,
            #                           betas=(config.DEG.AAE.beta1_E, config.DEG.AAE.beta2_E))
            if self.device == 'cuda':
                self.Tensor = torch.cuda.FloatTensor
            else:
                self.Tensor = torch.FloatTensor

            self.ckpt_dir = os.path.join(self.global_model_data_path +config.DEG.AAE.ckpt_dir, 'saved_model')
            self.ckpt_save_iter = config.DEG.AAE.ckpt_save_iter
            if not local_test_flag:
                mkdirs(self.ckpt_dir)

            self.output_dir = os.path.join(self.global_model_data_path+config.DEG.AAE.output_dir, 'output')
            self.output_save = config.DEG.AAE.output_save
            self.viz_ta_iter = config.DEG.AAE.viz_ta_iter
            if not local_test_flag:
                mkdirs(self.output_dir)
        else:
            raise ValueError('Unknown deg type {0}'.format(deg_type))




    def train_aae(self):
        netG, netD, netE = self.nets
        self.net_mode(train=True)
        ae_criterion = torch.nn.MSELoss().to(self.device)
        d_criterion = torch.nn.BCELoss().to(self.device)

        valid = torch.autograd.Variable(self.Tensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(self.Tensor(self.batch_size, 1).fill_(0.0), requires_grad=False)

        from tqdm import tqdm
        self.pbar = tqdm(total=self.max_iter)

        out = False
        while not out:
            for x_true1, cond1, x_true2, cond2 in self.data_loader:
                """ reconstruction loss"""
                real_data_v = torch.autograd.Variable(x_true1).to(self.device)
                real_data_resized_v = real_data_v.view(self.batch_size, -1)
                encode_z = netE(real_data_resized_v)
                decode_img = netG(encode_z)

                fool_loss = d_criterion(netD(encode_z), valid)
                ae_loss = ae_criterion(decode_img, real_data_v)

                g_loss = 0.001 * fool_loss + 0.999 * ae_loss

                netG.zero_grad()
                netE.zero_grad()
                g_loss.backward()
                self.optim_E.step()
                self.optim_G.step()

                """ Discriminator loss """
                # netE.eval()
                real_z = torch.autograd.Variable(self.Tensor(np.random.normal(0, 1, (self.batch_size, self.z_dim))))
                real_loss = d_criterion(netD(real_z), valid)
                fake_z = netE(real_data_resized_v)
                fake_loss = d_criterion(netD(fake_z), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                self.optim_D.zero_grad()
                d_loss.backward()
                self.optim_D.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(
                    "[global_iter %d/%d] [D loss: %f] [G loss: %f]"
                    % (self.global_iter , self.max_iter, d_loss.item(), g_loss.item())
                )
                if self.global_iter % self.ckpt_save_iter == 0:
                    print('Saving VAE models')
                    self.save_checkpoint('AAE-' + str(self.global_iter), type='AAE', verbose=True)

                if self.global_iter % self.viz_ta_iter == 0:
                    with torch.no_grad():
                        self.visualize_traverse(image_length=self.image_length,
                                                image_width=self.image_width,
                                                model_name='AAE',)
                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1



    def train_cvae(self):
        # from tqdm import tqdm
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        # self.pbar = tqdm(total=self.max_iter)
        out = False
        while not out:
            for x_true1, cond_y_1, x_true2, cond_y_2 in self.data_loader:
                x_true1 = torch.squeeze(x_true1, 0).to(self.device).float()
                cond1 = torch.squeeze(cond_y_1[:,:self.action_size+1], 0).to(self.device).float()
                y_true1 = torch.squeeze(cond_y_1[:, -1], 0).to(self.device).float()

                x_true2 = torch.squeeze(x_true2, 0).to(self.device).float()
                cond2 = torch.squeeze(cond_y_2[:,:self.action_size+1], 0).to(self.device).float()
                y_true2 = torch.squeeze(cond_y_2[:, -1], 0).to(self.device).float()

                # self.pbar.update(1)
                x_recon, mu_q, logvar_q, z_q, mu_p, logvar_p, z_p, y_predict = self.CVAE(x_true1, cond1)
                if self.name == 'icehockey':
                    vae_recon_loss = recon_loss(x_true1, x_recon, if_cross_entropy=False)
                else:
                    vae_recon_loss = recon_loss(x_true1, x_recon, if_cross_entropy=True)

                cvae_kld = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
                vae_kld = kl_divergence( mu_p, logvar_p)
                # vae_kld = kl_divergence(mu_q, logvar_q)

                p_loss = 100*prediction_loss(y_predict, y_true1)
                # p_loss = 0
                # print(x_recon)

                D_z = self.fvaeD(z_q)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                ones_z = torch.ones([self.batch_size, self.z_dim], dtype=torch.long, device=self.device)
                vae_var_loss = torch.abs(torch.squeeze(logvar_q.exp())-ones_z).mean()

                # vae_tc_loss = torch.zeros([1])
                # vae_var_loss = torch.zeros([1])

                vae_loss = vae_recon_loss + cvae_kld + vae_kld + p_loss + self.gamma * vae_tc_loss
                           # + (self.gamma/100)*vae_var_loss
                # vae_loss = vae_recon_loss

                self.optim_CVAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_CVAE.step()

                # D_tc_loss = torch.zeros([1])
                x_true2 = x_true2.to(self.device)
                z_prime = self.CVAE(x_true2, cond2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.fvaeD(z_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))
                #
                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                if self.global_iter % self.print_iter == 0:
                    print(
                        '[{}] vae_recon_loss:{:.3f}, cvae_kld:{:.3f}, vae_kld:{:.3f}, vae_tc_loss:{:.3f}, D_tc_loss:{:.3f}, '
                        'Predict_loss:{:.3f}'.format(self.global_iter, vae_recon_loss.item(), cvae_kld.item(), vae_kld.item(),
                                                     vae_tc_loss.item(), D_tc_loss.item(), p_loss))

                if self.global_iter % self.ckpt_save_iter == 0:
                    print('Saving VAE models')
                    self.save_checkpoint('CVAE-' + str(self.global_iter), type='CVAE', verbose=True)

                # if self.global_iter % self.viz_ta_iter == 0:
                #     with torch.no_grad():
                #         self.visualize_traverse(image_length=self.image_length,
                #                                 image_width=self.image_width,
                #                                 model_name='CVAE')

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1

        # self.pbar.write("[Training Finished]")
        # self.pbar.close()


    def train_fvae(self, apply_tc=True):
        # from tqdm import tqdm
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        # self.pbar = tqdm(total=self.max_iter)
        out = False
        while not out:
            for x_true1, cond1, x_true2, cond2 in self.data_loader:
                # self.pbar.update(1)
                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)
                if apply_tc:
                    D_z = self.fvaeD(z)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                    ones_z = torch.ones([self.batch_size, self.z_dim], dtype=torch.long, device=self.device)
                    vae_var_loss = torch.abs(torch.squeeze(logvar.exp())-ones_z).mean()

                    vae_loss = vae_recon_loss + vae_kld + self.gamma * vae_tc_loss + (self.gamma/100)*vae_var_loss

                    self.optim_VAE.zero_grad()
                    vae_loss.backward(retain_graph=True)
                    self.optim_VAE.step()

                    x_true2 = x_true2.to(self.device)
                    z_prime = self.VAE(x_true2, no_dec=True)
                    z_pperm = permute_dims(z_prime).detach()
                    D_z_pperm = self.fvaeD(z_pperm)
                    D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                    self.optim_D.zero_grad()
                    D_tc_loss.backward()
                    self.optim_D.step()

                    if self.global_iter % self.print_iter == 0:
                        print(
                            '[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                                self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(),
                                D_tc_loss.item()))

                    if self.global_iter % self.ckpt_save_iter == 0:
                        print('Saving VAE models')
                        self.save_checkpoint('FVAE-' + str(self.global_iter), type='FVAE', verbose=True)

                    if self.global_iter % self.viz_ta_iter == 0:
                        with torch.no_grad():
                            self.visualize_traverse(image_length=self.image_length,
                                                    image_width=self.image_width,
                                                    model_name='FVAE')
                else:
                    vae_loss = vae_recon_loss + vae_kld

                    self.optim_VAE.zero_grad()
                    vae_loss.backward(retain_graph=True)
                    self.optim_VAE.step()

                    if self.global_iter % self.print_iter == 0:
                        print(
                            '[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f}'.format(
                                self.global_iter, vae_recon_loss.item(), vae_kld.item()))

                    if self.global_iter % self.ckpt_save_iter == 0:
                        print('Saving VAE models')
                        self.save_checkpoint('VAE-' + str(self.global_iter), type='VAE', verbose=True)

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1

        # self.pbar.write("[Training Finished]")
        # self.pbar.close()


    def test(self, model_name, testing_output_dir):
        self.load_checkpoint(ckptname=model_name, testing_flag=True)
        with torch.no_grad():
            self.visualize_traverse(image_length=self.image_length,
                                    image_width=self.image_width,
                                    testing_output_dir=testing_output_dir,
                                    model_name=model_name)

    def visualize_traverse(self,
                           image_length, image_width, inter_number=10,
                           testing_output_dir=None, model_name=None):
        self.net_mode(train=False)
        if model_name.split('-')[0]=='CVAE':
            s_encoder = self.CVAE.state_encoder
            c_encoder = self.CVAE.condition_encoder
            q_encoder = self.CVAE.conditional_q_nn
            decoder = self.CVAE.decode
            # encode = torch.cat((self.state_encoder(x).squeeze(), self.condition_encoder(condition)), 1)
            # stats_q = self.conditional_q_nn(encode)
        elif model_name.split('-')[0]=='FVAE':
            decoder = self.VAE.decode
            encoder = self.VAE.encode
        elif model_name.split('-')[0]=='AAE':
            decoder = self.aeGnet
            encoder = self.aeEnet
        else:
            raise EnvironmentError("Unknown model name {0}.".format(model_name))

        z_checked_all = None
        total_checking_number = 1000
        for data in self.data_loader:
            input_images = data[0].to(self.device)
            input_conditions = data[1][:,:-1].to(self.device)
            # tmp = input_images.cpu().numpy()
            if model_name.split('-')[0]=='CVAE':
                encode_cat = torch.cat((s_encoder(input_images).squeeze(), c_encoder(input_conditions)), 1)
                z_output = q_encoder(encode_cat)[:, :self.z_dim].unsqueeze(-1).unsqueeze(-1)
            else:
                z_output = encoder(input_images)[:, :self.z_dim]
            if z_checked_all is None:
                z_checked_all = z_output
            else:
                z_checked_all = torch.cat([z_checked_all, z_output], dim=0)
            # tmp = z_checked_all.size()[0]
            if z_checked_all.size()[0] > total_checking_number:
                break
        avg_img_z = torch.mean(z_checked_all, 0, keepdim=True)

        dim_minmax_tuple_list = []
        dim_interpolation_list = []
        for k in range(self.z_dim):
            total_dim_data = z_checked_all[:, k]
            dim_max = float(torch.max(total_dim_data).cpu().numpy())
            dim_min = float(torch.min(total_dim_data).cpu().numpy())
            dim_minmax_tuple_list.append((dim_min, dim_max))
            inter = float(dim_max-dim_min)/inter_number
            interpolation = torch.arange(dim_min, dim_max, float(dim_max-dim_min)/inter_number)
            dim_interpolation_list.append(interpolation)

        random_img = self.data_loader.dataset.__getitem__(0)[2]
        random_conds = self.data_loader.dataset.__getitem__(0)[3][:-1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_conds = random_conds.to(self.device).unsqueeze(0)

        # if self.name == 'flappybird':
        fixed_idx = 111
        fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
        fixed_conds = self.data_loader.dataset.__getitem__(fixed_idx)[1][:-1]
        fixed_img = fixed_img.to(self.device).unsqueeze(0)
        fixed_conds = fixed_conds.to(self.device).unsqueeze(0)
        # random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

        if model_name.split('-')[0] == 'CVAE':
            random_encode_cat = torch.cat((s_encoder(random_img).squeeze(-1).squeeze(-1), c_encoder(random_conds)), 1)
            random_img_z = q_encoder(random_encode_cat)[:, :self.z_dim].unsqueeze(-1).unsqueeze(-1)
            fixed_encode_cat = torch.cat((s_encoder(fixed_img).squeeze(-1).squeeze(-1), c_encoder(fixed_conds)), 1)
            fixed_img_z = q_encoder(fixed_encode_cat)[:, :self.z_dim].unsqueeze(-1).unsqueeze(-1)
        else:
            random_img_z = encoder(random_img)[:, :self.z_dim]
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

        Z = {'fixed_img': fixed_img_z, 'random_img': random_img_z, 'avg_img':avg_img_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            # samples = []
            for k in range(self.z_dim):
                z = z_ori.clone()
                for val in dim_interpolation_list[k]:
                    z[:, k] = val
                    sample = F.sigmoid(decoder(z)).data
                    # samples.append(sample)
                    gifs.append(sample)
            # samples = torch.cat(samples, dim=0).cpu()
            # title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
        output_dir = None
        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter)+'/')
        if testing_output_dir is not None:
            output_dir = testing_output_dir
        if output_dir is not None:
            mkdirs(output_dir+'images/')
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, inter_number, self.nc, image_length, image_width).transpose(1, 2)

            masked_gif_tensor, masked_dim_number = compute_latent_importance(gif_tensor=gifs, sample_dimension=len(Z.keys()),
                                      inter_dimension=inter_number, latent_dimension=self.z_dim,
                                      image_width=image_width, image_length=image_length)

            for i, key in enumerate(Z.keys()):
                for j in range(inter_number):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir+'images/', '{0}_{1}_{2}_origin.jpg'.format(model_name, key, j)),
                               nrow=self.z_dim, pad_value=1)
                    if masked_gif_tensor is not None:
                        save_image(tensor=masked_gif_tensor[i][j],
                                   fp=os.path.join(output_dir+'images/', '{0}_{1}_{2}_masked.jpg'.format(model_name, key, j)),
                                   nrow=masked_dim_number, pad_value=1)
                grid2gif(str(os.path.join(output_dir+'images/', model_name+'_'+key + '*_origin.jpg')),
                         str(os.path.join(output_dir, model_name+'_'+key + '_origin.gif')), delay=10)
                if masked_gif_tensor is not None:
                    grid2gif(str(os.path.join(output_dir+'images/', model_name+'_'+key + '*_masked.jpg')),
                             str(os.path.join(output_dir, model_name+'_'+key + '_masked.gif')), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', type=None, verbose=True):

        if type == "FVAE":
            model_states = {'D': self.fvaeD.state_dict(),
                            'VAE': self.VAE.state_dict()}
            optim_states = {'optim_D': self.optim_D.state_dict(),
                            'optim_VAE': self.optim_VAE.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}
        elif type == "VAE":
            model_states = {'VAE': self.VAE.state_dict()}
            optim_states = {'optim_VAE': self.optim_VAE.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}

        elif type == "AAE":
            model_states = {'aeEnet':self.aeEnet.state_dict(),
                            'aeGnet':self.aeGnet.state_dict(),
                            'aeDnet':self.aeDnet.state_dict()}
            optim_states = {'optim_E':self.optim_E.state_dict(),
                            'optim_G':self.optim_G.state_dict(),
                            'optim_D':self.optim_D.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}
        elif type == "CVAE":
            model_states = {'D': self.fvaeD.state_dict(),
                            'CVAE': self.CVAE.state_dict()}
            optim_states = {'optim_D': self.optim_D.state_dict(),
                            'optim_CVAE': self.optim_CVAE.state_dict()}
            states = {'iter': self.global_iter,
                      'model_states': model_states,
                      'optim_states': optim_states}
        else:
            raise EnvironmentError("Saving type {0} is undefined".format(type))


        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            # self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))
            print("saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname, verbose=True, testing_flag=False, log_file=None):

        # if ckptname == 'last':
        #     ckpts = os.listdir(self.ckpt_dir)
        #     if not ckpts:
        #         if verbose:
        #             self.pbar.write("=> no checkpoint found")
        #         return
        #
        #     ckpts = [int(ckpt.split('-')[1]) for ckpt in ckpts]
        #     ckpts.sort(reverse=True)
        #     ckptname = 'FVAE-' + str(ckpts[0])
        from tqdm import tqdm
        if not testing_flag:
            self.pbar = tqdm(total=self.max_iter)
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                if torch.cuda.is_available():
                    checkpoint = torch.load(f)
                else:
                    checkpoint = torch.load(f, map_location=torch.device('cpu'))

            self.global_iter = checkpoint['iter']
            # if ckptname.split("-")[0] == "CVAE":
            if "CVAE" in ckptname:
                self.CVAE.load_state_dict(checkpoint['model_states']['CVAE'])
                self.optim_CVAE.load_state_dict(checkpoint['optim_states']['optim_CVAE'])
                self.fvaeD.load_state_dict(checkpoint['model_states']['D'])
                self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            # elif ckptname.split("-")[0] == "FVAE":
            elif "FVAE" in ckptname:
                self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
                self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
                self.fvaeD.load_state_dict(checkpoint['model_states']['D'])
                self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            else:
                self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
                self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            if not testing_flag:
                self.pbar.update(self.global_iter)
            if verbose:
                print("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter), file=log_file)
        else:
            if verbose:
                print("=> no checkpoint found at '{}'".format(filepath), file=log_file)
