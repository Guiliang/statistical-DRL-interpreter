"""solver.py"""

import os
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils.general_utils import DataGather, mkdirs, grid2gif, return_data
from utils.model_utils import recon_loss, kl_divergence, permute_dims
from data_disentanglement.nn_deg.fvae_model import FactorVAE1, FactorVAE2, Discriminator


class Disentanglement(object):
    def __init__(self, config):
        # Misc
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = config.DEG.FVAE.name
        self.max_iter = config.DEG.FVAE.max_iter
        self.print_iter = config.DEG.FVAE.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.batch_size = config.DEG.FVAE.batch_size
        self.data_loader = return_data(config)

        # Networks & Optimizers
        self.z_dim = config.DEG.FVAE.z_dim
        self.gamma = config.DEG.FVAE.gamma

        self.lr_VAE = config.DEG.FVAE.lr_VAE
        self.beta1_VAE = config.DEG.FVAE.beta1_VAE
        self.beta2_VAE = config.DEG.FVAE.beta2_VAE

        self.lr_D = config.DEG.FVAE.lr_D
        self.beta1_D = config.DEG.FVAE.beta1_D
        self.beta2_D = config.DEG.FVAE.beta2_D

        self.VAE = FactorVAE2(self.z_dim).to(self.device)
        self.nc = 3
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]

        # Checkpoint
        self.ckpt_dir = os.path.join(config.DEG.FVAE.ckpt_dir, config.DEG.FVAE.name)
        self.ckpt_save_iter = config.DEG.FVAE.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if config.DEG.FVAE.ckpt_load:
            self.load_checkpoint(config.DEG.FVAE.ckpt_load)

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(config.DEG.FVAE.output_dir, config.name)
        self.output_save = config.DEG.FVAE.output_save
        mkdirs(self.output_dir)


    def train(self):
        self.net_mode(train=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                vae_kld = kl_divergence(mu, logvar)

                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

                vae_loss = vae_recon_loss + vae_kld + self.gamma * vae_tc_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()

                if self.global_iter % self.print_iter == 0:
                    self.pbar.write(
                        '[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'.format(
                            self.global_iter, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(),
                            D_tc_loss.item()))


                if self.viz_on and (self.global_iter % self.viz_ta_iter == 0):
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2 / 3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break
                self.global_iter += 1

        self.pbar.write("[Training Finished]")
        self.pbar.close()


    def visualize_traverse(self, limit=3, inter=2 / 3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit + 0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040  # square
            fixed_idx2 = 332800  # ellipse
            fixed_idx3 = 578560  # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square': fixed_img_z1, 'fixed_ellipse': fixed_img_z2,
                 'fixed_heart': fixed_img_z3, 'random_img': random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281  # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307  # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535  # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1': fixed_img_z1, 'fixed_2': fixed_img_z2,
                 'fixed_3': fixed_img_z3, 'fixed_4': fixed_img_z4,
                 'random': random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919  # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330  # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1': fixed_img_z1, 'fixed_2': fixed_img_z2,
                 'fixed_3': fixed_img_z3, 'random': random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img': fixed_img_z, 'random_img': random_img_z, 'random_z': random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples, env=self.name + '/traverse',
                            opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key + '*.jpg')),
                         str(os.path.join(output_dir, key + '.gif')), delay=10)

        self.net_mode(train=True)

    def viz_init(self):
        zero_init = torch.zeros([1])
        self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name + '/lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='D(.)',
                          legend=['D(z)', 'D(z_perm)']))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['recon'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='reconstruction loss', ))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['acc'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='discriminator accuracy', ))
        self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name + '/lines',
                      win=self.win_id['kld'],
                      opts=dict(
                          xlabel='iteration',
                          ylabel='kl divergence', ))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D': self.D.state_dict(),
                        'VAE': self.VAE.state_dict()}
        optim_states = {'optim_D': self.optim_D.state_dict(),
                        'optim_VAE': self.optim_VAE.state_dict()}
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt.split('-')[1]) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = 'FVAE-'+str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
