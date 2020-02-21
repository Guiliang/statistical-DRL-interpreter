import random
import sys
import os
import data_generator.fb_game.wrapped_flappy_bird as fb_game
import data_generator.nn_drl.nn_fb as nn_fb
from collections import deque
import numpy as np
from tqdm import tqdm
import torchvision as tv
import torchvision.transforms.functional as tf
import torch
import torch.optim as optim
from utils.general_utils import mkdirs
from utils.model_utils import square_loss, handle_image_input
from PIL import Image
import matplotlib.pyplot as plt


class DRLDataGenerator():
    def __init__(self, game_name, config):

        self.global_iter = 0
        self.config = config
        use_cuda = config.DRL.Learn.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.ckpt_dir = self.config.DRL.Learn.ckpt_dir
        self.ckpt_save_iter = self.config.DRL.Learn.ckpt_save_iter
        self.pbar = tqdm(total=self.config.DRL.Learn.max_iter)
        mkdirs(self.ckpt_dir)

        if config.DRL.Learn.ckpt_load:
            self.load_checkpoint()
        if game_name == 'flappybird':
            self.actions_number = self.config.DRL.Learn.actions
            self.nn = nn_fb.FlappyBirdDRLNN(config=self.config).to(self.device)
            self.optim = optim.Adam(self.nn.parameters(), lr=self.config.DRL.Learn.learning_rate,
                                    betas=(self.config.DRL.Learn.beta1_D, self.config.DRL.Learn.beta2_D))

        # self.trainTransform = tv.transforms.Compose([tv.transforms.Resize(size=(80, 80)),
        #                                              tv.transforms.Grayscale(num_output_channels=1),
        #                                              tv.transforms.ToTensor(),
        #                                              # tv.transforms.Normalize()
        #                                              ])

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'DRLNN': self.nn.state_dict()}
        optim_states = {'optim_DRLNN': self.optim.state_dict()}
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
            ckptname = 'DRL-' + str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.nn.load_state_dict(checkpoint['model_states']['DRLNN'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim_DRLNN'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    def train_DRl_model(self):

        # open up a game state to communicate with emulator
        game_state = fb_game.GameState()

        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(self.actions_number)
        do_nothing[0] = 1
        x_t0_colored, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = handle_image_input(img_colored=x_t0_colored)
        s_t = torch.stack(tensors=[x_t], dim=0).to(self.device)

        # start training
        epsilon = self.config.DRL.Learn.initial_epsilon
        t = 0
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            # readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            # self.nn = self.nn.eval()
            with torch.no_grad():
                readout_t = self.nn(s_t.unsqueeze(0))
            readout_t = readout_t.cpu().numpy()
            a_t = np.zeros([self.config.DRL.Learn.actions])
            if t % self.config.DRL.Learn.frame_per_action == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(self.config.DRL.Learn.actions)
                    a_t[action_index] = 1
                else:
                    # values, indices = torch.max(readout_t, 1)
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
                # print('action is {0}'.format(str(a_t)))
            else:
                a_t[0] = 1  # do nothing

            # scale down epsilon
            if epsilon > self.config.DRL.Learn.final_epsilon and t > self.config.DRL.Learn.observe:
                epsilon -= (self.config.DRL.Learn.initial_epsilon - self.config.DRL.Learn.final_epsilon) / self.config.DRL.Learn.explore
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = handle_image_input(img_colored=x_t1_colored)
            s_t1 = torch.stack(tensors=[x_t1], dim=0).to(self.device)

            # store the transition in D
            D.append((s_t, torch.tensor(a_t,  dtype=torch.float32),
                      torch.tensor([r_t], dtype=torch.float32), s_t1, terminal))
            if len(D) > self.config.DRL.Learn.replay_memory:
                D.popleft()

            # only train if done observing
            if t > self.config.DRL.Learn.observe:
                # sample a minibatch to train on
                minibatch = random.sample(D, self.config.DRL.Learn.batch)

                # get the batch variables
                s_t_batch = torch.stack([d[0] for d in minibatch]).to(self.device)
                a_batch = torch.stack([d[1] for d in minibatch]).to(self.device)
                r_batch = torch.stack([d[2] for d in minibatch]).to(self.device)
                s_t1_batch = torch.stack([d[3] for d in minibatch]).to(self.device)

                y_batch = []
                # readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
                readout_t1_batch = self.nn(s_t1_batch)
                readout_t0_batch = self.nn(s_t_batch)
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        max_readout_t1_batch = torch.max(readout_t1_batch[i], dim=0)[0]
                        y_batch.append(r_batch[i] + max_readout_t1_batch)

                readout_action = torch.sum(torch.mul(readout_t0_batch, a_batch), dim=1)
                DRL_loss = square_loss(x=readout_action, y=torch.stack(y_batch))
                self.optim.zero_grad()
                DRL_loss.backward(retain_graph=True)
                self.optim.step()

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            # if self.global_iter % self.ckpt_save_iter == 0:
            #     print('Saving VAE models')
            #     self.save_checkpoint('DRL-' + str(self.global_iter), verbose=True)

            # print info
            state = ""
            if t <= self.config.DRL.Learn.observe:
                state = "observe"
            elif self.config.DRL.Learn.observe < t <= self.config.DRL.Learn.observe + self.config.DRL.Learn.explore:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
                  "/ Q_MAX %e" % np.max(readout_t))
