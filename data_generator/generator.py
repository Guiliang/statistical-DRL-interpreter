import random
import sys
import os
import data_generator.fb_game.wrapped_flappy_bird as fb_game
import data_generator.nn_drl.nn_fb as nn_fb
from collections import deque
import numpy as np
from tqdm import tqdm
import torchvision as tv
import torch
import torch.optim as optim
from config.flappy_bird_config import FlappyBirdCongfig
from utils.general_utils import mkdirs
from utils.model_utils import square_loss


class DRLDataGenerator():
    def __init__(self, game_name, config):

        self.global_iter = 0
        self.config = config
        use_cuda = config.Learn.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.ckpt_dir = os.path.join(self.config.Learn.ckpt_dir, self.config.Learn.name)
        self.ckpt_save_iter = self.config.Learn.ckpt_save_iter
        self.pbar = tqdm(total=self.config.Learn.max_iter)
        mkdirs(self.ckpt_dir)

        if config.Learn.ckpt_load:
            self.load_checkpoint(self.config.Learn.ckpt_load)
        if game_name == 'flappy-bird':
            self.actions_number = self.config.Learn.actions
            self.nn = nn_fb.FlappyBirdDRLNN(config=self.config).to(self.device)
            self.optim = optim.Adam(self.nn.parameters(), lr=self.config.Learn.learning_rate,
                                    betas=(self.config.Learn.beta1_D, self.config.Learn.beta2_D)).to(self.device)

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

        trainTransform = tv.transforms.Compose([tv.transforms.Resize(size=(80, 80)),
                                                tv.transforms.Grayscale(num_output_channels=1),
                                                tv.transforms.ToTensor(),
                                                # tv.transforms.Normalize()
                                                ])

        x_t = trainTransform(x_t0_colored)
        # Apply threshold
        data = x_t > 128  # mean value
        data = data.float()

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).to(self.device)

        # start training
        epsilon = self.config.Learn.initial_epsilon
        t = 0
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            # readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            readout_t = self.nn(s_t)
            a_t = np.zeros([self.config.Learn.actions])
            action_index = 0
            if t % self.config.Learn.frame_per_action == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(self.config.Learn.actions)
                    a_t[random.randrange(self.config.Learn.actions)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1  # do nothing

            # scale down epsilon
            if epsilon > self.config.Learn.final_epsilon and t > self.config.Learn.observe:
                epsilon -= (self.config.Learn.initial_epsilon - self.config.Learn.final_epsilon) / self.config.Learn.explore

            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = trainTransform(x_t1_colored)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, 1:], axis=2).to(self.device)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > self.config.Learn.replay_memory:
                D.popleft()

            # only train if done observing
            if t > self.config.Learn.observe:
                # sample a minibatch to train on
                minibatch = random.sample(D, self.config.Learn.batch)

                # get the batch variables
                s_t_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_t1_batch = [d[3] for d in minibatch]

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
                        y_batch.append(r_batch[i] + self.config.Learn.gamma * np.max(readout_t1_batch[i]))

                readout_action = np.sum(np.matmul(a=readout_t0_batch, b=a_batch), axis=1)
                DRL_loss = square_loss(x=readout_action, y=y_batch)
                self.optim.zero_grad()
                DRL_loss.backward(retain_graph=True)
                self.optim.step()

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if self.global_iter % self.ckpt_save_iter == 0:
                print('Saving VAE models')
                self.save_checkpoint('DRL-' + str(self.global_iter), verbose=True)

            # print info
            state = ""
            if t <= self.config.Learn.observe:
                state = "observe"
            elif self.config.Learn.observe < t <= self.config.Learn.observe + self.config.Learn.explore:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
                  "/ Q_MAX %e" % np.max(readout_t))


def run():
    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    icehockey_cvrnn_config = FlappyBirdCongfig.load(flappybird_config_path)
