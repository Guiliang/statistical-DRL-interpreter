import os
import random
# from src.deep_q_network import DeepQNetwork
from collections import deque

import numpy as np
import torch
import torch.nn.functional as tnf
import torch.optim as optim
from tqdm import tqdm

from data_generator.fb_game.flappy_bird import FlappyBird
from data_generator.nn_drl.dqn_fb import FlappyBirdDQN
from utils.general_utils import mkdirs
from utils.memory_utils import PrioritizedReplay
from utils.model_utils import handle_image_input, store_state_action_data


class DRLDataGenerator():
    def __init__(self, game_name, config):
        mkdirs(config.DRL.Learn.data_save_path)
        self.game_name = game_name
        self.data_save_path = config.DRL.Learn.data_save_path
        self.config = config
        self.global_iter = 0
        use_cuda = config.DRL.Learn.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.ckpt_dir = self.config.DRL.Learn.ckpt_dir
        self.ckpt_save_iter = self.config.DRL.Learn.ckpt_save_iter
        self.pbar = tqdm(total=self.config.DRL.Learn.max_iter)
        mkdirs(self.ckpt_dir)

        self.apply_prioritize_memory = False
        if self.apply_prioritize_memory:
            self.memory = PrioritizedReplay(capacity=self.config.DRL.Learn.replay_memory_size)
        else:
            # store the previous observations in replay memory
            self.memory = deque()

        if game_name == 'flappybird':
            self.actions_number = self.config.DRL.Learn.actions
            # self.nn = DeepQNetwork().to(self.device)

        self.nn = FlappyBirdDQN().to(self.device)
        self.optim = optim.Adam(self.nn.parameters(), lr=self.config.DRL.Learn.learning_rate)
        if config.DRL.Learn.ckpt_load:
            self.load_checkpoint(model_name='flappy_bird_model')

        # torch.save(self.nn.state_dict(), "{}/flappy_bird_state".format(self.config.DRL.Learn.ckpt_dir))
        # self.trainTransform = tv.transforms.Compose([tv.transforms.Resize(size=(80, 80)),
        #                                              tv.transforms.Grayscale(num_output_channels=1),
        #                                              tv.transforms.ToTensor(),
        #                                              # tv.transforms.Normalize()
        #                                              ])

        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.game_state = FlappyBird()

    def append_sample(self, s_t, epsilon, time_step, discount_factor=1):

        with torch.no_grad():
            readout_t0 = self.nn(s_t.unsqueeze(0))
        readout_t0 = readout_t0.cpu().numpy()
        a_t = np.zeros([self.config.DRL.Learn.actions])
        if time_step % self.config.DRL.Learn.frame_per_action != 0 and time_step < self.config.DRL.Learn.explore:
            action_index = 0
        else:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(self.config.DRL.Learn.actions)
            else:
                # values, indices = torch.max(readout_t, 1)
                action_index = np.argmax(readout_t0)
            # print('action is {0}'.format(str(a_t)))
        a_t[action_index] = 1
        # scale down epsilon
        if epsilon > self.config.DRL.Learn.final_epsilon and time_step > self.config.DRL.Learn.observe:
            epsilon -= (
                               self.config.DRL.Learn.initial_epsilon - self.config.DRL.Learn.final_epsilon) / self.config.DRL.Learn.explore
        x_t1_colored, r_t, terminal = self.game_state.next_frame(action_index)
        x_t1 = handle_image_input(x_t1_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
        # save_image_path=self.image_save_path, iter=time_step).to(self.device)
        # s_t1 = torch.stack(tensors=[x_t1], dim=0).to(self.device)
        s_t1 = torch.cat((s_t[1:, :, :], x_t1.to(self.device)))
        with torch.no_grad():
            readout_t1 = self.nn(s_t1.unsqueeze(0)).cpu().numpy()

        if self.apply_prioritize_memory:
            old_val = readout_t0[0][action_index]
            if terminal:
                readout_t0_update = r_t
            else:
                readout_t0_update = r_t + discount_factor * max(readout_t1[0])

            error = abs(old_val - readout_t0_update)
            self.memory.add(error, (s_t, torch.tensor(a_t, dtype=torch.float32),
                                    torch.tensor([r_t], dtype=torch.float32), s_t1, terminal))
        else:
            # store the transition in D
            self.memory.append((s_t, torch.tensor(a_t, dtype=torch.float32),
                                torch.tensor([r_t], dtype=torch.float32), s_t1, terminal))
            if len(self.memory) > self.config.DRL.Learn.replay_memory_size:
                self.memory.popleft()

        return readout_t0, s_t1, action_index, r_t, epsilon

    def save_checkpoint(self, time_step):
        model_states = {'FlappyBirdDQN': self.nn.state_dict()}
        optim_states = {'optim_DQN': self.optim.state_dict()}
        states = {'iter': time_step,
                  'model_states': model_states,
                  'optim_states': optim_states}

        filepath = os.path.join(self.ckpt_dir, "flappy_bird_model")
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_checkpoint(self, model_name):
        filepath = os.path.join(self.ckpt_dir, model_name)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.global_iter = checkpoint['iter']
            self.nn.load_state_dict(checkpoint['model_states']['FlappyBirdDQN'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim_DQN'])

    def sample_batch(self):
        if self.apply_prioritize_memory:
            minibatch, idxs, is_weights = self.memory.sample(self.config.DRL.Learn.batch)
        else:
            minibatch = random.sample(self.memory, self.config.DRL.Learn.batch)
            is_weights = np.ones(shape=[self.config.DRL.Learn.batch])
            idxs = None

        return minibatch, idxs, is_weights

    def test_model_and_generate_data(self, test_size=10000):

        with open(self.data_save_path+'action_values.txt', 'w')as action_values_file:
            action_index = 0
            x_t0_colored, r_t, terminal = self.game_state.next_frame(action_index)
            x_t0 = handle_image_input(x_t0_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
            s_t0 = torch.cat(tuple(x_t0 for _ in range(4))).to(self.device)
            self.global_iter += 1
            x_t1_colored = x_t0_colored
            while self.global_iter < test_size:
                with torch.no_grad():
                    readout = self.nn(s_t0.unsqueeze(0))
                readout = readout.cpu().numpy()
                store_state_action_data(img_colored=x_t1_colored[:self.game_state.screen_width, :int(self.game_state.base_y)],
                                        action_values=readout[0], reward=r_t, action_index=action_index,
                                        save_image_path=self.data_save_path, action_values_file=action_values_file,
                                        game_name=self.game_name, iteration_number=self.global_iter)
                action_index = np.argmax(readout)
                x_t1_colored, r_t, terminal = self.game_state.next_frame(action_index)
                x_t1 = handle_image_input(x_t1_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
                s_t1 = torch.cat((s_t0[1:, :, :], x_t1.to(self.device)))
                s_t0 = s_t1
                self.global_iter += 1

    def train_DRl_model(self):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        x_t0_colored, r_0, terminal = self.game_state.next_frame(0)
        x_t = handle_image_input(x_t0_colored[:self.game_state.screen_width, :int(self.game_state.base_y)])
        # s_t = torch.stack(tensors=[x_t], dim=0)
        s_t = torch.cat(tuple(x_t for _ in range(4))).to(self.device)

        # start training
        epsilon = self.config.DRL.Learn.initial_epsilon
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            readout_t0, s_t1, action_index, r_t, epsilon = self.append_sample(s_t, epsilon, self.global_iter)

            # only train if done observing
            if self.global_iter > self.config.DRL.Learn.observe:
                # sample a minibatch to train on
                minibatch, idxs, is_weights = self.sample_batch()

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
                        y_batch.append(r_batch[i] + self.config.DRL.Learn.gamma * max_readout_t1_batch)
                readout_action = torch.sum(torch.mul(readout_t0_batch, a_batch), dim=1)

                y_batch = torch.stack(y_batch).squeeze()
                errors = torch.abs(readout_action - y_batch).data.cpu().numpy()
                # update priority
                if self.apply_prioritize_memory:
                    for i in range(self.config.DRL.Learn.batch):
                        idx = idxs[i]
                        self.memory.update(idx, errors[i])

                DRL_loss = (torch.FloatTensor(is_weights).to(self.device) * tnf.mse_loss(readout_action,
                                                                                         y_batch)).mean()

                # DRL_loss = square_loss(x=readout_action, y=y_batch)
                self.optim.zero_grad()
                DRL_loss.backward(retain_graph=True)
                self.optim.step()

            # update the old values
            s_t = s_t1
            self.global_iter += 1

            # save progress every 10000 iterations
            # if self.global_iter % self.ckpt_save_iter == 0:
            #     print('Saving VAE models')
            #     self.save_checkpoint('DRL-' + str(self.global_iter), verbose=True)

            # print info
            state = ""
            if self.global_iter <= self.config.DRL.Learn.observe:
                state = "observe"
            elif self.config.DRL.Learn.observe < self.global_iter <= self.config.DRL.Learn.observe + self.config.DRL.Learn.explore:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", self.global_iter, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,
                  "/ REWARD",
                  r_t, "/ Q_MAX %e" % np.max(readout_t0))
