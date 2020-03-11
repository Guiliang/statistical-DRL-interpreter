from datetime import datetime

import torch

from mimic_learner.mcts_learner.mcts import execute_episode
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf

from utils.memory_utils import PrioritizedReplay


class MimicLearner():
    def __init__(self, game_name, config, local_test_flag):
        self.mimic_env = MimicEnv(n_action_types=config.DEG.FVAE.z_dim*2)
        self.game_name = game_name
        self.action_number = config.DRL.Learn.actions

        self.num_simulations = config.Mimic.Learn.num_simulations
        self.episodic_sample_number = config.Mimic.Learn.episodic_sample_number
        self.data_save_dir = config.DEG.FVAE.dset_dir
        self.image_type = config.DEG.FVAE.image_type
        self.iteration_number = 0

        # initialize dientangler
        self.dientangler = Disentanglement(config, local_test_flag)

        if not local_test_flag:
            self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = []

    def data_loader(self, episode_number):

        def gather_data_values(action_value):
            action_value_items = action_value.split(',')
            action_index = int(action_value_items[0])
            action_values_list = []
            for i in range(self.action_number):
                action_values_list.append(float(action_value_items[i + 1]))
            reward = float(action_value_items[-1])
            return action_index, action_values_list, reward

        with open(self.data_save_dir + '/' + self.game_name + '/action_values.txt', 'r') as f:
            action_values = f.readlines()

        while self.iteration_number < self.episodic_sample_number * episode_number:
            action_index_t0, action_values_list_t0, reward_t0 = gather_data_values(action_values[self.iteration_number])
            action_index_t1, action_values_list_t1, reward_t1 = gather_data_values(action_values[self.iteration_number])

            image = Image.open('{0}/{1}/{2}/images/{1}-{3}_{2}.png'.format(self.data_save_dir,
                                                                           self.game_name,
                                                                           self.image_type,
                                                                           self.iteration_number))
            x_t0_resized = ttf.resize(image, size=(64, 64))
            with torch.no_grad():
                x_t0 = ttf.to_tensor(x_t0_resized).unsqueeze(0).to(self.dientangler.device)
                _, _, _, z0 = self.dientangler.VAE(x_t0)
                z0 = z0.cpu().numpy()

            image = Image.open('{0}/{1}/{2}/images/{1}-{3}_{2}.png'.format(self.data_save_dir,
                                                                           self.game_name,
                                                                           self.image_type,
                                                                           self.iteration_number + 1))
            x_t1_resized = ttf.resize(image, size=(64, 64))
            with torch.no_grad():
                x_t1 = ttf.to_tensor(x_t1_resized).unsqueeze(0).to(self.dientangler.device)
                _, _, _, z1 = self.dientangler.VAE(x_t1)
                z1 = z1.cpu().numpy()

            self.iteration_number += 1
            delta = abs(reward_t0 + max(action_values_list_t1) - action_values_list_t0[action_index_t0])

            # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
            self.memory.append([z0, action_index_t0, reward_t0, z1, delta])

    def train_mimic_model(self):

        with open('../mimic_learner/tree_plots/tree_plot_{0}.txt'.format(datetime.today().strftime('%Y-%m-%d-%H:%M')), 'w') as tree_writer:
            for episode_number in range(1, 100):
                self.data_loader(episode_number)
                self.mimic_env.add_data(self.memory)

                execute_episode(num_simulations=self.num_simulations,
                                TreeEnv=self.mimic_env,
                                data=self.memory,
                                tree_writer=tree_writer)
