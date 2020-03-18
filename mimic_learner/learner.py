from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from mimic_learner.mcts_learner.mcts import execute_episode_parallel, test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf

from utils.memory_utils import PrioritizedReplay
from utils.model_utils import build_decode_input


class MimicLearner():
    def __init__(self, game_name, config, local_test_flag, global_model_data_path):
        self.mimic_env = MimicEnv(n_action_types=config.DEG.FVAE.z_dim * 2)
        self.game_name = game_name
        self.action_number = config.DRL.Learn.actions
        self.action_type = config.DRL.Learn.action_type
        self.global_model_data_path = global_model_data_path

        self.num_simulations = config.Mimic.Learn.num_simulations
        self.episodic_sample_number = config.Mimic.Learn.episodic_sample_number
        self.data_save_dir = self.global_model_data_path + config.DEG.FVAE.dset_dir
        self.image_type = config.DEG.FVAE.image_type
        self.iteration_number = 0

        # initialize dientangler
        self.dientangler = Disentanglement(config, local_test_flag, self.global_model_data_path)

        if not local_test_flag:
            self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = []

        self.mcst_saved_dir = "" if local_test_flag else config.Mimic.Learn.saved_dir
        self.max_k = config.Mimic.Learn.max_k

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
            action_index_t1, action_values_list_t1, reward_t1 = gather_data_values(
                action_values[self.iteration_number + 1])

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

    def test_mimic_model(self):
        self.data_loader(1)
        self.mimic_env.add_data(self.memory)
        for action_id in range(self.action_number):
            model_dir = "/Users/liu/PycharmProjects/statistical-DRL-interpreter/mimic_learner/save_tmp/mcts_save_action0_single_plays500.0_2020-03-17-14.pkl"
            final_splitted_states = test_mcts(model_dir, TreeEnv=self.mimic_env, action_id=action_id)
            final_splitted_states_avg = []
            for state in final_splitted_states:
                state_features = []
                for data_index in state:
                    state_features.append(np.concatenate([self.mimic_env.data_all[data_index][0],
                                                          self.mimic_env.data_all[data_index][3]], axis=0))
                state_features = np.asarray(state_features)
                state_features_avg = np.average(state_features, axis=0)
                z_1 = build_decode_input(state_features_avg[:int(self.mimic_env.n_action_types / 2)])
                with torch.no_grad():
                    x_recon_1= self.dientangler.VAE.decode(z_1)
                plt.figure()
                plt.imshow(x_recon_1[0].permute(1, 2, 0))
                plt.show()
                # x_recon_2 = self.dientangler.VAE.decode(state_features_avg[int(self.mimic_env.n_action_types / 2): ])

                final_splitted_states_avg.append(state_features_avg)

    def train_mimic_model(self):

        # for episode_number in range(1, 100):
        self.data_loader(1)
        self.mimic_env.add_data(self.memory)
        if self.action_type == 'discrete':
            for action_id in range(self.action_number):
                print('\nCurrent action is {0}'.format(action_id))
                init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
                with open('../mimic_learner/tree_plots/tree_plot_{0}_action{1}.txt'
                                  .format(datetime.today().strftime('%Y-%m-%d-%H'), action_id), 'w') as tree_writer:
                    execute_episode_single(num_simulations=self.num_simulations,
                                           TreeEnv=self.mimic_env,
                                           tree_writer=tree_writer,
                                           mcts_saved_dir=self.global_model_data_path + self.mcst_saved_dir,
                                           max_k=self.max_k,
                                           init_state=init_state,
                                           init_var_list=init_var_list,
                                           action_id=action_id
                                           )
