from datetime import datetime
import ast
import os
import torch
import numpy as np
from copy import deepcopy

from mimic_learner.comparsion_learners.m5 import generate_weka_training_data, M5Tree
from mimic_learner.mcts_learner.mcts import test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf
from mimic_learner.comparsion_learners.cart import CARTRegressionTree

from utils.model_utils import visualize_split


class MimicLearner():
    def __init__(self, game_name, method, config, deg_model_name,
                 local_test_flag, global_model_data_path, log_file):
        self.mimic_env = MimicEnv(n_action_types=config.DEG.Learn.z_dim)
        self.z_dim = config.DEG.Learn.z_dim
        self.game_name = game_name
        self.action_number = config.DRL.Learn.actions
        self.action_type = config.DRL.Learn.action_type
        self.global_model_data_path = global_model_data_path

        self.num_simulations = config.Mimic.Learn.num_simulations
        self.episodic_sample_number = config.Mimic.Learn.episodic_sample_number
        self.data_save_dir = self.global_model_data_path + config.DEG.Learn.dset_dir
        self.image_type = config.DEG.Learn.image_type
        self.iteration_number = 0
        self.method = method
        if 'cart' in self.method:
            self.mimic_model = CARTRegressionTree()
        elif 'm5' in self.method:
            self.mimic_model = M5Tree(model_name=self.method)
        else:
            self.mimic_model = None

        if self.method == 'mcts' or self.method == 'cart-fvae':
            # initialize dientangler
            self.dientangler = Disentanglement(config, 'FVAE' , local_test_flag, self.global_model_data_path)
            self.dientangler.load_checkpoint(ckptname=deg_model_name, testing_flag=True, log_file=log_file)

        # if not local_test_flag:
        #     self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = None
        self.mcts_saved_dir = "" if local_test_flag else config.Mimic.Learn.mcts_saved_dir
        self.max_k = config.Mimic.Learn.max_k
        self.ignored_dim = ast.literal_eval(config.Mimic.Learn.ignore_dim)
        print("Ignored dim is {0}".format(config.Mimic.Learn.ignore_dim), file=log_file)

        self.shell_saved_model_dir = self.global_model_data_path + self.mcts_saved_dir



    def data_loader(self, episode_number, target):
        self.memory = []
        if target == "raw":
            self.image_type = 'color'

        def gather_data_values(action_value):
            action_value_items = action_value.split(',')
            action_index = int(action_value_items[0])
            action_values_list = []
            for i in range(self.action_number):
                action_values_list.append(float(action_value_items[i + 1]))
            reward = float(action_value_items[-1])
            if reward > 1:
                reward = 1
            return action_index, action_values_list, reward

        with open(self.data_save_dir + '/' + self.game_name + '/action_values.txt', 'r') as f:
            action_values = f.readlines()

        image = Image.open('{0}/{1}/{2}/images/{1}-{3}_{2}.png'.format(self.data_save_dir,
                                                                       self.game_name,
                                                                       self.image_type,
                                                                       self.iteration_number))

        action_index_t0, action_values_list_t0, reward_t0 = gather_data_values(action_values[self.iteration_number])
        if target == "latent":
            x_t0_resized = image
            with torch.no_grad():
                x_t0 = ttf.to_tensor(x_t0_resized).unsqueeze(0).to(self.dientangler.device)
                z0 = self.dientangler.VAE.encode(x_t0).squeeze()[:self.z_dim]
                z0 = z0.cpu().numpy()
        elif target == "raw":
            flatten_image_t0 = np.array(image).flatten()
        else:
            raise ValueError("Unknown data loader target {0}".format(target))

        while self.iteration_number < self.episodic_sample_number * episode_number:
            action_index_t1, action_values_list_t1, reward_t1 = gather_data_values(
                action_values[self.iteration_number + 1])
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0

            image = Image.open('{0}/{1}/{2}/images/{1}-{3}_{2}.png'.format(self.data_save_dir,
                                                                           self.game_name,
                                                                           self.image_type,
                                                                           self.iteration_number + 1))
            if target == "latent":
                x_t1_resized = image
                with torch.no_grad():
                    x_t1 = ttf.to_tensor(x_t1_resized).unsqueeze(0).to(self.dientangler.device)
                    z1 = self.dientangler.VAE.encode(x_t1).squeeze()[:self.z_dim]
                    z1 = z1.cpu().numpy()
                # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
                self.memory.append([z0, action_index_t0, reward_t0, z1, delta])
                z0 = z1
            elif target == "raw":
                flatten_image_t1 = np.array(image).flatten()
                self.memory.append([flatten_image_t0, action_index_t0, reward_t0, flatten_image_t1, delta])
                flatten_image_t0 = flatten_image_t1
            else:
                raise ValueError("Unknown data loader target {0}".format(target))

            self.iteration_number += 1
            action_index_t0 = action_index_t1
            action_values_list_t0 = action_values_list_t1
            reward_t0 = reward_t1



    def predict(self, data, action_id, saved_nodes_dir, visualize_flag=False):
        self.mimic_env.assign_data(data)
        init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
        moved_nodes = test_mcts(saved_nodes_dir=saved_nodes_dir, TreeEnv=self.mimic_env, action_id=action_id)
        level = 0
        parent_state = init_state
        parent_var_list = init_var_list
        state = init_state
        moved_nodes = moved_nodes
        for moved_node in moved_nodes:
            selected_action = moved_node.action
            if selected_action is not None:
                state, new_var_list = self.mimic_env.next_state(state=parent_state, action=selected_action,
                                                                parent_var_list = parent_var_list)
                if visualize_flag:
                    decoder = self.dientangler.VAE.decode
                    device = self.dientangler.device
                    z_dim = int(self.mimic_env.n_action_types)
                    data_all = self.mimic_env.data_all
                    visualize_split(selected_action, state, data_all, decoder, device, z_dim, level)
                parent_state = state
                parent_var_list = new_var_list
            level += 1

        predictions = [None for i in range(len(data))]
        assert len(state) == len(moved_nodes[-1].state_prediction)
        for subset_index in range(len(state)):
            subset = state[subset_index]
            for data_index in subset:
                predictions[data_index] = moved_nodes[-1].state_prediction[subset_index]

            # TODO: validate the prediction, please comment it out after evaluation
            subset_deltas = []
            for data_index in subset:
                subset_deltas.append(data[data_index][-1])
            subset_delta_avg = sum(subset_deltas) / len(subset_deltas)

            assert moved_nodes[-1].state_prediction[subset_index] == subset_delta_avg

        return predictions




    def test_mimic_model(self, action_id, log_file):
        self.iteration_number = int(self.episodic_sample_number * 4.5)

        if self.method == 'mcts':
            self.data_loader(episode_number=5, target="latent")  # divided into training, validation and testing
            saved_nodes_dir = "/Local-Scratch/oschulte/Galen/DRL-interpreter-model/MCTS/flappybird/" \
                        "saved_nodes_2020-04-15/".format(action_id)
            self.predict(self.memory, action_id, saved_nodes_dir)
        elif self.method == 'cart-fvae':
            self.data_loader(episode_number=5, target="latent")  # divided into training, validation and testing
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            testing_data = [[], []]
            for data_index in init_state[0]:
                data_input = self.memory[data_index][0]
                data_output = self.memory[data_index][4]
                testing_data[0].append(data_input)
                testing_data[1].append(data_output)
            testing_data[0] = np.stack(testing_data[0], axis=0)
            self.mimic_model.test_mimic(testing_data=testing_data, mimic_env=self.mimic_env)
        elif self.method == 'cart':
            self.data_loader(episode_number=5, target="raw")
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            testing_data = [[], []]
            for data_index in init_state[0]:
                data_input = self.memory[data_index][0]
                data_output = self.memory[data_index][4]
                testing_data[0].append(data_input)
                testing_data[1].append(data_output)
            testing_data[0] = np.stack(testing_data[0], axis=0)
            self.mimic_model.test_mimic(testing_data=testing_data, mimic_env=self.mimic_env)
        elif self.method == 'm5-rt':
            self.data_loader(episode_number=5, target="raw")
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            data_dir = self.data_save_dir + '/' + self.game_name + '/m5-weka/m5-tree-testing.csv'
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/M5/' \
                                                           '{0}/m5-rt-weka.model'.format(self.game_name)
            if not os.path.exists(data_dir):
                generate_weka_training_data(data=self.memory, action_id=action_id, dir=data_dir)
            self.mimic_model.test_weka_model(testing_data_dir=data_dir, save_model_dir=save_model_dir,
                                             log_file=log_file, mimic_env=self.mimic_env)

    def train_mimic_model(self, action_id, shell_round_number, log_file):
        mcts_file_name = None
        # for episode_number in range(1, 100):
        if self.action_type == 'discrete':
            # for action_id in range(self.action_number):
            # print('\nCurrent action is {0}'.format(action_id))

            if self.method == 'mcts':
                self.data_loader(episode_number=4, target="latent")
                self.mimic_env.assign_data(self.memory)
                init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
                execute_episode_single(num_simulations=self.num_simulations,
                                       TreeEnv=self.mimic_env,
                                       tree_writer=None,
                                       mcts_saved_dir=self.global_model_data_path + self.mcts_saved_dir,
                                       max_k=self.max_k,
                                       init_state=init_state,
                                       init_var_list=init_var_list,
                                       action_id=action_id,
                                       ignored_dim=self.ignored_dim,
                                       shell_round_number= shell_round_number,
                                       log_file = log_file,
                                       apply_split_parallel=True)
            elif self.method == 'cart-fvae':
                self.data_loader(episode_number=4, target="latent")
                self.mimic_env.assign_data(self.memory)
                init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
                training_data = [[],[]]
                for data_index in init_state[0]:
                    data_input = np.concatenate([self.memory[data_index][0]],axis=0)
                    data_output = self.memory[data_index][4]
                    training_data[0].append(data_input)
                    training_data[1].append(data_output)
                # cart.train_2d_tree()
                self.mimic_model.train_mimic(training_data=training_data, mimic_env = self.mimic_env)
            elif self.method == 'cart':
                self.data_loader(episode_number=4, target="raw")
                self.mimic_env.assign_data(self.memory)
                init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
                training_data = [[], []]
                for data_index in init_state[0]:
                    data_input = self.memory[data_index][0]
                    data_output = self.memory[data_index][4]
                    training_data[0].append(data_input)
                    training_data[1].append(data_output)
                training_data[0] = np.stack(training_data[0], axis=0)
                # cart.train_2d_tree()
                self.mimic_model.train_mimic(training_data=training_data, mimic_env = self.mimic_env)
                pass
            elif self.method == 'm5-rt':
                self.data_loader(episode_number=4, target="raw")
                self.mimic_env.assign_data(self.memory)
                init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
                data_dir = self.data_save_dir + '/' + self.game_name + '/m5-weka/m5-tree-training.csv'
                save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/M5/' \
                                                               '{0}/m5-rt-weka.model'.format(self.game_name)
                if not os.path.exists(data_dir):
                    generate_weka_training_data(data=self.memory, action_id=action_id, dir=data_dir)
                self.mimic_model.train_weka_model(training_data_dir=data_dir, save_model_dir=save_model_dir,
                                                    log_file=log_file, mimic_env=self.mimic_env)

            elif self.method == 'm5-mt':
                pass
            else:
                raise ValueError('Unknown method {0}'.format(self.method))


        # return mcts_file_name

