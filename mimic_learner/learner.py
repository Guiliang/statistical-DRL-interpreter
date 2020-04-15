from datetime import datetime
import ast
import torch
import numpy as np
from copy import deepcopy

from mimic_learner.mcts_learner.mcts import test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf
from mimic_learner.comparsion_learners.cart import RegressionTree

from utils.model_utils import visualize_split


class MimicLearner():
    def __init__(self, game_name, config,
                 local_test_flag, global_model_data_path, log_file):
        self.mimic_env = MimicEnv(n_action_types=config.DEG.Learn.z_dim * 2)
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

        # initialize dientangler
        self.dientangler = Disentanglement(config, 'FVAE' , local_test_flag, self.global_model_data_path)

        model_name = 'FVAE-1000000-bak-3-20'
        self.dientangler.load_checkpoint(ckptname=model_name, testing_flag=True, log_file=log_file)

        # if not local_test_flag:
        #     self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = []

        self.mcst_saved_dir = "" if local_test_flag else config.Mimic.Learn.saved_dir
        self.max_k = config.Mimic.Learn.max_k
        self.ignored_dim = ast.literal_eval(config.Mimic.Learn.ignore_dim)
        print("Ignored dim is {0}".format(config.Mimic.Learn.ignore_dim), file=log_file)

        self.shell_saved_model_dir = self.global_model_data_path + self.mcst_saved_dir



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
                z0 = self.dientangler.VAE.encode(x_t0).squeeze()[:self.z_dim]
                z0 = z0.cpu().numpy()

            image = Image.open('{0}/{1}/{2}/images/{1}-{3}_{2}.png'.format(self.data_save_dir,
                                                                           self.game_name,
                                                                           self.image_type,
                                                                           self.iteration_number + 1))
            x_t1_resized = ttf.resize(image, size=(64, 64))
            with torch.no_grad():
                x_t1 = ttf.to_tensor(x_t1_resized).unsqueeze(0).to(self.dientangler.device)
                z1 = self.dientangler.VAE.encode(x_t1).squeeze()[:self.z_dim]
                z1 = z1.cpu().numpy()

            self.iteration_number += 1
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0]

            # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
            self.memory.append([z0, action_index_t0, reward_t0, z1, delta])


    def predict(self, data, action_id, model_dir, visualize_flag=False):
        self.mimic_env.data_all = None
        self.mimic_env.add_data(data)
        init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
        moved_nodes = test_mcts(model_dir, TreeEnv=self.mimic_env, action_id=action_id)
        level = 0
        parent_state = init_state
        parent_var_list = init_var_list
        state = init_state
        moved_nodes = moved_nodes[:-1]  # TODO: -1 for current debug
        for moved_node in moved_nodes:
            selected_action = moved_node.action
            if selected_action is not None:
                state, new_var_list = self.mimic_env.next_state(state=parent_state, action=selected_action,
                                                                parent_var_list = parent_var_list)
                if visualize_flag:
                    decoder = self.dientangler.VAE.decode
                    device = self.dientangler.device
                    z_dim = int(self.mimic_env.n_action_types / 2)
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




    def test_mimic_model(self):
        self.data_loader(1)
        for action_id in range(self.action_number):
            model_dir = "/Local-Scratch/oschulte/Galen/DRL-interpreter-model/MCTS/flappybird/" \
                        "saved_model_action{0}_single_plays24000_2020-04-09-13.pkl".format(action_id)
            self. predict(self.memory, action_id, model_dir)



    def train_mimic_model(self, method, action_id, shell_round_number, log_file):
        mcts_file_name = None
        # for episode_number in range(1, 100):
        self.data_loader(1)
        self.mimic_env.add_data(self.memory)
        if self.action_type == 'discrete':
            # for action_id in range(self.action_number):
            # print('\nCurrent action is {0}'.format(action_id))
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            with open('../mimic_learner/tree_plots/{0}_tree_plot_{1}_action{2}.txt'
                              .format(method, datetime.today().strftime('%Y-%m-%d-%H'),
                                      action_id), 'w') as tree_writer:
                if method == 'mcts':
                    execute_episode_single(num_simulations=self.num_simulations,
                                           TreeEnv=self.mimic_env,
                                           tree_writer=tree_writer,
                                           mcts_saved_dir=self.global_model_data_path + self.mcst_saved_dir,
                                           max_k=self.max_k,
                                           init_state=init_state,
                                           init_var_list=init_var_list,
                                           action_id=action_id,
                                           ignored_dim=self.ignored_dim,
                                           shell_round_number= shell_round_number,
                                           log_file = log_file,
                                           apply_split_parallel=True
                                           )
                elif method == 'cart':
                    training_data = [[],[]]
                    for data_index in init_state[0]:
                        data_input = np.concatenate([self.memory[data_index][0], self.memory[data_index][3]],axis=0)
                        data_output = self.memory[data_index][4]
                        training_data[0].append(data_input)
                        training_data[1].append(data_output)
                    cart = RegressionTree(training_data=training_data, mimic_env = self.mimic_env)
                    cart.train_2d_tree()
                    # cart.train()
                else:
                    raise ValueError('Unknown method {0}'.format(method))


        # return mcts_file_name

