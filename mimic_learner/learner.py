from datetime import datetime
import ast
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from mimic_learner.mcts_learner.mcts import execute_episode_parallel, test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf
from mimic_learner.comparsion_learners.cart import RegressionTree

from utils.memory_utils import PrioritizedReplay
from utils.model_utils import build_decode_input, compute_diff_masked_images


class MimicLearner():
    def __init__(self, game_name, config,
                 local_test_flag, global_model_data_path):
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
        self.dientangler = Disentanglement(config, 'FVAE' , local_test_flag, self.global_model_data_path)

        model_name = 'FVAE-1000000-bak-3-20'
        self.dientangler.load_checkpoint(ckptname=model_name, testing_flag=True)

        # if not local_test_flag:
        #     self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = []

        self.mcst_saved_dir = "" if local_test_flag else config.Mimic.Learn.saved_dir
        self.max_k = config.Mimic.Learn.max_k
        self.ignored_dim = ast.literal_eval(config.Mimic.Learn.ignore_dim)
        print("Ignored dim is {0}".format(config.Mimic.Learn.ignore_dim))

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
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0]

            # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
            self.memory.append([z0, action_index_t0, reward_t0, z1, delta])

    def test_mimic_model(self):
        self.data_loader(1)
        self.mimic_env.add_data(self.memory)
        for action_id in range(self.action_number):
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            model_dir = "/Local-Scratch/oschulte/Galen/DRL-interpreter-model/MCTS/flappybird/" \
                        "saved_model_action{0}_single_plays24000_2020-03-30-11.pkl".format(action_id)
            final_splitted_states, moved_nodes = test_mcts(model_dir, TreeEnv=self.mimic_env, action_id=action_id)

            level = 0

            for moved_node in moved_nodes:
                selected_action = moved_node.action

                if selected_action is not None:

                    selected_state_index = int(selected_action.split('_')[0])
                    selected_dim = int(selected_action.split('_')[1])
                    selected_split_value = float(selected_action.split('_')[2])

                    splitted_states = moved_node.state[selected_state_index:selected_state_index+2]
                    splitted_states_avg = []
                    state_features_all = []
                    for state_index in range(len(splitted_states)):
                        state_features = []
                        state = splitted_states[state_index]
                        for data_index in state:
                            z_index = np.concatenate([self.mimic_env.data_all[data_index][0],
                                            self.mimic_env.data_all[data_index][3]], axis=0)
                            state_features.append(z_index)
                            state_features_all.append(z_index)
                        state_features_avg = np.average(np.asarray(state_features), axis=0)
                        splitted_states_avg.append(state_features_avg)
                        print(splitted_states_avg)
                    state_features_all_avg = np.average(np.asarray(state_features_all), axis=0)
                    x_recon_all = None
                    for state_index in range(len(splitted_states_avg)):
                        state_features_avg = state_features_all_avg
                        state_features_avg[selected_dim] = splitted_states_avg[state_index][selected_dim]
                        z_1_state = build_decode_input(state_features_avg[:int(self.mimic_env.n_action_types / 2)])
                        z_2_state = build_decode_input(state_features_avg[int(self.mimic_env.n_action_types / 2):])
                        z_state = torch.cat([z_1_state, z_2_state], axis=0)
                        with torch.no_grad():
                            x_recon= F.sigmoid(
                                self.dientangler.VAE.decode(z_state.to(self.dientangler.device))).data

                        if x_recon_all is None:
                            x_recon_all = x_recon
                        else:
                            x_recon_all = torch.stack([x_recon_all, x_recon], axis=0)

                    x_recon_all = torch.cat([x_recon_all[:, 0, :, :, :], x_recon_all[:, 1, :, :, :]], axis=-2)
                    masked_images = compute_diff_masked_images(x_recon_all)
                    # masked_images = x_recon_all
                    masked_images = torch.stack(torch.split(masked_images, int(masked_images.shape[-2]/2), dim=-2), axis=1)
                    save_image(tensor=masked_images[0], fp="../mimic_learner/action_images_plots/level_{0}_action_{1}_image_left.jpg".
                               format(level, selected_action), nrow=2, pad_value=1)
                    save_image(tensor=masked_images[1], fp="../mimic_learner/action_images_plots/level_{0}_action_{1}_image_right.jpg".
                               format(level, selected_action), nrow=2, pad_value=1)

                level += 1





    def train_mimic_model(self, method):

        # for episode_number in range(1, 100):
        self.data_loader(1)
        self.mimic_env.add_data(self.memory)
        if self.action_type == 'discrete':
            for action_id in range(self.action_number):
                print('\nCurrent action is {0}'.format(action_id))
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

                break

