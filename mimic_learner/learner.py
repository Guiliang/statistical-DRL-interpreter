import csv
from datetime import datetime
import ast
import os

import logging
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from mimic_learner.mcts_learner import mcts
# print (mcts.c_PUCT)
from mimic_learner.mcts_learner.mcts import test_mcts, execute_episode_single
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement
from PIL import Image
import torchvision.transforms.functional as ttf
from mimic_learner.comparsion_learners.cart import CARTRegressionTree
from utils.general_utils import gather_data_values

from utils.model_utils import visualize_split, build_decode_input

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


class MimicLearner():
    def __init__(self, game_name, method, config, deg_model_name,
                 local_test_flag, global_model_data_path, log_file, options=[]):
        self.mimic_env = MimicEnv(n_action_types=config.DEG.Learn.z_dim)
        self.z_dim = config.DEG.Learn.z_dim
        self.game_name = game_name
        self.action_number = config.DRL.Learn.actions
        self.action_type = config.DRL.Learn.action_type
        self.global_model_data_path = global_model_data_path

        self.num_simulations = config.Mimic.Learn.num_simulations
        self.episodic_sample_number = config.Mimic.Learn.episodic_sample_number
        self.data_save_dir = self.global_model_data_path + config.DEG.Learn.dset_dir
        # self.image_type = config.DEG.Learn.image_type
        self.image_type = None
        self.iteration_number = 0
        self.method = method
        if 'cart' in self.method:
            self.mimic_model = CARTRegressionTree(model_name=self.method, options=options)
        elif 'm5' in self.method:
            from mimic_learner.comparsion_learners.m5 import M5Tree
            self.mimic_model = M5Tree(model_name=self.method, options=options)
        else:
            self.mimic_model = None

        if self.method == 'mcts' or self.method == 'cart-fvae':
            self.deg_model_name = deg_model_name
            if deg_model_name.split('-')[0] == 'FVAE':
                self.ignored_dim = ast.literal_eval(config.DEG.FVAE.ignore_dim)
            elif deg_model_name.split('-')[0] == 'CVAE':
                self.ignored_dim = ast.literal_eval(config.DEG.CVAE.ignore_dim)
            elif deg_model_name.split('-')[0] == 'VAE':
                self.ignored_dim = ast.literal_eval(config.DEG.CVAE.ignore_dim)
            print("Ignored dim is {0}".format(self.ignored_dim), file=log_file)
            # initialize dientangler
            # self.dientangler = Disentanglement(config, deg_model_name.split('-')[0] , local_test_flag, self.global_model_data_path)
            # self.dientangler.load_checkpoint(ckptname=deg_model_name, testing_flag=True, log_file=log_file)
            if len(options) > 0:
                self.binary_max_node = options[1]
                self.saved_model_c_puct = options[3]
                self.play_number = options[5]
                pass

        # if not local_test_flag:
        #     self.dientangler.load_checkpoint()

        # experience replay
        # self.memory = PrioritizedReplay(capacity=config.Mimic.Learn.replay_memory_size)
        self.memory = None
        self.mcts_saved_dir = "" if local_test_flag else config.Mimic.Learn.mcts_saved_dir
        self.max_k = config.Mimic.Learn.max_k

        self.shell_saved_model_dir = self.global_model_data_path + self.mcts_saved_dir

    def static_data_loader(self, action_id, log_file, img_type, training_flag=True, run_tmp_test=False, test_id = 0):
        print("Reading from static data", file=log_file)
        if run_tmp_test:
            tmp_msg = 'tmp_'
        else:
            tmp_msg = ''
        self.memory = []
        cwd = os.getcwd()

        if training_flag:
            if img_type == 'latent':
                read_data_dir = cwd.replace('/interface', '') + '/LMUT_data/' \
                                                                     '{3}impact_training_latent_data_' \
                                                                     '{0}_action_{1}_{2}_expand.csv'.format(self.game_name,
                                                                                                     action_id,
                                                                                                     self.deg_model_name.split('-')[0],
                                                                                                     tmp_msg)
            else:
                read_data_dir = cwd.replace('/interface', '') + '/LMUT_data/' \
                                                                     '{2}impact_training_{3}_data_' \
                                                                     '{0}_action_{1}.csv'.format(self.game_name,
                                                                                                 action_id,
                                                                                                 tmp_msg,
                                                                                                 img_type)
            # else:
            #     raise ValueError("Unknown image type {0}".format(img_type))
        else:
            if img_type == 'latent':
                read_data_dir = cwd.replace('/interface', '') + '/LMUT_data/' \
                                                                    '{3}impact_testing_latent_data_' \
                                                                    '{0}_action_{1}_{2}_iter{4}.csv'.format(self.game_name,
                                                                                                            action_id,
                                                                                                            self.deg_model_name.split('-')[0],
                                                                                                            tmp_msg,
                                                                                                            test_id)
            else:
                read_data_dir = cwd.replace('/interface', '') + '/LMUT_data/' \
                                                                     '{2}impact_testing_{4}_data_' \
                                                                     '{0}_action_{1}_iter{3}.csv'.format(self.game_name,
                                                                                                         action_id,
                                                                                                         tmp_msg,
                                                                                                         test_id,
                                                                                                         img_type)
            # else:
            #     raise ValueError("Unknown image type {0}".format(img_type))

        print("reading data from {0}".format(read_data_dir), file=log_file)
        # print("Testing")
        skip_line = True
        with open(read_data_dir, 'r') as csvfile:
            csv_read_line = csv.reader(csvfile)
            for row in csv_read_line:
                if skip_line:
                    skip_line = False
                    continue
                impact = float(row[0])
                action_id = int(row[-2])
                cumu_reward = float(row[-1])
                z0 = np.asarray([float(i) for i in row[1:-2]])
                # action_id = None
                # cumu_reward = None
                # z0 = np.asarray([float(i) for i in row[1:]])
                self.memory.append([z0, action_id, cumu_reward, impact])

    def data_loader(self, episode_number, target, action_id):
        self.memory = []
        if target == "raw":
            self.image_type = 'origin'
        elif target == 'latent':
            self.image_type = 'origin'
        else:
            self.image_type = target

        print_latent_total_number = 10
        print_latent_iter = 0

        with open(self.data_save_dir + '/' + self.game_name + '/action_values.txt', 'r') as f:
            action_values = f.readlines()

        [action_index_t0, action_values_list_t0,
         reward_t0, value_t0] = gather_data_values(action_values[self.iteration_number], self.action_number, self.game_name)
        image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(self.data_save_dir,
                                                                                 self.game_name,
                                                                                 self.image_type,
                                                                                 self.iteration_number,
                                                                                 action_index_t0))

        if target == "latent":
            x_t0_resized = image
            with torch.no_grad():
                x_t0 = ttf.to_tensor(x_t0_resized).unsqueeze(0).to(self.dientangler.device)
                z0 = self.dientangler.VAE.encode(x_t0).squeeze()[:self.z_dim]
                z0 = z0.cpu().numpy()
        elif target == "raw" or target == 'color' or target == 'binary':
            flatten_image_t0 = np.array(image).flatten()/255.0
            # image = np.array(image)
            # image = image/255.0
        else:
            raise ValueError("Unknown data loader target {0}".format(target))
        data_length = self.episodic_sample_number * episode_number-self.iteration_number
        while len(self.memory) < data_length:
            [action_index_t1, action_values_list_t1,
             reward_t1, value_t1] = gather_data_values(action_values[self.iteration_number + 1], self.action_number, self.game_name)
            if self.game_name == 'flappybird':
                delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0
            elif self.game_name == 'Assault-v0' or self.game_name == 'SpaceInvaders-v0':
                delta = value_t1 - value_t0 + reward_t0
            else:
                raise ValueError('Unknown game {0}'.format(self.game_name))

            image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(self.data_save_dir,
                                                                                     self.game_name,
                                                                                     self.image_type,
                                                                                     self.iteration_number + 1,
                                                                                     action_index_t1))
            if target == "latent":
                x_t1_resized = image
                with torch.no_grad():
                    x_t1 = ttf.to_tensor(x_t1_resized).unsqueeze(0).to(self.dientangler.device)
                    z1 = self.dientangler.VAE.encode(x_t1).squeeze()[:self.z_dim]
                    z1 = z1.cpu().numpy()
                if print_latent_iter < print_latent_total_number:
                    # print(z0)
                    print_latent_iter += 1
                # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
                if action_index_t0 == action_id:
                    # print(len(self.memory))
                    self.memory.append([z0, action_index_t0, reward_t0, z1, delta])
                z0 = z1
            elif target == "raw" or target == 'color' or target == 'binary':
                flatten_image_t1 = np.array(image).flatten()/255.0
                if action_index_t0 == action_id:
                    self.memory.append([flatten_image_t0, action_index_t0, reward_t0, flatten_image_t1, delta])
                flatten_image_t0 = flatten_image_t1
            else:
                raise ValueError("Unknown data loader target {0}".format(target))

            self.iteration_number += 1
            action_index_t0 = action_index_t1
            action_values_list_t0 = action_values_list_t1
            reward_t0 = reward_t1
            value_t0 = value_t1

        print('loading finished')



    def iterative_read_binary_tree(self, binary_node, log_file, selected_binary_node_index=[],
                                   visualize_flag=False, indent_number=0, img_id=0):

        # if binary_node.level >= self.binary_max_node:
        #     selected_binary_node_index.append(binary_node)
        #     return selected_binary_node_index, img_id-1

        if binary_node.left_child is None and binary_node.right_child is None:
            selected_binary_node_index.append(binary_node)
            print("---" * indent_number + 'Leaf Node with action {0} and Impact {1}.'.format(
                binary_node.action,
                binary_node.prediction
            ),
                  file=log_file)
            return selected_binary_node_index, img_id-1

        print("---" * indent_number + 'Split Node with action {0} and Impact {1} and Image id {2}.'.format(
            binary_node.action,
            binary_node.prediction,
            img_id
        ),
              file=log_file)

        if visualize_flag and indent_number == 0:
            state_features_all = []
            action_all = []
            reward_all = []
            for data_index in binary_node.state:
                state_features_all.append(self.mimic_env.data_all[data_index][0])
                action_all.append(self.mimic_env.data_all[data_index][1])
                reward_all.append(self.mimic_env.data_all[data_index][2])
            state_features_all_avg = np.average(np.asarray(state_features_all), axis=0)
            z_state = build_decode_input(state_features_all_avg)
            with torch.no_grad():
                if self.deg_model_name.split('-')[0] == 'CVAE':
                    x_recon = F.sigmoid(self.dientangler.CVAE.decode(z_state.float().to(self.dientangler.device))).data
                else:
                    x_recon = F.sigmoid(self.dientangler.VAE.decode(z_state.float().to(self.dientangler.device))).data
                from torchvision.utils import save_image
                save_image(tensor=x_recon,
                           fp="../mimic_learner/action_images_plots/img_root_image.jpg", nrow=1, pad_value=1)

        indent_number+=1

        if visualize_flag:
            decoder = self.dientangler.VAE.decode
            device = self.dientangler.device
            z_dim = int(self.mimic_env.n_action_types)
            data_all = self.mimic_env.data_all
            visualize_split(binary_node.action, [binary_node.left_child.state, binary_node.right_child.state],
                            data_all, decoder, device, z_dim, img_id)

        selected_binary_node_index, img_id = self.iterative_read_binary_tree(binary_node.left_child,
                                                                             log_file,
                                                                             selected_binary_node_index,
                                                                             visualize_flag,
                                                                             indent_number,
                                                                             img_id + 1)
        selected_binary_node_index, img_id = self.iterative_read_binary_tree(binary_node.right_child,
                                                                             log_file,
                                                                             selected_binary_node_index,
                                                                             visualize_flag,
                                                                             indent_number,
                                                                             img_id + 1)

        return selected_binary_node_index, img_id

    def predict_mcts(self, data, action_id, saved_nodes_dir, log_file, visualize_flag=False, feature_importance_flag=False):
        self.mimic_env.assign_data(data)
        init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
        moved_nodes = test_mcts(saved_nodes_dir=saved_nodes_dir, TreeEnv=self.mimic_env, action_id=action_id)
        return self.generate_prediction_results(init_state, init_var_list, moved_nodes,
                                                self.binary_max_node, log_file, visualize_flag, feature_importance_flag)

    def predict_mcts_by_splits(self, action_id, saved_nodes_dir):
        init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
        moved_nodes = test_mcts(saved_nodes_dir=saved_nodes_dir, TreeEnv=self.mimic_env, action_id=action_id)
        return_value_log_all = []
        return_value_log_struct_all = []
        return_value_var_reduction_all = []
        return_value_var_reduction_by_leaf_all = []
        mae_all = []
        rmse_all = []
        leaves_number_all = []
        for node_index in range(0, len(moved_nodes)):
            return_value_log, return_value_log_struct, return_value_var_reduction, mae, rmse, leaves_number = \
                self.generate_prediction_results(init_state, init_var_list, moved_nodes,
                                                 node_index, log_file=None, visualize_flag=False,
                                                 feature_importance_flag=False)
            return_value_log_all.append(return_value_log)
            return_value_log_struct_all.append(return_value_log_struct)
            return_value_var_reduction_all.append(return_value_var_reduction)
            if node_index>0:
                return_value_var_reduction_by_leaf_all.append(float(return_value_var_reduction)/node_index)
            else:
                return_value_var_reduction_by_leaf_all.append(float(0))
            mae_all.append(mae)
            rmse_all.append(rmse)
            leaves_number_all.append(leaves_number)
        return return_value_log_all, return_value_log_struct_all, return_value_var_reduction_all, \
               return_value_var_reduction_by_leaf_all, mae_all, rmse_all, leaves_number_all

    def generate_prediction_results(self, init_state, init_var_list, moved_nodes, max_node, log_file,
                                    visualize_flag, feature_importance_flag):
        parent_state = init_state
        state = init_state
        total_data_length = len(state[0])
        parent_var_list = init_var_list
        root_binary = BinaryTreeNode(state=init_state[0], level=0, prediction=moved_nodes[0].state_prediction[0])
        binary_node_index = [root_binary]

        if feature_importance_flag:
            feature_importance_all = {}

        for moved_node in moved_nodes[:max_node]:
            selected_action = moved_node.action
            if selected_action is not None:
                state, new_var_list = self.mimic_env.next_state(state=parent_state, action=selected_action,
                                                                parent_var_list = parent_var_list)
                split_index = int(selected_action.split('_')[0])
                split_binary_node = binary_node_index[split_index]
                parent_state = state
                parent_var_list = new_var_list
                split_binary_node.action = selected_action
                split_binary_node.left_child = BinaryTreeNode(state=state[split_index],
                                                              level=split_binary_node.level+1,
                                                              prediction = moved_node.state_prediction[split_index])
                split_binary_node.right_child = BinaryTreeNode(state=state[split_index+1],
                                                               level=split_binary_node.level+1,
                                                               prediction = moved_node.state_prediction[split_index+1])
                binary_node_index.pop(split_index)
                binary_node_index.insert(split_index, split_binary_node.left_child)
                binary_node_index.insert(split_index+1, split_binary_node.right_child)

                if feature_importance_flag:
                    feature_dim = selected_action.split('_')[1]
                    parent_impacts = []
                    for index in split_binary_node.state:
                        parent_impacts.append(self.mimic_env.data_all[index][-1])
                    parent_var = np.var(parent_impacts)

                    left_child_impacts = []
                    for index in state[split_index]:
                        left_child_impacts.append(self.mimic_env.data_all[index][-1])
                    left_child_var = np.var(left_child_impacts)

                    right_child_impacts = []
                    for index in state[split_index+1]:
                        right_child_impacts.append(self.mimic_env.data_all[index][-1])
                    right_child_var = np.var(right_child_impacts)

                    var_reduction = float(len(split_binary_node.state))/total_data_length*parent_var - \
                                    float(len(state[split_index]))/total_data_length*left_child_var-\
                                    float(len(state[split_index+1]))/total_data_length*right_child_var

                    if feature_importance_all.get(feature_dim) is not None:
                        feature_importance_all[feature_dim] += var_reduction
                    else:
                        feature_importance_all.update({feature_dim:var_reduction})

        binary_states = []
        binary_predictions = [None for i in range(len(self.mimic_env.data_all))]

        if  visualize_flag:
            self.iterative_read_binary_tree(root_binary, log_file, visualize_flag=visualize_flag)
        if feature_importance_flag:
            print(feature_importance_all)

        selected_binary_node_index = binary_node_index

        # state_predictions = []
        # for binary_node in selected_binary_node_index:
        #     state_target_values = []
        #     for data_index in binary_node.state:
        #         state_target_values.append(data[data_index][-1])
        #     state_predictions.append(sum(state_target_values) / len(state_target_values))

        for binary_node in selected_binary_node_index:
            binary_states.append(binary_node.state)
            for data_index in binary_node.state:
                binary_predictions[data_index] = binary_node.prediction

        return_value_log = self.mimic_env.get_return(state=binary_states)
        return_value_log_struct = self.mimic_env.get_return(state=binary_states, apply_structure_cost=True)
        return_value_var_reduction = self.mimic_env.get_return(state=binary_states, apply_variance_reduction=True)

        # predictions = [None for i in range(len(data))]
        # assert len(state) == len(moved_nodes[-1].state_prediction)
        # for subset_index in range(len(state)):
        #     subset = state[subset_index]
        #     for data_index in subset:
        #         predictions[data_index] = moved_nodes[-1].state_prediction[subset_index]
        #
        # for predict_index in range(len(predictions)):
        #     pred_diff = binary_predictions[predict_index] - predictions[predict_index]
        #     print(pred_diff)

        ae_all = []
        se_all = []
        for data_index in range(len(binary_predictions)):
            if binary_predictions[data_index] is not None:
                real_value = self.mimic_env.data_all[data_index][-1]
                predicted_value = binary_predictions[data_index]
                ae = abs(real_value-predicted_value)
                ae_all.append(ae)
                mse = ae**2
                se_all.append(mse)
        mae = np.mean(ae_all)
        mse = np.mean(se_all)
        rmse = (mse)**0.5
        leaves_number = len(state)
        return return_value_log, return_value_log_struct, return_value_var_reduction, mae, rmse, leaves_number

    def test_mimic_model(self, action_id, log_file, data_type, disentangler_name, run_tmp_test, test_id):
        self.iteration_number = int(self.episodic_sample_number * (45+test_id*0.1)) # the last 5k(/50k) is for testing
        if self.method == 'mcts':
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=False,
                                    run_tmp_test=run_tmp_test, test_id=test_id)
            # self.data_loader(episode_number=45.5, target=data_type, action_id=action_id)  # divided into training, validation and testing
            saved_nodes_dir = self.get_MCTS_nodes_dir(action_id, disentangler_name)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                 = self.predict_mcts(self.memory, action_id, saved_nodes_dir, log_file, visualize_flag=False,
                                     feature_importance_flag=False)
            # return_value, mae, rmse, leaves_number = [None, None, None, None]
        elif self.method == 'cart-fvae':
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=False,
                                    run_tmp_test=run_tmp_test, test_id=test_id)
            # self.data_loader(episode_number=45.5, target=data_type, action_id=action_id)  # divided into training, validation and testing
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            testing_data = [[], []]
            for data_index in init_state[0]:
                data_input_row_list = []
                for i in range(len(self.memory[data_index][0])):
                    if i not in self.ignored_dim:
                        data_input_row_list.append(self.memory[data_index][0][i])
                data_input = np.asarray(data_input_row_list)
                # data_input = self.memory[data_index][0]
                data_output = self.memory[data_index][3]
                testing_data[0].append(data_input)
                testing_data[1].append(data_output)
            testing_data[0] = np.stack(testing_data[0], axis=0)
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/cart/' \
                                                           '{0}/{1}-aid{2}-sklearn-{3}.model'.format(self.game_name,
                                                                                                 self.method,
                                                                                                 action_id,
                                                                                                     disentangler_name)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                 = self.mimic_model.test_mimic(testing_data=testing_data,
                                               save_model_dir=save_model_dir,
                                               mimic_env=self.mimic_env,
                                               log_file=log_file)
        elif self.method == 'cart':
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=False,
                                    run_tmp_test=run_tmp_test, test_id=test_id)
            # self.data_loader(episode_number=45.5+test_id*0.1, target=data_type, action_id=action_id)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            testing_data = [[], []]
            for data_index in init_state[0]:
                data_input = self.memory[data_index][0]
                data_output = self.memory[data_index][3]
                testing_data[0].append(data_input)
                testing_data[1].append(data_output)
            testing_data[0] = np.stack(testing_data[0], axis=0)
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/cart/' \
                                                           '{0}/{1}-aid{2}-sklearn-{3}.model'.format(self.game_name,
                                                                                                 self.method,
                                                                                                 action_id,
                                                                                                     disentangler_name)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                 =  self.mimic_model.test_mimic(testing_data=testing_data,
                                                save_model_dir=save_model_dir,
                                                mimic_env=self.mimic_env,
                                                log_file=log_file)
        elif self.method == 'm5-rt' or self.method == 'm5-mt':
            from mimic_learner.comparsion_learners.m5 import generate_weka_training_data
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=False,
                                    run_tmp_test=run_tmp_test, test_id=test_id)
            # self.data_loader(episode_number=45.5+test_id*0.1, target=data_type, action_id=action_id)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            data_dir = self.data_save_dir + '/' + self.game_name + '/m5-weka/m5-aid{0}-tree-testing-iter{1}.csv'.format(action_id, test_id)
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/M5/' \
                                                           '{0}/{1}-aid{2}-weka.model'.format(self.game_name,
                                                                                              self.method,
                                                                                              action_id)
            # if not os.path.exists(data_dir):
            generate_weka_training_data(data=self.memory, action_id=action_id, dir=data_dir)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number = self.mimic_model.test_weka_model(testing_data_dir=data_dir,
                                                                                     save_model_dir=save_model_dir,
                                                                                     log_file=log_file,
                                                                                     mimic_env=self.mimic_env)
        else:
            raise ValueError('Unknown method {0}'.format(self.method))


        results_str = "Testing action {0}: return_value_log:{1}, " \
                      "return_value_log_struct: {2}, return_value_var_reduction {3}," \
                      "mae:{4}, rmse:{5}, leaves:{6}".format(action_id,
                                                             str(return_value_log)+"({0})".format(
                                                                 float(return_value_log)/leaves_number),
                                                             str(return_value_log_struct) + "({0})".format(
                                                                 float(return_value_log_struct) / leaves_number),
                                                             str(return_value_var_reduction) + "({0})".format(
                                                                 float(return_value_var_reduction) / leaves_number),
                                                             str(mae) + "({0})".format(float(mae) / leaves_number),
                                                             str(rmse) + "({0})".format(float(rmse) / leaves_number),
                                                             leaves_number)

        print(results_str, file=log_file)
        return return_value_log, return_value_log_struct, \
               return_value_var_reduction, mae, rmse, leaves_number, results_str


    def get_MCTS_nodes_dir(self, action_id, disentangler_name):

        if self.game_name == 'flappybird' and action_id == 0:
            if disentangler_name == 'FVAE':
                if self.saved_model_c_puct == 0.01 and self.play_number == 1000:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_2020-04-30/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 1  and self.play_number == 1000:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_2020-07-01/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.05  and self.play_number == 1000:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_2020-07-10/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.02  and self.play_number == 1000:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_02_2020-07-13/'.format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.05 and self.play_number == 250:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_05_2020-07-20/'.format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0 and self.play_number == 2:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_2020-07-22/'.format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.01 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_01_2020-07-25/'.format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.1 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_1_2020-07-27/'.format(self.game_name, action_id)
                else:
                    raise ValueError("Unknown save model")
            elif disentangler_name == 'CVAE':
                if self.saved_model_c_puct == 0.01 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_01_DEGCVAE_2020-11-11/'.format(self.game_name,
                                                                                                   action_id)
                elif self.saved_model_c_puct == 0.005 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_005_DEGCVAE_2020-11-16/'.format(self.game_name,
                                                                                                   action_id)
                elif self.saved_model_c_puct == 0.1 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      'saved_nodes_action{1}_CPUCT0_1_DEGCVAE_2020-11-22/'.format(self.game_name,
                                                                                                   action_id)
                else:
                    raise ValueError("Unknown save model")
        elif self.game_name == 'SpaceInvaders-v0' and action_id == 4:
            if disentangler_name == 'FVAE':
                if self.saved_model_c_puct == 0.001 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_001_2020-08-03/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.005 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_005_2020-08-13/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.01 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_01_2020-08-14/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0 and self.play_number == 2:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_0_2020-08-10/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.0005 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_0005_2020-08-24/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.003 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_003_2020-08-31/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.008 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_008_2020-08-14/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.015 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_015_2020-08-15/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.0001 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_0001_2020-08-25/".format(self.game_name, action_id)
            elif disentangler_name == 'CVAE':
                if self.saved_model_c_puct == 0.01 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_01_DEGCVAE_2020-11-27/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.05 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_05_DEGCVAE_2020-12-04/".format(self.game_name, action_id)
                elif self.saved_model_c_puct == 0.1 and self.play_number == 200:
                    saved_nodes_dir = self.global_model_data_path + \
                                      "/DRL-interpreter-model/MCTS/{0}/" \
                                      "saved_nodes_action{1}_CPUCT0_1_DEGCVAE_2020-12-09/".format(self.game_name, action_id)
        else:
            raise ValueError('Unknown MCTS dir')

        return saved_nodes_dir

    def train_mimic_model(self, action_id, shell_round_number, log_file, launch_time, data_type, disentangler_name,
                          run_mcts=False, c_puct=None, play=None, run_tmp_test=False):
        # mcts_file_name = None
        return_value_log = None
        if self.method == 'mcts':
            # self.data_loader(episode_number=4, target=data_type, action_id=action_id)
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=True)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            if run_mcts:
                if c_puct is not None:
                    mcts.c_PUCT = c_puct
                mcts_saved_dir = self.global_model_data_path + self.mcts_saved_dir
                shell_saved_model_dir = mcts_saved_dir+'_tmp_shell_saved_action{0}_CPUCT{1}' \
                                                       '_play{2}_{3}.pkl'.format(action_id, c_puct, play, launch_time)
                execute_episode_single(num_simulations=self.num_simulations,
                                       TreeEnv=self.mimic_env,
                                       tree_writer=None,
                                       mcts_saved_dir=mcts_saved_dir,
                                       max_k=self.max_k,
                                       init_state=init_state,
                                       init_var_list=init_var_list,
                                       action_id=action_id,
                                       ignored_dim=self.ignored_dim,
                                       shell_round_number= shell_round_number,
                                       shell_saved_model_dir = shell_saved_model_dir,
                                       log_file = log_file,
                                       disentangler_name=disentangler_name,
                                       apply_split_parallel=True,
                                       play=play)
                return_value, mae, rmse, leaves_number = [None, None, None, None]
            else:
                saved_nodes_dir = self.get_MCTS_nodes_dir(action_id, disentangler_name)
                return_value_log, return_value_log_struct, \
                return_value_var_reduction, mae, rmse, leaves_number \
                    = self.predict_mcts(self.memory, action_id, saved_nodes_dir, log_file, visualize_flag=True,
                                        feature_importance_flag=True)
        elif self.method == 'cart-fvae':
            # self.data_loader(episode_number=4, target=data_type, action_id=action_id)
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=True, run_tmp_test=run_tmp_test)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            training_data = [[],[]]
            for data_index in init_state[0]:
                data_input_row_list = []
                for i in range(len(self.memory[data_index][0])):
                    if i not in self.ignored_dim:
                        data_input_row_list.append(self.memory[data_index][0][i])
                data_input = np.asarray(data_input_row_list)
                # data_input = np.concatenate([self.memory[data_index][0]],axis=0)
                data_output = self.memory[data_index][3]
                training_data[0].append(data_input)
                training_data[1].append(data_output)
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/cart/' \
                                                           '{0}/{1}-aid{2}-sklearn-{3}.model'.format(self.game_name,
                                                                                                 self.method,
                                                                                                 action_id,
                                                                                                     disentangler_name)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                = self.mimic_model.train_mimic(training_data=training_data,
                                               save_model_dir=save_model_dir,
                                               mimic_env = self.mimic_env,
                                               log_file=log_file)
        elif self.method == 'cart':
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=True)
            # self.data_loader(episode_number=4, target=data_type, action_id=action_id)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            training_data = [[], []]
            for data_index in init_state[0]:
                data_input = self.memory[data_index][0]
                data_output = self.memory[data_index][3]
                training_data[0].append(data_input)
                training_data[1].append(data_output)
            training_data[0] = np.stack(training_data[0], axis=0)
            # cart.train_2d_tree()
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/cart/' \
                                                           '{0}/{1}-aid{2}-sklearn-{3}.model'.format(self.game_name,
                                                                                                 self.method,
                                                                                                 action_id,
                                                                                                     disentangler_name)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                = self.mimic_model.train_mimic(training_data=training_data,
                                               save_model_dir=save_model_dir,
                                               mimic_env = self.mimic_env,
                                               log_file=log_file)

        elif self.method == 'm5-rt' or self.method == 'm5-mt':
            from mimic_learner.comparsion_learners.m5 import generate_weka_training_data
            # self.data_loader(episode_number=4, target=data_type, action_id=action_id)
            self.static_data_loader(action_id, img_type=data_type, log_file=log_file, training_flag=True)
            self.mimic_env.assign_data(self.memory)
            init_state, init_var_list = self.mimic_env.initial_state(action=action_id)
            data_dir = self.data_save_dir + '/' + self.game_name + '/m5-weka/m5-aid{0}-tree-training.csv'.format(action_id)
            save_model_dir = self.global_model_data_path + '/DRL-interpreter-model/comparison/M5/' \
                                                           '{0}/{1}-aid{2}-weka.model'.format(self.game_name,
                                                                                              self.method,
                                                                                              action_id)

            # if not os.path.exists(data_dir):
            generate_weka_training_data(data=self.memory, action_id=action_id, dir=data_dir)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number = self.mimic_model.train_weka_model(training_data_dir=data_dir,
                                                                                      save_model_dir=save_model_dir,
                                                                                      log_file=log_file,
                                                                                      mimic_env=self.mimic_env)

        else:
            raise ValueError('Unknown method {0}'.format(self.method))
        if return_value_log is not None:
            results_str = "Training action {0}: return_value_log:{1}, " \
                          "return_value_log_struct: {2}, return_value_var_reduction {3}," \
                          "mae:{4}, rmse:{5}, leaves:{6}".format(action_id,
                                                                 str(return_value_log)+"({0})".format(
                                                                     float(return_value_log)/leaves_number),
                                                                 str(return_value_log_struct) + "({0})".format(
                                                                     float(return_value_log_struct) / leaves_number),
                                                                 str(return_value_var_reduction) + "({0})".format(
                                                                     float(return_value_var_reduction) / leaves_number),
                                                                 str(mae) + "({0})".format(float(mae) / leaves_number),
                                                                 str(rmse) + "({0})".format(float(rmse) / leaves_number),
                                                                 leaves_number)

            print(results_str, file=log_file)
            return return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number, results_str





class BinaryTreeNode():
    def __init__(self, state, level, prediction):
        self.state = state
        self.left_child = None
        self.right_child = None
        self.action = None
        self.level = level
        self.prediction = prediction

