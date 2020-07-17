import csv
import os
import matplotlib.pyplot as plt
import traceback
import numpy as np

from utils.plot_utils import plot_values_by_node

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from config.mimic_config import DRLMimicConfig
from mimic_learner.learner import MimicLearner


def run_plot(game_name):
    reward_log_all = {}
    reward_log_struct_all = {}
    var_reduction_all = {}
    mae_all = {}
    rmse_all = {}
    leaves_all = {}

    if game_name == 'flappybird':
        cpuct = 0.01
        action_ids = [0, 1]
    else:
        raise ValueError('Unknown game name')

    for action in action_ids:
        reward_log_all.update({action: []})
        reward_log_struct_all.update({action: []})
        var_reduction_all.update({action: []})
        mae_all.update({action: []})
        rmse_all.update({action: []})
        leaves_all.update({action: []})

        plot_results_path = '../results/plot_results/' \
                            'testing-flappybird-action{0}-by-splits' \
                            '-results-mcts-max_node-None-cpuct-{0}.txt'.format(cpuct, action)
        with open(plot_results_path, 'rb') as f:
            read_value_lines = f.readlines()
        for read_value_line in read_value_lines[1:]:
            [reward_log, reward_log_struct,
             var_reduction, mae, rmse, leaves] = read_value_line.split(',')
            reward_log_all[action].append(reward_log)
            reward_log_struct_all[action].append(reward_log_struct)
            var_reduction_all[action].append(var_reduction)
            mae_all[action].append(mae)
            rmse_all[action].append(rmse)
            leaves_all[action].append(leaves)

    plot_x_values  = []
    plot_y_values = []
    for leaf_node in range(1, 60, 1):
        plot_x_values.append(leaf_node)
        plot_y_values.append([])
        for action in action_ids:
            for value in var_reduction_all[action]:
                if value[-1] == leaf_node:
                    plot_y_values[-1].append(value)
                    break
        assert len(plot_y_values[-1]) == len(action_ids)
    plot_y_values = np.asarray(plot_y_values).mean(axis=1)

    plot_values_by_node([plot_x_values], [plot_y_values], value_type='var_reduction')




def run_generate_values():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    game_name = 'flappybird'
    method = 'mcts'
    action_id = 0
    if game_name == 'Assault-v0':
        # action_ids = [2, 3, 4]  # {0: 118, 1: 165, 2: 1076, 3: 1293, 4: 1246, 5: 50, 6: 52}
        model_name = 'FVAE-1000000'
        config_path = "../environment_settings/assault_v0_config.yaml"
    elif game_name == 'SpaceInvaders-v0':
        model_name = 'FVAE-1000000'
        config_path = "../environment_settings/space_invaders_v0_config.yaml"
    elif game_name == 'flappybird':
        # action_ids = [0, 1]
        model_name = 'FVAE-1000000'
        config_path = "../environment_settings/flappybird_config.yaml"
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    if method == 'mcts':
        options_dict = {
            'flappybird':['max_node', None, 'cpuct', 0.02],
            # 'Assault-v0':[]
        }
    elif method == 'cart-fvae':
        options_dict = {
            'flappybird': ['max_leaf_nodes', None, 'criterion', 'mae', 'best'],
            # 'Assault-v0': ['max_leaf_nodes', 15, 'criterion', 'mae']
        }
    else:
        raise ValueError("unknown model name {0}".format(method))
    options = options_dict[game_name]

    option_str = '-'.join([str(option) for option in options])

    training_results_saving_dir = '../results/plot_results/training-{0}-action{1}' \
                                  '-by-splits-results-{2}-{3}.txt'.format(game_name, action_id, method, option_str)
    training_results_writer = open(training_results_saving_dir, 'w')
    train_results_csv_writer = csv.writer(training_results_writer)

    testing_results_saving_dir = '../results/plot_results/testing-{0}-action{1}' \
                                 '-by-splits-results-{2}-{3}.txt'.format(game_name, action_id, method, option_str)
    testing_results_writer = open(testing_results_saving_dir, 'w')
    test_results_csv_writer = csv.writer(testing_results_writer)

    local_test_flag = False
    if local_test_flag:
        mimic_config = DRLMimicConfig.load(config_path)
        mimic_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
        mimic_config.Mimic.Learn.episodic_sample_number = 49
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        mimic_config = DRLMimicConfig.load(config_path)
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        mimic_config = DRLMimicConfig.load(config_path)
        global_model_data_path = "/home/functor/scratch/Galen/project-DRL-Interpreter"
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")

    print('global path is : {0}'.format(global_model_data_path))
    log_file = None

    print("\nRunning for game {0} with {1}".format(game_name, method), file=log_file)
    mimic_learner = MimicLearner(game_name=game_name,
                                 method=method,
                                 config=mimic_config,
                                 deg_model_name=model_name,
                                 local_test_flag=local_test_flag,
                                 global_model_data_path=global_model_data_path,
                                 log_file=log_file,
                                 options=options)
    # for action_id in [1]:
    mimic_learner.iteration_number = 0
    train_results_csv_writer.writerow(['return_value_log', 'return_value_log_struct', 'return_value_var_reduction',
                                 'mae', 'rmse', 'leaves'])
    mimic_learner.data_loader(episode_number=4, target="latent", action_id=action_id)
    mimic_learner.mimic_env.assign_data(mimic_learner.memory)
    if method == 'mcts':
        saved_nodes_dir = mimic_learner.get_MCTS_nodes_dir(action_id)
        return_value_log_all, return_value_log_struct_all, return_value_var_reduction_all, \
        mae_all, rmse_all, leaves_number_all = mimic_learner.predict_mcts_by_splits(action_id, saved_nodes_dir)
    elif method == 'cart-fvae':
        return_value_log_all = []
        return_value_log_struct_all = []
        return_value_var_reduction_all = []
        mae_all = []
        rmse_all = []
        leaves_number_all = []
        init_state, init_var_list = mimic_learner.mimic_env.initial_state(action=action_id)
        training_data = [[], []]
        for data_index in init_state[0]:
            data_input = np.concatenate([mimic_learner.memory[data_index][0]], axis=0)
            data_output = mimic_learner.memory[data_index][4]
            training_data[0].append(data_input)
            training_data[1].append(data_output)
        for i in range(2, 61):
            save_model_dir = mimic_learner.global_model_data_path + '/DRL-interpreter-model/comparison' \
                                                                    '/cart/{0}/{1}-aid{2}-node{3}' \
                                                                    '-sklearn.model'.format(mimic_learner.game_name,
                                                                                            mimic_learner.method,
                                                                                            action_id,
                                                                                            i)
            mimic_learner.mimic_model.max_leaf_nodes = i
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                = mimic_learner.mimic_model.train_mimic(training_data=training_data,
                                               save_model_dir=save_model_dir,
                                               mimic_env=mimic_learner.mimic_env,
                                               log_file=log_file)
            return_value_log_all.append(return_value_log)
            return_value_log_struct_all.append(return_value_log_struct)
            return_value_var_reduction_all.append(return_value_var_reduction)
            mae_all.append(mae)
            rmse_all.append(rmse)
            leaves_number_all.append(leaves_number)
    else:
        raise ValueError("Unknown method {0}".format(method))

    for i in range(len(return_value_log_all)):
        train_results_csv_writer.writerow([round(return_value_log_all[i], 4),
                                     round(return_value_log_struct_all[i], 4),
                                     round(return_value_var_reduction_all[i], 4),
                                     round(mae_all[i], 4),
                                     round(rmse_all[i], 4),
                                     leaves_number_all[i]])

    mimic_learner.iteration_number = int(mimic_learner.episodic_sample_number * 45)
    test_results_csv_writer.writerow(['return_value_log', 'return_value_log_struct', 'return_value_var_reduction',
                                 'mae', 'rmse', 'leaves'])
    mimic_learner.data_loader(episode_number=45.5, target="latent", action_id=action_id)
    mimic_learner.mimic_env.assign_data(mimic_learner.memory)
    if method == 'mcts':
        saved_nodes_dir = mimic_learner.get_MCTS_nodes_dir(action_id)
        return_value_log_all, return_value_log_struct_all, return_value_var_reduction_all, \
        mae_all, rmse_all, leaves_number_all = mimic_learner.predict_mcts_by_splits(action_id, saved_nodes_dir)
    elif method == 'cart-fvae':
        return_value_log_all = []
        return_value_log_struct_all = []
        return_value_var_reduction_all = []
        mae_all = []
        rmse_all = []
        leaves_number_all = []
        init_state, init_var_list = mimic_learner.mimic_env.initial_state(action=action_id)
        testing_data = [[], []]
        for data_index in init_state[0]:
            data_input = mimic_learner.memory[data_index][0]
            data_output = mimic_learner.memory[data_index][4]
            testing_data[0].append(data_input)
            testing_data[1].append(data_output)
        testing_data[0] = np.stack(testing_data[0], axis=0)
        for i in range(2, 61):
            save_model_dir = mimic_learner.global_model_data_path + '/DRL-interpreter-model/comparison' \
                                                                    '/cart/{0}/{1}-aid{2}-node{3}' \
                                                                    '-sklearn.model'.format(mimic_learner.game_name,
                                                                                            mimic_learner.method,
                                                                                            action_id,
                                                                                            i)
            return_value_log, return_value_log_struct, \
            return_value_var_reduction, mae, rmse, leaves_number \
                = mimic_learner.mimic_model.test_mimic(testing_data=testing_data,
                                                       save_model_dir=save_model_dir,
                                                       mimic_env=mimic_learner.mimic_env,
                                                       log_file=log_file)
            return_value_log_all.append(return_value_log)
            return_value_log_struct_all.append(return_value_log_struct)
            return_value_var_reduction_all.append(return_value_var_reduction)
            mae_all.append(mae)
            rmse_all.append(rmse)
            leaves_number_all.append(leaves_number)
    else:
        raise ValueError("Unknown method {0}".format(method))

    for i in range(len(return_value_log_all)):
        test_results_csv_writer.writerow([round(return_value_log_all[i], 4),
                                     round(return_value_log_struct_all[i], 4),
                                     round(return_value_var_reduction_all[i], 4),
                                     round(mae_all[i], 4),
                                     round(rmse_all[i], 4),
                                     leaves_number_all[i]])




if __name__ == "__main__":
    run_generate_values()
    exit(0)

