import csv
import os
import numpy as np

from utils.plot_utils import plot_values_by_node

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from config.mimic_config import DRLMimicConfig
from mimic_learner.learner import MimicLearner


def run_plot():


    plotting_target = 'var_reduction'
    game_name = 'flappybird'

    if game_name == 'flappybird':
        action_ids = [0]
        methods = [
            "cart-fvae",
            "vr-lmt-fvae",
            'gn-lmt-fave',
            "mcts"
        ]
    else:
        raise ValueError('Unknown game name')

    plot_x_values_all  = []
    plot_y_values_all = []

    for method in methods:
        var_reduction_all = {}
        rmse_all = {}
        for action in action_ids:
            if game_name == 'flappybird':
                if method == 'cart-fvae':
                    # if plotting_target == 'var_reduction':
                    plot_results_path = '../results/plot_results/' \
                                        'testing-flappybird-action0-by-splits-' \
                                        'results-cart-fvae-max_leaf_nodes-None-' \
                                        'criterion-mae-best-min_samples_leaf-1.txt'.format(action)

                elif method == 'vr-lmt-fvae':
                    # if plotting_target == 'var_reduction':
                    plot_results_path = '../results/plot_results/' \
                                        'testing_tree_impact_training_latent_data' \
                                        '_flappybird_action_{0}_minInst10_regul0.1_bin25_minQ0_splitm3'.format(action)
                elif method == 'gn-lmt-fave':
                    # if plotting_target == 'var_reduction':
                    plot_results_path = '../results/plot_results/' \
                                        'testing_tree_impact_training_latent_data' \
                                        '_flappybird_action_{0}_minInst10_regul1_bin100_minQ0.01_splitm1'.format(action)
                elif method == 'mcts':
                    cpuct = 0.01
                    play_number = 1000
                    plot_results_path = '../results/plot_results/' \
                                        'testing-flappybird-action{0}-by-splits' \
                                        '-results-mcts-max_node-None-cpuct-{1}-play-{2}.txt'.format(action,
                                                                                                    cpuct,
                                                                                                    play_number)
                # elif plotting_target == 'rmse':
                #     if method == 'cart-fvae':
                #         plot_results_path = ''
                #     elif method == 'vr-lmt-fvae':
                #         plot_results_path = ''
                #     elif method == 'gn-lmt-fave':
                #         plot_results_path = ''
                #     elif method == 'mcts':
                #         plot_results_path = ''

            else:
                raise ValueError('Uknown method {0}'.format(method))

            with open(plot_results_path, 'r') as f:
                read_value_lines = f.readlines()
            for read_value_line in read_value_lines[1:]:
                [reward_log, reward_log_struct,
                 var_reduction, mae, rmse, leaves] = read_value_line.split(',')
                leaves = float(leaves.replace('\n',''))
                var_reduction = float(var_reduction)
                rmse = float(rmse)
                if var_reduction_all.get(leaves) is None:
                    var_reduction_all.update({leaves:[var_reduction]})
                    rmse_all.update({leaves:[rmse]})
                else:
                    var_reduction_all[leaves].append(var_reduction)
                    rmse_all[leaves].append(rmse)

            plot_x_values  = []
            plot_y_values = []
            for leaf_node in range(1, 31, 1):
                plot_x_values.append(leaf_node)
                if plotting_target == 'var_reduction':
                    plot_y_values.append(var_reduction_all[leaf_node])
                elif plotting_target == 'rmse':
                    plot_y_values.append(rmse_all[leaf_node])
                assert len(plot_y_values[-1]) == len(action_ids)

            plot_y_values = np.asarray(plot_y_values).mean(axis=1)
            plot_x_values_all.append(plot_x_values)
            plot_y_values_all.append(plot_y_values)

    plot_values_by_node(plot_x_values_all, plot_y_values_all,
                        value_type=plotting_target,
                        plotting_target = game_name,
                        methods=methods)




def run_generate_values():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    game_name = 'SpaceInvaders-v0'
    method = 'mcts'
    action_id = 4
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
            'flappybird':['max_node', None, 'cpuct', 0.1, 'play', 200],
            'SpaceInvaders-v0': ['max_node', None, 'cpuct', 0.001, 'play', 200],
        }
    elif method == 'cart-fvae':
        options_dict = {
            'flappybird': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 20],
            'SpaceInvaders-v0': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 20],
        }
    elif method == 'cart':
        options_dict = {
            'flappybird': ['max_leaf_nodes', None, 'criterion', 'mae', 'random', 'min_samples_leaf', 1],
        }
    else:
        raise ValueError("unknown model name {0}".format(method))
    options = options_dict[game_name]

    option_str = '-'.join([str(option) for option in options])

    training_results_saving_dir = '../results/plot_results/{0}/training-{0}-action{1}' \
                                  '-by-splits-results-{2}-{3}.txt'.format(game_name, action_id, method, option_str)
    training_results_writer = open(training_results_saving_dir, 'w')
    train_results_csv_writer = csv.writer(training_results_writer)

    testing_results_saving_dir = '../results/plot_results/{0}/testing-{0}-action{1}' \
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
    if method == 'mcts':
        mimic_learner.data_loader(episode_number=4, target="latent", action_id=action_id)
        mimic_learner.mimic_env.assign_data(mimic_learner.memory)
        saved_nodes_dir = mimic_learner.get_MCTS_nodes_dir(action_id)
        return_value_log_all, return_value_log_struct_all, return_value_var_reduction_all, \
        mae_all, rmse_all, leaves_number_all = mimic_learner.predict_mcts_by_splits(action_id, saved_nodes_dir)
    elif method == 'cart-fvae' or method == 'cart':
        target = "latent" if method == 'cart-fvae' else "raw"
        mimic_learner.data_loader(episode_number=4, target=target, action_id=action_id)
        mimic_learner.mimic_env.assign_data(mimic_learner.memory)
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

    j = 1 if method == 'mcts' else 0 # skip some redundant results

    for i in range(j, len(return_value_log_all)):
        train_results_csv_writer.writerow([round(return_value_log_all[i], 4),
                                     round(return_value_log_struct_all[i], 4),
                                     round(return_value_var_reduction_all[i], 4),
                                     round(mae_all[i], 4),
                                     round(rmse_all[i], 4),
                                     leaves_number_all[i]])

    mimic_learner.iteration_number = int(mimic_learner.episodic_sample_number * 45)
    test_results_csv_writer.writerow(['return_value_log', 'return_value_log_struct', 'return_value_var_reduction',
                                 'mae', 'rmse', 'leaves'])

    if method == 'mcts':
        mimic_learner.data_loader(episode_number=45.5, target="latent", action_id=action_id)
        mimic_learner.mimic_env.assign_data(mimic_learner.memory)
        saved_nodes_dir = mimic_learner.get_MCTS_nodes_dir(action_id)
        return_value_log_all, return_value_log_struct_all, return_value_var_reduction_all, \
        mae_all, rmse_all, leaves_number_all = mimic_learner.predict_mcts_by_splits(action_id, saved_nodes_dir)
    elif method == 'cart-fvae' or method == "cart":
        target = "latent" if method == 'cart-fvae' else "raw"
        mimic_learner.data_loader(episode_number=45.5, target=target, action_id=action_id)
        mimic_learner.mimic_env.assign_data(mimic_learner.memory)
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

    for i in range(j, len(return_value_log_all)):
        test_results_csv_writer.writerow([round(return_value_log_all[i], 4),
                                     round(return_value_log_struct_all[i], 4),
                                     round(return_value_var_reduction_all[i], 8),
                                     round(mae_all[i], 4),
                                     round(rmse_all[i], 4),
                                     leaves_number_all[i]])




if __name__ == "__main__":
    # run_plot()
    run_generate_values()
    exit(0)

