import os
import traceback
import numpy as np

from interface.run_static_data_generator import run_static_data_generation

cwd = os.getcwd()
import sys

sys.path.append(cwd.replace('/interface', ''))
print(sys.path)
from config.mimic_config import DRLMimicConfig
from mimic_learner.learner import MimicLearner



def run(game_name=None, disentangler_name=None, run_tmp_test=False, method = None, iter_test_num=5):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if game_name is None:
        game_name = 'Assault-v0'

    if disentangler_name is None:
        disentangler_name = 'CVAE'

    if method is None:
        method = 'cart-fvae'

    if game_name == 'flappybird':
        action_ids = [0]
        model_name = '{0}-None'.format(disentangler_name)
        config_path = "../environment_settings/flappybird_config.yaml"
    elif game_name == 'SpaceInvaders-v0':
        action_ids = [4]
        model_name = '{0}-None'.format(disentangler_name)
        config_path = "../environment_settings/space_invaders_v0_config.yaml"
    elif game_name == 'Enduro-v0':
        action_ids = [7]  # 1: speed, 7 right, 8 left
        model_name = '{0}-None'.format(disentangler_name)
        config_path = "../environment_settings/enduro_v0_config.yaml"
    elif game_name == 'Assault-v0':
        action_ids = [4]  # 2: shot, 3 right, 4 left
        model_name = '{0}-None'.format(disentangler_name)
        config_path = '../environment_settings/assault_v0_config.yaml'
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    if method == 'mcts':
        options_dict = {
            'flappybird':['max_node', 30, 'cpuct', 0.1, 'play_number', 200],
            # 'Assault-v0':[]
        }
        action_ids = [0]
        data_type = 'latent'
    elif method == 'cart':
        disentangler_name = None
        options_dict = {
            'flappybird': ['max_leaf_nodes', None, 'criterion', 'mae', 'random', 'min_samples_leaf', 2],
            'Assault-v0': ['max_leaf_nodes', None, 'criterion', 'mae', 'random', 'min_samples_leaf', 4],
            'SpaceInvaders-v0': ['max_leaf_nodes', None, 'criterion', 'mae', 'best', 'min_samples_leaf',2],
            'Enduro-v0': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 1],
        }
        data_type = 'binary'
        # data_type = 'raw'
    elif method == 'cart-fvae':
        options_dict = {
            'flappybird': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 15],
            'Assault-v0': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 14],
            'SpaceInvaders-v0': ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 6],
            # 'Assault-v0': ['max_leaf_nodes', 80, 'criterion', 'mse', 'best', 'min_samples_leaf', 1],
        }
        data_type = 'latent'
    elif method == 'm5-rt':  # m5 regression tree
        disentangler_name = None
        options_dict = {
            'flappybird': ["-R", "-N", "-M", "10"],
            # 'Assault-v0': ["-R", "-N", "-M", "20"],
            'SpaceInvaders-v0': ["-R", "-N", "-M", "5"],
        }
        data_type = 'color'
        # options = ["-R"]
    elif method == 'm5-mt':  # m5 model tree
        # options = ["-M", "10"]
        disentangler_name = None
        options_dict = {
            'flappybird':["-N", "-M", "10"],
            # 'Assault-v0':["-N", "-M", "25"],
            'SpaceInvaders-v0': ["-N", "-M", "5"],
        }
        data_type = 'color'
    else:
        raise ValueError("unknown model name {0}".format(method))
    options = options_dict[game_name]

    option_str = '-'.join([str(option) for option in options])
    results_saving_dir = '../results/comparison_results/{0}/{0}-results-{1}-{2}-{3}.txt'.format(game_name,
                                                                                                method,
                                                                                                option_str,
                                                                                                disentangler_name)
    results_writer = open(results_saving_dir, 'w')

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


    train_record_results = {'return_value_log':[], 'return_value_log_struct':[], 'return_value_var_reduction':[],
                            'mae':[], 'rmse':[], 'leaves_number':[], 'results_strs':[]}

    testing_record_results_all = []
    for test_id in range(iter_test_num):
        testing_record_results = {'return_value_log':[], 'return_value_log_struct':[], 'return_value_var_reduction':[],
                                  'mae': [], 'rmse': [], 'leaves_number': [], 'results_strs':[]}
        testing_record_results_all.append(testing_record_results)

    # try:
    print("\nRunning for game {0} with {1}".format(game_name, method), file=log_file)
    mimic_learner = MimicLearner(game_name=game_name,
                                 method=method,
                                 config=mimic_config,
                                 deg_model_name=model_name,
                                 local_test_flag=local_test_flag,
                                 global_model_data_path=global_model_data_path,
                                 log_file=log_file,
                                 options=options)

    for action_id in action_ids:
    # for action_id in [1]:
        mimic_learner.iteration_number = 0
        [return_value_log, return_value_log_struct,
         return_value_var_reduction, mae, rmse,
         leaves_number, results_str] = mimic_learner.train_mimic_model(action_id=action_id,
                                                                       shell_round_number=None,
                                                                       log_file=log_file,
                                                                       launch_time=None,
                                                                       data_type=data_type,
                                                                       run_mcts=False,
                                                                       disentangler_name=disentangler_name,
                                                                       run_tmp_test=run_tmp_test
                                                                       )
        train_record_results['return_value_log'].append(return_value_log)
        train_record_results['return_value_log_struct'].append(return_value_log_struct)
        train_record_results['return_value_var_reduction'].append(return_value_var_reduction)
        train_record_results['mae'].append(mae)
        train_record_results['rmse'].append(rmse)
        train_record_results['leaves_number'].append(leaves_number)
        train_record_results['results_strs'].append(results_str)


        for test_id in range(iter_test_num):
            [return_value_log, return_value_log_struct,
             return_value_var_reduction, mae, rmse,
             leaves_number, results_str] = mimic_learner.test_mimic_model(action_id= action_id,
                                                                          log_file=log_file,
                                                                          data_type=data_type,
                                                                          disentangler_name=disentangler_name,
                                                                          run_tmp_test=run_tmp_test,
                                                                          test_id=test_id)
            testing_record_results_all[test_id]['return_value_log'].append(return_value_log)
            testing_record_results_all[test_id]['return_value_log_struct'].append(return_value_log_struct)
            testing_record_results_all[test_id]['return_value_var_reduction'].append(return_value_var_reduction)
            testing_record_results_all[test_id]['mae'].append(mae)
            testing_record_results_all[test_id]['rmse'].append(rmse)
            testing_record_results_all[test_id]['leaves_number'].append(leaves_number)
            testing_record_results_all[test_id]['results_strs'].append(results_str)
    # except Exception as e:
    #     traceback.print_exc(file=log_file)
    #     results_writer.close()
    #     if log_file is not None:
    #         log_file.write(str(e))
    #         log_file.flush()
    #         log_file.close()
    #         # sys.stderr.write('finish shell round {0}'.format(shell_round_number))

    for results_str in train_record_results['results_strs']:
        results_writer.write(results_str+'\n')

    mean_train_return_value_log= np.mean(train_record_results['return_value_log'])
    mean_train_return_value_log_struct = np.mean(train_record_results['return_value_log_struct'])
    mean_train_return_value_var_reduction= np.mean(train_record_results['return_value_var_reduction'])
    mean_train_mae = np.mean(train_record_results['mae'])
    mean_train_rmse = np.mean(train_record_results['rmse'])
    results_str = "Training method {0}: Avg.return_value_log:{1}, " \
                  "Avg.return_value_log_struct:{2}, Avg.return_value_var_reduction:{3}," \
                  "Avg.mae:{4}, Avg.rmse:{5}, Avg.leaves:{6}\n\n".format(method,
                                                                         str(mean_train_return_value_log)+ "({0})".format(
                                                                             float(mean_train_return_value_log) / leaves_number),
                                                                         str(mean_train_return_value_log_struct) + "({0})".format(
                                                                             float(mean_train_return_value_log_struct) / leaves_number),
                                                                         str(mean_train_return_value_var_reduction) + "({0})".format(
                                                                             float(mean_train_return_value_var_reduction) / leaves_number),
                                                                         str(mean_train_mae) + "({0})".format(
                                                                             float(mean_train_mae) / leaves_number),
                                                                         str(mean_train_rmse) + "({0})".format(
                                                                             float(mean_train_rmse) / leaves_number),
                                                                         np.mean(train_record_results['leaves_number']))
    results_writer.write(results_str)

    test_return_value_log_all = []
    test_return_value_log_struct_all = []
    test_return_value_var_reduction_all = []
    test_return_value_var_reduction_per_leaf_all = []
    test_mae_all = []
    test_rmse_all = []
    test_leaf_num_all = []
    for test_id in range(iter_test_num):
        testing_record_results = testing_record_results_all[test_id]
        for results_str in testing_record_results['results_strs']:
            results_writer.write(results_str+'iter{0}\n'.format(test_id))

        mean_test_return_value_log= np.mean(testing_record_results['return_value_log'])
        mean_test_return_value_log_struct = np.mean(testing_record_results['return_value_log_struct'])
        mean_test_return_value_var_reduction= np.mean(testing_record_results['return_value_var_reduction'])
        mean_test_mae = np.mean(testing_record_results['mae'])
        mean_test_rmse = np.mean(testing_record_results['rmse'])
        results_str = "Testing method {0} iter{7}: Avg.return_value_log:{1}, " \
                      "Avg.return_value_log_struct:{2}, Avg.return_value_var_reduction:{3}," \
                      "Avg.mae:{4}, Avg.rmse:{5}, Avg.leaves:{6}\n\n".format(method,
                                                                             str(mean_test_return_value_log)+ "({0})".format(
                                                                                 float(mean_test_return_value_log) / leaves_number),
                                                                             str(mean_test_return_value_log_struct) + "({0})".format(
                                                                                 float(mean_test_return_value_log_struct) / leaves_number),
                                                                             str(mean_test_return_value_var_reduction) + "({0})".format(
                                                                                 float(mean_test_return_value_var_reduction) / leaves_number),
                                                                             str(mean_test_mae) + "({0})".format(
                                                                                 float(mean_test_mae) / leaves_number),
                                                                             str(mean_test_rmse) + "({0})".format(
                                                                                 float(mean_test_rmse) / leaves_number),
                                                                             np.mean(testing_record_results['leaves_number']),
                                                                             test_id)
        results_writer.write(results_str)
        test_return_value_log_all.append(mean_test_return_value_log)
        test_return_value_log_struct_all.append(mean_test_return_value_log_struct)
        test_return_value_var_reduction_all.append(mean_test_return_value_var_reduction)
        test_return_value_var_reduction_per_leaf_all.append(mean_test_return_value_var_reduction/ leaves_number)
        test_mae_all.append(mean_test_mae)
        test_rmse_all.append(mean_test_rmse)
        test_leaf_num_all.append(leaves_number)


    results_str = "Testing method {0}: Mean.var_reduction:{1}," \
                  "Mean.var_reduction_per_leaf: {2}, " \
                  "Mean.mae:{3}, Mean.rmse:{4}, Mean.leaves:{5}\n\n".format(method,
                                                                          "{0}({1})".format(np.mean(test_return_value_var_reduction_all),
                                                                                            np.var(test_return_value_var_reduction_all)),
                                                                          "{0}({1})".format(np.mean(test_return_value_var_reduction_per_leaf_all),
                                                                                            np.var(test_return_value_var_reduction_per_leaf_all)),
                                                                          "{0}({1})".format(np.mean(test_mae_all),
                                                                                              np.var(test_mae_all)),
                                                                          "{0}({1})".format(np.mean(test_rmse_all),
                                                                                              np.var(test_rmse_all)),
                                                                          "{0}({1})".format(np.mean(test_leaf_num_all),
                                                                                              np.var(test_leaf_num_all)),
                                                                            )
    results_writer.write(results_str)
    print(results_str, file=log_file)

    results_writer.close()
    if 'mcts' not in method:
        mimic_learner.mimic_model.__del__()

    if log_file is not None:
        log_file.close()



def run_examine_data():
    # run()
    game_name = 'Assault-v0'
    disentangler_type = 'CVAE'
    for model_number in range(0, 1010000, 10000):
        # model_number = 1980000
        run_static_data_generation(model_number = model_number,
                                   game_name = game_name,
                                   disentangler_type = disentangler_type,
                                   image_type='latent',
                                   run_tmp_test=True)
        run(game_name=game_name,
            disentangler_name=disentangler_type,
            run_tmp_test=True,
            method='cart-fvae')

if __name__ == "__main__":
    # run_examine_data()
    run()
    exit(0)

