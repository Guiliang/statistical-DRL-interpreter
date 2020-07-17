from sklearn.tree.tree import DecisionTreeRegressor
import numpy as np
import pickle

from data_reader_tmp import data_loader
from mimic_learner.mcts_learner.mimic_env import MimicEnv


def compute_regression_results(predictions, labels):
    ae_sum = []
    se_sum = []

    length = len(predictions)
    for index in range(0, length):
        ae_sum.append(abs(predictions[index] - labels[index]))
        se_sum.append((predictions[index] - labels[index]) ** 2)

    mae = float(sum(ae_sum)) / length
    rmse = (float(sum(se_sum)) / length) ** 0.5

    return mae, rmse

class CARTRegressionTree():
    def __init__(self, model_name, options=[]):
        self.model = None
        self.max_leaf_nodes = options[1]
        self.criterion = options[3]
        self.mode = options[4]

    def train_mimic(self, training_data, mimic_env, save_model_dir, log_file):
        self.model = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                                           criterion= self.criterion,
                                           splitter=self.mode)
        self.model.fit(training_data[0], training_data[1])
        # self.print_tree()
        leaves_number = (self.model.tree_.node_count+1)/2
        print("Leaves number is {0}".format(leaves_number))
        predict_dictionary = {}
        predictions = self.model.predict(training_data[0])
        for predict_index in range(len(predictions)):
            predict_value = predictions[predict_index]
            if predict_value in predict_dictionary.keys():
                predict_dictionary[predict_value].append(predict_index)
            else:
                predict_dictionary.update({predict_value:[predict_index]})

        return_value_log = mimic_env.get_return(state=list(predict_dictionary.values()))
        return_value_log_struct = mimic_env.get_return(state=list(predict_dictionary.values()), apply_structure_cost=True)
        return_value_var_reduction = mimic_env.get_return(state=list(predict_dictionary.values()), apply_variance_reduction=True)
        mae, rmse = compute_regression_results(predictions=predictions, labels=training_data[1])
        # print("Training return:{0} with mae:{1} and rmse:{2}".format(return_value, mae, rmse), file=log_file)

        with open(save_model_dir, 'wb') as f:
            pickle.dump(obj=self.model, file=f)

        return return_value_log, return_value_log_struct, \
               return_value_var_reduction, mae, rmse, leaves_number


    def test_mimic(self, testing_data, mimic_env, save_model_dir, log_file):
        with open(save_model_dir, 'rb') as f:
            self.model = pickle.load(file=f)

        leaves_number = (self.model.tree_.node_count + 1) / 2
        predict_dictionary = {}
        predictions = self.model.predict(testing_data[0])
        for predict_index in range(len(predictions)):
            predict_value = predictions[predict_index]
            if predict_value in predict_dictionary.keys():
                predict_dictionary[predict_value].append(predict_index)
            else:
                predict_dictionary.update({predict_value:[predict_index]})

        return_value_log = mimic_env.get_return(state=list(predict_dictionary.values()))
        return_value_log_struct = mimic_env.get_return(state=list(predict_dictionary.values()), apply_structure_cost=True)
        return_value_var_reduction = mimic_env.get_return(state=list(predict_dictionary.values()), apply_variance_reduction=True)

        mae, rmse = compute_regression_results(predictions=predictions, labels=testing_data[1])
        # print("Testing return:{0} with mae:{1} and rmse:{2}".format(return_value, mae, rmse), file=log_file)

        return return_value_log, return_value_log_struct, \
               return_value_var_reduction, mae, rmse, leaves_number

options_dict = { 'flappybird': ['max_leaf_nodes', 50, 'criterion', 'mae', 'best'],}
game_name = 'flappybird'
method = 'cart'
mimic_model = CARTRegressionTree(model_name=method, options=options_dict[game_name])
action_id = 0

memory = data_loader(episode_number=4, action_id=action_id, iteration_number=0)
mimic_env = MimicEnv()
mimic_env.assign_data(memory)
init_state, init_var_list = mimic_env.initial_state(action=action_id)
training_data = [[], []]
for data_index in init_state[0]:
    data_input = np.concatenate([memory[data_index][0]], axis=0)
    data_output = memory[data_index][4]
    training_data[0].append(data_input)
    training_data[1].append(data_output)
save_model_dir = 'cs/oschulte/DRL-interpreter-model/comparison/cart/' \
                                               '{0}/{1}-aid{2}-sklearn.model'.format(game_name,
                                                                                     method,
                                                                                     action_id)
return_value_log, return_value_log_struct, \
return_value_var_reduction, mae, rmse, leaves_number \
    = mimic_model.train_mimic(training_data=training_data,
                                   save_model_dir=save_model_dir,
                                   mimic_env=mimic_env,
                                   log_file=None)

iteration_number = 1000 * 45
memory = data_loader(episode_number=45.5, action_id=action_id, iteration_number=iteration_number)
mimic_env.assign_data(memory)
init_state, init_var_list = mimic_env.initial_state(action=action_id)
testing_data = [[], []]
for data_index in init_state[0]:
    data_input = memory[data_index][0]
    data_output = memory[data_index][4]
    testing_data[0].append(data_input)
    testing_data[1].append(data_output)
testing_data[0] = np.stack(testing_data[0], axis=0)

return_value_log, return_value_log_struct, \
return_value_var_reduction, mae, rmse, leaves_number \
    = mimic_model.test_mimic(testing_data=testing_data,
                                   save_model_dir=save_model_dir,
                                   mimic_env=mimic_env,
                                   log_file=None)
