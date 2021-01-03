from sklearn.tree.tree import DecisionTreeRegressor
import numpy as np
import pickle
from utils.general_utils import compute_regression_results
from utils.plot_utils import plot_decision_boundary


class CARTRegressionTree():
    def __init__(self, model_name, options=[]):
        self.model = None
        self.max_leaf_nodes = options[1]
        self.criterion = options[3]
        self.mode = options[4]
        self.min_samples_leaf = options[6]
        # ['max_leaf_nodes', None, 'criterion', 'mse', 'best', 'min_samples_leaf', 10]

    def train_mimic(self, training_data, mimic_env, save_model_dir, log_file):
        self.model = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                                           criterion= self.criterion,
                                           splitter=self.mode,
                                           min_samples_leaf=self.min_samples_leaf,
                                           # max_features = 9
                                           )
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

    def train_2d_tree(self, training_data, selected_dim = (4, 6)):

        training_data = np.stack([np.asarray(training_data[0])[:, selected_dim[0]],
                                        np.asarray(training_data[0])[:, selected_dim[1]]], axis=1)
        data_number = 150
        self.model.fit(training_data[:data_number], training_data[1][:data_number])
        plot_decision_boundary(input_data=training_data[:data_number],
                               target_data = training_data[1][:data_number], tree_model=self.model)

    def __del__(self):
        pass

    def print_tree(self):
        n_nodes = self.model.tree_.node_count
        children_left = self.model.tree_.children_left
        children_right = self.model.tree_.children_right
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % n_nodes)
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (node_depth[i] * "\t",
                         i,
                         children_left[i],
                         feature[i],
                         threshold[i],
                         children_right[i],
                         ))


