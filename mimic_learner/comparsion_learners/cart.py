from sklearn.tree.tree import DecisionTreeRegressor
import numpy as np

from utils.plot_utils import plot_decision_boundary


class CARTRegressionTree():
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=100, criterion='friedman_mse')

    def train_mimic(self, training_data, mimic_env):
        self.model.fit(training_data[0], training_data[1])
        # self.print_tree()

        predict_dictionary = {}
        predicts = self.model.predict(training_data[0])
        for predict_index in range(len(predicts)):
            predict_value = predicts[predict_index]
            if predict_value in predict_dictionary.keys():
                predict_dictionary[predict_value].append(predict_index)
            else:
                predict_dictionary.update({predict_value:[predict_index]})

        return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
        print(return_value)

    def test_mimic(self, testing_data, mimic_env):
        predict_dictionary = {}
        predicts = self.model.predict(testing_data[0])
        for predict_index in range(len(predicts)):
            predict_value = predicts[predict_index]
            if predict_value in predict_dictionary.keys():
                predict_dictionary[predict_value].append(predict_index)
            else:
                predict_dictionary.update({predict_value:[predict_index]})

        return_value = mimic_env.get_return(state=list(predict_dictionary.values()))
        print(return_value)

    def train_2d_tree(self, training_data, selected_dim = (4, 6)):

        training_data = np.stack([np.asarray(training_data[0])[:, selected_dim[0]],
                                        np.asarray(training_data[0])[:, selected_dim[1]]], axis=1)
        data_number = 150
        self.model.fit(training_data[:data_number], training_data[1][:data_number])
        plot_decision_boundary(input_data=training_data[:data_number],
                               target_data = training_data[1][:data_number], tree_model=self.model)



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


