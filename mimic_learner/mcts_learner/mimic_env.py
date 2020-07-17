from copy import deepcopy
import math
import numpy as np
# from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

from mimic_learner.mcts_learner.static_env import StaticEnv
from utils.model_utils import tree_construct_loss

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class MimicEnv(StaticEnv):

    def __init__(self, n_action_types=None):
        self.n_action_types = n_action_types
        self.data_all = None
        self.initial_var = None

    def add_data(self, data):
        if self.data_all is None:
            self.data_all = data
        else:
            for transition in data:
                self.data_all.append(transition)

    def assign_data(self, data):
        self.data_all = data
        # delta_all = []
        # for data_line in self.data_all:
        #     delta_all.append(data_line[-1])
        # mu, std = norm.fit(delta_all)
        # self.initial_std = std

    def reset(self):
        pass
        # self.pos = (6, 0)
        # self.step_idx = 0
        # state = self.pos[0] * self.shape[0] + self.pos[1]
        # return state, 0, False, None

    def step(self, action):
        pass
        # self.step_idx += 1
        # alt_before = self.altitudes[self.pos[0]][self.pos[1]]
        # if action == UP:
        #     self.pos = (self.pos[0] - 1, self.pos[1])
        # if action == DOWN:
        #     self.pos = (self.pos[0] + 1, self.pos[1])
        # if action == LEFT:
        #     self.pos = (self.pos[0], self.pos[1] - 1)
        # if action == RIGHT:
        #     self.pos = (self.pos[0], self.pos[1] + 1)
        # self.pos = self._limit_coordinates(self.pos, self.shape)
        # alt_after = self.altitudes[self.pos[0]][self.pos[1]]
        # reward = alt_after - alt_before - 0.5  # -0.5 for encouraging speed
        # state = self.pos[0] * self.shape[0] + self.pos[1]
        # done = self.pos == (0, 6) or self.step_idx == self.ep_length
        # return state, reward, done, None

    def next_state(self, state, action, parent_var_list):
        state = deepcopy(state)
        action = deepcopy(action)
        parent_var_list = deepcopy(parent_var_list)
        action_values = action.split('_')
        subset_index = int(action_values[0])
        dim = int(action_values[1])
        split_value = float(action_values[2])
        # print('spitting rule is {0}'.format(action))
        subset_state1 = []
        subset_delta1 = []
        subset_state2 = []
        subset_delta2 = []
        for data_index in state[subset_index]:
            data_line = self.data_all[data_index]
            # if dim < float(self.n_action_types) / 2:
            if data_line[0][dim] < split_value:
                subset_state1.append(data_index)
                subset_delta1.append(self.data_all[data_index][-1])
            else:
                subset_state2.append(data_index)
                subset_delta2.append(self.data_all[data_index][-1])
            # else:
            #     if data_line[3][dim - int(self.n_action_types / 2)] < split_value:
            #         subset_state1.append(data_index)
            #         subset_delta1.append(self.data_all[data_index][-1])
            #     else:
            #         subset_state2.append(data_index)
            #         subset_delta2.append(self.data_all[data_index][-1])
        del state[subset_index]
        state.insert(subset_index, subset_state2)
        state.insert(subset_index, subset_state1)
        if len(subset_delta1) > 0:
            var1 = np.var(subset_delta1)
            # _, std1 = norm.fit(subset_delta1)
        else:
            var1 = 0
        if len(subset_state2) > 0:
            var2 = np.var(subset_delta2)
            # _, std2 = norm.fit(subset_delta2)
        else:
            var2 = 0
        # if parent_var_list is not None:
        new_var_list = parent_var_list
        del new_var_list[subset_index]
        new_var_list.insert(subset_index, var2)
        new_var_list.insert(subset_index, var1)
        return state, new_var_list

    @staticmethod
    def is_done_state(state, step_idx):
        # return np.unravel_index(state, shape) == (0, 6) or step_idx >= 15
        return False

    def initial_state(self, action=None):
        state_data = self.data_all
        state_index = []
        delta_all = []
        for i in range(len(state_data)):
            if action is not None:
                if action == state_data[i][1]:
                    state_index.append(i)
                    delta_all.append(state_data[i][-1])
            else:
                state_index.append(i)
                delta_all.append(state_data[i][-1])
        #     delta.append(state_data[i][-1])
        # _, std = norm.fit(delta_all)
        var = np.var(delta_all)
        self.initial_var = var
        return [state_index], [var]

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states)

    def regression_tree_rollout(self, state, is_training=False,max_node=100):
        subsection_lengths = [len(subsection) for subsection in state]
        total_length = float(sum(subsection_lengths))
        predict_dictionary = {}
        for subsection in state:
            sub_max_node = len(subsection)/total_length*(max_node-len(state)+1)

            training_data = []
            training_labels = []
            for i in range(len(subsection)):
                training_data.append(self.data_all[subsection[i]][0])
                training_labels.append(self.data_all[subsection[i]][-1])
            cart_model = DecisionTreeRegressor(max_depth=sub_max_node,
                                               criterion= 'mse',
                                               splitter='best')
            cart_model.fit(training_data, training_labels)

            predictions = cart_model.predict(training_data)
            for predict_index in range(len(predictions)):
                predict_value = predictions[predict_index]
                if predict_value in predict_dictionary.keys():
                    predict_dictionary[predict_value].append(predict_index)
                else:
                    predict_dictionary.update({predict_value:[predict_index]})

        return_value = self.get_return(state=list(predict_dictionary.values()), is_training=is_training)

        return return_value

    def get_return(self, state,
                   step_idx=None,
                   apply_structure_cost = False,
                   apply_variance_reduction = False,
                   is_training = False
                   ):
        var_weighted_sum = 0
        # log_var_weighted_sum = 0
        subsection_lengths = [len(subsection) for subsection in state]
        total_length = float(sum(subsection_lengths))
        ses_all = []
        for subsection in state:
            delta_all = []
            for i in range(len(subsection)):
                delta_all.append(self.data_all[subsection[i]][-1])
            if len(delta_all) > 0:
                # mu, std = norm.fit(delta_all)
                # var_tmp = std**2
                mean = np.mean(delta_all)
                for i in range(len(subsection)):
                    ses_all.append((mean-self.data_all[subsection[i]][-1])**2)
                var = np.var(delta_all)
                var_weighted_sum += (float(len(subsection)) / total_length) * var
                if var < 1e-6:
                    var = 1e-6
                # log_var_weighted_sum += (float(len(subsection)) / total_length) * math.log(var)
        # if self.initial_std - std_weighted_sum > 0.01306:
        #     print('testing')
        mse = sum(ses_all) / len(ses_all)

        if is_training:
            return self.initial_var - var_weighted_sum

        if apply_variance_reduction:
            return self.initial_var - var_weighted_sum

        log_var_weighted_sum = math.log(var_weighted_sum)
        if apply_structure_cost:
            structure_cost = 0
            leaf_number = float(len(state))
            if leaf_number > 1:
                structure_cost = tree_construct_loss(leaf_number)
            return -log_var_weighted_sum - 0.05*structure_cost
            # return math.log(self.initial_var) -log_var_weighted_sum-structure_cost
        else:
            return -log_var_weighted_sum
            # return math.log(self.initial_var) - log_var_weighted_sum
        # TODO: punish the split number



if __name__ == '__main__':
    env = MimicEnv()
