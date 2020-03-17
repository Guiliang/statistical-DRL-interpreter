from copy import deepcopy

import numpy as np
import numpy as np
from scipy.stats import norm
from mimic_learner.mcts_learner.static_env import StaticEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class MimicEnv(StaticEnv):

    def __init__(self, n_action_types=None):
        self.n_action_types = n_action_types
        self.data_all = None
        self.initial_std = None

    def add_data(self, data):
        if self.data_all is None:
            self.data_all = data
        else:
            for transition in data:
                self.data_all.append(transition)
        delta_all = []
        for data_line in self.data_all:
            delta_all.append(data_line[-1])
        mu, std = norm.fit(delta_all)
        self.initial_std = std

    def reset(self):
        self.pos = (6, 0)
        self.step_idx = 0
        state = self.pos[0] * self.shape[0] + self.pos[1]
        return state, 0, False, None

    def step(self, action):
        self.step_idx += 1
        alt_before = self.altitudes[self.pos[0]][self.pos[1]]
        if action == UP:
            self.pos = (self.pos[0] - 1, self.pos[1])
        if action == DOWN:
            self.pos = (self.pos[0] + 1, self.pos[1])
        if action == LEFT:
            self.pos = (self.pos[0], self.pos[1] - 1)
        if action == RIGHT:
            self.pos = (self.pos[0], self.pos[1] + 1)
        self.pos = self._limit_coordinates(self.pos, self.shape)
        alt_after = self.altitudes[self.pos[0]][self.pos[1]]
        reward = alt_after - alt_before - 0.5  # -0.5 for encouraging speed
        state = self.pos[0] * self.shape[0] + self.pos[1]
        done = self.pos == (0, 6) or self.step_idx == self.ep_length
        return state, reward, done, None

    def next_state(self, state, action, parent_var_list=None):
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
            if dim < float(self.n_action_types) / 2:
                if data_line[0][dim] < split_value:
                    subset_state1.append(data_index)
                    subset_delta1.append(self.data_all[data_index][-1])
                else:
                    subset_state2.append(data_index)
                    subset_delta2.append(self.data_all[data_index][-1])
            else:
                if data_line[3][dim - int(self.n_action_types / 2)] < split_value:
                    subset_state1.append(data_index)
                    subset_delta1.append(self.data_all[data_index][-1])
                else:
                    subset_state2.append(data_index)
                    subset_delta2.append(self.data_all[data_index][-1])
        del state[subset_index]
        state.insert(subset_index, subset_state2)
        state.insert(subset_index, subset_state1)
        if len(subset_delta1) > 0:
            _, var1 = norm.fit(subset_delta1)
        else:
            var1 = 0
        if len(subset_state2) > 0:
            _, var2 = norm.fit(subset_delta2)
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
        _, std = norm.fit(delta_all)
        return [state_index], [std]

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states)

    def get_return(self, state, step_idx):
        std_weighted_sum = 0
        subsection_lengths = [len(subsection) for subsection in state]
        total_length = float(sum(subsection_lengths))
        for subsection in state:
            delta_all = []
            for i in range(len(subsection)):
                delta_all.append(self.data_all[subsection[i]][-1])
            if len(delta_all) > 0:
                mu, std = norm.fit(delta_all)
                std_weighted_sum += float(len(subsection)) / total_length * std
        # if self.initial_std - std_weighted_sum > 0.01306:
        #     print('testing')
        # TODO: punish the split number
        return self.initial_std - std_weighted_sum

    @staticmethod
    def _limit_coordinates(coord, shape):
        """
        Prevent the agent from falling out of the grid world.
        Adapted from https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
        """
        coord = list(coord)
        coord[0] = min(coord[0], shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return tuple(coord)


if __name__ == '__main__':
    env = MimicEnv()
