import numpy as np
import numpy as np
from scipy.stats import norm
from mimic_learner.mcts_learner.static_env import StaticEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class MimicEnv(StaticEnv):

    def __init__(self, n_action_types=None, data_all=None):
        self.n_action_types = n_action_types
        self.data_all = data_all

    def add_data(self, data):
        if self.data_all is None:
            self.data_all = data
        else:
            for transition in data:
                self.data_all.append(transition)

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

    @staticmethod
    def next_state(state, action, shape=(7, 7)):
        pos = np.unravel_index(state, shape)
        if action == UP:
            pos = (pos[0] - 1, pos[1])
        if action == DOWN:
            pos = (pos[0] + 1, pos[1])
        if action == LEFT:
            pos = (pos[0], pos[1] - 1)
        if action == RIGHT:
            pos = (pos[0], pos[1] + 1)
        pos = MimicEnv._limit_coordinates(pos, shape)
        return pos[0] * shape[0] + pos[1]

    @staticmethod
    def is_done_state(state, step_idx):
        # return np.unravel_index(state, shape) == (0, 6) or step_idx >= 15
        return False

    @staticmethod
    def initial_state(state_data=None):
        state_index = []
        for i in range(len(state_data)):
            state_index.append(i)
        #     delta.append(state_data[i][-1])
        # mu, std = norm.fit(delta)
        return [state_index]

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
            mu, std = norm.fit(delta_all)
            std_weighted_sum += float(len(subsection))/total_length * std
        return -std_weighted_sum

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
