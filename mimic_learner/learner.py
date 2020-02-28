from mimic_learner.mcts_learner.mcts import execute_episode
from mimic_learner.mcts_learner.mimic_env import MimicEnv


class MimicLearner():
    def __init__(self, game_name, config):
        self.mimic_env = MimicEnv
        self.game_name = game_name
        self.num_simulations = config.Mimic.Learn.num_simulations

    def train_mimic_model(self):
        execute_episode(agent_netw=None, num_simulations=self.num_simulations, TreeEnv=self.mimic_env)
