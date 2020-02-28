from mimic_learner.mcts_learner.mcts import execute_episode
from mimic_learner.mcts_learner.mimic_env import MimicEnv
from data_disentanglement.disentanglement import Disentanglement


class MimicLearner():
    def __init__(self, game_name, config):
        self.mimic_env = MimicEnv
        self.game_name = game_name
        self.num_simulations = config.Mimic.Learn.num_simulations
        self.dientangler = Disentanglement(config)
        self.dientangler.load_checkpoint()
        self.data_save_dir = config.DEG.FVAE.data_save_path

    def data_loader(self):
        with open(self.data_save_dir, 'w') as f:
            action_values = f.readlines()

    def train_mimic_model(self):
        execute_episode(agent_netw=None, num_simulations=self.num_simulations, TreeEnv=self.mimic_env)
