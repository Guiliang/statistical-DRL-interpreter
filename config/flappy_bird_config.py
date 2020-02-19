import yaml
from utils.config_utils import InitWithDict


class FlappyBirdCongfig(object):
    def __init__(self, init):
        self.DRL = FlappyBirdCongfig.DRL(init["DRL"])

    class DRL(InitWithDict):
        game = None
        actions = None
        gamma = None
        observe = None
        explore = None
        final_epsilon = None
        initial_epsilon = None
        replay_memory = None
        batch = None
        frame_per_action = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return FlappyBirdCongfig(config)
