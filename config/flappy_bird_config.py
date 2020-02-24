import yaml
from utils.config_utils import InitWithDict


class FlappyBirdConfig(object):
    def __init__(self, init):
        self.DRL = FlappyBirdConfig.DRL(init["DRL"])

    class DEG(InitWithDict):

        def __init__(self, init):
            super(FlappyBirdConfig.DEG, self).__init__(init)
            self.Learn = FlappyBirdConfig.DEG.FVAE(init["FVAE"])

        class FVAE(InitWithDict):
            name = None
            cuda = None
            max_iter = None
            batch_size = None
            z_dim = None
            gamma = None
            lr_VAE = None
            beta1_VAE = None
            beta2_VAE = None
            lr_D = None
            beta1_D = None
            beta2_D = None
            dset_dir = None
            image_size = None
            num_workers = None
            print_iter = None
            ckpt_save_iter = None
            output_save = None
            gmma = None

    class DRL(InitWithDict):

        def __init__(self, init):
            super(FlappyBirdConfig.DRL, self).__init__(init)
            self.Learn = FlappyBirdConfig.DRL.Learn(init["Learn"])

        class Learn(InitWithDict):
            game = None
            actions = None
            gamma = None
            observe = None
            explore = None
            final_epsilon = None
            initial_epsilon = None
            replay_memory_size = None
            batch = None
            frame_per_action = None
            cuda = None
            ckpt_dir = None
            ckpt_save_iter = None
            max_iter = None
            ckpt_load = None
            beta1_D = None
            beta2_D = None
            input_image_size = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return FlappyBirdConfig(config)