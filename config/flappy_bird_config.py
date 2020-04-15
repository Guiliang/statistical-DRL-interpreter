import yaml
from utils.config_utils import InitWithDict


class FlappyBirdConfig(object):
    def __init__(self, init):
        self.DEG = FlappyBirdConfig.DEG(init["DEG"])
        self.DRL = FlappyBirdConfig.DRL(init["DRL"])
        self.Mimic = FlappyBirdConfig.Mimic(init["Mimic"])

    class DEG(InitWithDict):

        def __init__(self, init):
            super(FlappyBirdConfig.DEG, self).__init__(init)
            self.AAE = FlappyBirdConfig.DEG.AAE(init["AAE"])
            self.Learn = FlappyBirdConfig.DEG.Learn(init["Learn"])
            self.FVAE = FlappyBirdConfig.DEG.FVAE(init["FVAE"])

        class Learn(InitWithDict):
            batch_size = None
            name = None
            cuda = None
            max_iter = None
            print_iter = None
            z_dim = None
            image_length = None
            image_width = None
            image_type = None
            num_workers = None
            dset_dir = None


        class AAE(InitWithDict):
            lr_D = None
            beta1_D = None
            beta2_D = None
            lr_G = None
            beta1_G = None
            beta2_G = None
            lr_E = None
            beta1_E = None
            beta2_E = None
            ckpt_save_iter = None


        class FVAE(InitWithDict):
            gamma = None
            lr_VAE = None
            beta1_VAE = None
            beta2_VAE = None
            lr_D = None
            beta1_D = None
            beta2_D = None
            ckpt_save_iter = None
            output_save = None
            gmma = None
            ckpt_load = None

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
            viz_ta_iter = None
            action_type = None

    class Mimic(InitWithDict):

        def __init__(self, init):
            super(FlappyBirdConfig.Mimic, self).__init__(init)
            self.Learn = FlappyBirdConfig.Mimic.Learn(init["Learn"])

        class Learn(InitWithDict):
            num_simulations = None
            replay_memory_size = None
            episodic_sample_number = None
            max_k = None

    @staticmethod
    def load(file_path):
        config = yaml.load(open(file_path, 'r'))
        return FlappyBirdConfig(config)
