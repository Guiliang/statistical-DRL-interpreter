import os
from config.flappy_bird_config import FlappyBirdConfig
from mimic_learner.learner import MimicLearner


def run():
    local_test_flag = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flappybird_config_path = "../environment_settings/" \
                             "flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    if local_test_flag:
        flappybird_config.DEG.FVAE.dset_dir = '../example_data'

    mimic_learner = MimicLearner(game_name='flappybird', config=flappybird_config, local_test_flag=local_test_flag)
    mimic_learner.train_mimic_model()


if __name__ == "__main__":
    run()
