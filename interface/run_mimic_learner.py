import os
from config.flappy_bird_config import FlappyBirdConfig
from mimic_learner.learner import MimicLearner


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flappybird_config_path = "../environment_settings/" \
                             "flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    mimic_learner = MimicLearner(game_name='flappybird', config=flappybird_config)
    mimic_learner.train_mimic_model()


if __name__ == "__main__":
    run()