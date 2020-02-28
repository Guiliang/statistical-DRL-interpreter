import os
from config.flappy_bird_config import FlappyBirdConfig
from mimic_learner.learner import MimicLearner


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    data_generator = MimicLearner(game_name='flappybird', config=flappybird_config)
    data_generator.test_model_and_generate_data()


if __name__ == "__main__":
    run()