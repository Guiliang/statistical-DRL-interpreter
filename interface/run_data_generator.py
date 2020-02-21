import os
from config.flappy_bird_config import FlappyBirdConfig
from data_generator.generator import DRLDataGenerator


def run():
    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    data_generator = DRLDataGenerator(game_name='flappybird', config=flappybird_config)
    data_generator.train_DRl_model()


if __name__ == "__main__":
    run()
