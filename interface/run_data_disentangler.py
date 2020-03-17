import os
cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print (sys.path)
from config.flappy_bird_config import FlappyBirdConfig
from data_disentanglement.disentanglement import Disentanglement


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flappybird_config_path = "/Local-Scratch/PycharmProjects/" \
                             "statistical-DRL-interpreter/environment_settings/" \
                             "flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    DEG = Disentanglement(config=flappybird_config)
    DEG.train()


if __name__ == "__main__":
    run()
