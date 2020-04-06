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
    flappybird_config_path = "../environment_settings/flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    local_test_flag = False
    if local_test_flag:
        flappybird_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
        flappybird_config.Mimic.Learn.episodic_sample_number = 49
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        global_model_data_path = "/home/functor/scratch/Galen/project-DRL-Interpreter"
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")
    deg_type = 'AAE'
    DEG = Disentanglement(config=flappybird_config, deg_type=deg_type,
                          global_model_data_path=global_model_data_path)
    if deg_type == 'FVAE':
        DEG.train_fave()
    elif deg_type == 'AAE':
        DEG.train_aae()
    else:
        raise ValueError('Unknown deg type {0}'.format(deg_type))

    # DEG.test(testing_output_dir='../data_disentanglement/output/flappybird/')


if __name__ == "__main__":
    run()
