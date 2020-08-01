import os
cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print (sys.path)
from config.mimic_config import DRLMimicConfig
from data_disentanglement.disentanglement import Disentanglement


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    game_name = 'Assault-v0'
    deg_type = 'FVAE'
    if game_name == 'Assault-v0':
        config_path = "../environment_settings/assault_v0_config.yaml"
    elif game_name == 'Breakout-v0':
        config_path = "../environment_settings/breakout_v0_config.yaml"
    elif game_name == 'SpaceInvaders-v0':
        config_path = "../environment_settings/space_invaders_v0_config.yaml"
    elif game_name == 'flappybird':
        config_path = "../environment_settings/flappybird_config.yaml"
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    deg_config = DRLMimicConfig.load(config_path)
    local_test_flag = False
    if local_test_flag:
        deg_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
        deg_config.Mimic.Learn.episodic_sample_number = 49
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        global_model_data_path = "/home/functor/scratch/Galen/project-DRL-Interpreter"
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")
    DEG = Disentanglement(config=deg_config, deg_type=deg_type,
                          global_model_data_path=global_model_data_path)
    # if deg_type == 'FVAE':
    #     DEG.train_fave()
    # elif deg_type == 'AAE':
    #     DEG.train_aae()
    # else:
    #     raise ValueError('Unknown deg type {0}'.format(deg_type))

    if game_name == "flappybird":
        model_name = 'FVAE-1000000'
    elif game_name == "Assault-v0":
        # model_name = "FVAE-1000000"
        model_name = "FVAE-1000000-bak-4-21"
    elif game_name == "Breakout-v0":
        model_name = "FVAE-1000000"
    elif game_name == "SpaceInvaders-v0":
        model_name = "FVAE-1000000"
    else:
        raise ValueError ("Unknown game name {0}".format(game_name))
    DEG.test(model_name =model_name, testing_output_dir='../data_disentanglement/output/{0}/'.format(game_name))


if __name__ == "__main__":
    run()
