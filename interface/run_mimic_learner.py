import os
from config.flappy_bird_config import FlappyBirdConfig
from mimic_learner.learner import MimicLearner


def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    flappybird_config_path = "../environment_settings/flappybird_config.yaml"
    flappybird_config = FlappyBirdConfig.load(flappybird_config_path)

    local_test_flag = False
    if local_test_flag:
        flappybird_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        global_model_data_path = "/home/functor/scratch/Galen/project-DRL-Interpreter"
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")


    mimic_learner = MimicLearner(game_name='flappybird',
                                 config=flappybird_config,
                                 local_test_flag=local_test_flag,
                                 global_model_data_path=global_model_data_path)
    # mimic_learner.test_mimic_model()
    mimic_learner.train_mimic_model()


if __name__ == "__main__":
    run()
