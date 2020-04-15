import os
import optparse
import traceback

cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print (sys.path)
from config.flappy_bird_config import FlappyBirdConfig
from mimic_learner.learner import MimicLearner

optparser = optparse.OptionParser()
optparser.add_option("-r", "--round number", dest="ROUND_NUMBER", default=None,
                     help="round number of mcts (default = None)")
optparser.add_option("-a", "--action id", dest="ACTION_ID", default=0,
                     help="the action id to fit (default = 0)")
optparser.add_option("-d", "--log dir", dest="LOG_DIR", default=None,
                     help="the dir of log")
# optparser.add_option("-d", "--dir of just saved mcts", dest="MCTS_DIR", default=None,
#                      help="dir of just saved mcts (default = None)")
opts = optparser.parse_args()[0]

def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    local_test_flag = False
    if local_test_flag:
        flappybird_config_path = "../environment_settings/flappybird_config.yaml"
        flappybird_config = FlappyBirdConfig.load(flappybird_config_path)
        flappybird_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
        flappybird_config.Mimic.Learn.episodic_sample_number = 49
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        flappybird_config_path = "../environment_settings/flappybird_config.yaml"
        flappybird_config = FlappyBirdConfig.load(flappybird_config_path)
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        flappybird_config_path = "/home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/environment_settings/flappybird_config.yaml"
        flappybird_config = FlappyBirdConfig.load(flappybird_config_path)
        global_model_data_path = "/home/functor/scratch/Galen/project-DRL-Interpreter"
    else:
        raise EnvironmentError("Unknown running setting, please set up your own environment")
    
    print('global path is : {0}'.format(global_model_data_path))
    if opts.LOG_DIR is not None:
        if os.path.exists(opts.LOG_DIR):
            log_file =  open(opts.LOG_DIR, 'a')
        else:
            log_file = open(opts.LOG_DIR, 'w')
    else:
        log_file=None
    try:
        # raise EnvironmentError("testing")
        mimic_learner = MimicLearner(game_name='flappybird',
                                     config=flappybird_config,
                                     local_test_flag=local_test_flag,
                                     global_model_data_path=global_model_data_path,
                                     log_file=log_file)
        # mimic_learner.test_mimic_model()
        shell_round_number = int(opts.ROUND_NUMBER) if opts.ROUND_NUMBER is not None else None
        mimic_learner.train_mimic_model(method='mcts',
                                        action_id = opts.ACTION_ID,
                                        shell_round_number=shell_round_number,
                                        log_file=log_file
                                        )
        log_file.close()
    except Exception as e:
        traceback.print_exc(file=log_file)
        log_file.write(str(e))
        log_file.flush()
        log_file.close()
        # sys.stderr.write('finish shell round {0}'.format(shell_round_number))


if __name__ == "__main__":
    run()
    exit(0)

