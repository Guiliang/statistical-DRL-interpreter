import os
import optparse
import traceback

cwd = os.getcwd()
import sys
sys.path.append(cwd.replace('/interface', ''))
print (sys.path)
from config.mimic_config import DRLMimicConfig
from mimic_learner.learner import MimicLearner

optparser = optparse.OptionParser()
optparser.add_option("-r", "--round number", dest="ROUND_NUMBER", default=None,
                     help="round number of mcts (default = None)")
optparser.add_option("-a", "--action id", dest="ACTION_ID", default="0",
                     help="the action id to fit (default = 0)")
optparser.add_option("-d", "--log dir", dest="LOG_DIR", default=None,
                     help="the dir of log")
optparser.add_option("-g", "--game name", dest="GAME_NAME", default=None,
                     help="the name of running game")
optparser.add_option("-m", "--method name", dest="METHOD_NAME", default=None,
                     help="the name of applied method")
optparser.add_option("-l", "--launch time", dest="LAUNCH_TIME", default=None,
                     help="the time we launch this program")
# optparser.add_option("-d", "--dir of just saved mcts", dest="MCTS_DIR", default=None,
#                      help="dir of just saved mcts (default = None)")
opts = optparser.parse_args()[0]

def run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if opts.GAME_NAME is not None:
        game_name = opts.GAME_NAME
    else:
        game_name = 'flappybird'

    if opts.METHOD_NAME is not None:
        method = opts.METHOD_NAME
    else:
        method = 'mcts'

    if game_name == 'flappybird':
        model_name = 'FVAE-1000000'
    elif game_name == 'Assault-v0':
        model_name = 'FVAE-1000000'
    elif game_name == 'Breakout-v0':
        model_name = 'FVAE-1000000'
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    local_test_flag = False
    if local_test_flag:
        mimic_config_path = "../environment_settings/{0}_config.yaml".format(game_name)
        mimic_config = DRLMimicConfig.load(mimic_config_path)
        mimic_config.DEG.FVAE.dset_dir = '../example_data'
        global_model_data_path = ''
        mimic_config.Mimic.Learn.episodic_sample_number = 49
    elif os.path.exists("/Local-Scratch/oschulte/Galen"):
        mimic_config_path = "../environment_settings/{0}_config.yaml".format(game_name)
        mimic_config = DRLMimicConfig.load(mimic_config_path)
        global_model_data_path = "/Local-Scratch/oschulte/Galen"
    elif os.path.exists("/home/functor/scratch/Galen/project-DRL-Interpreter"):
        mimic_config_path = "/home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/" \
                                 "environment_settings/{0}_config.yaml".format(game_name)
        mimic_config = DRLMimicConfig.load(mimic_config_path)
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
        print("\nRunning for game {0} with {1}".format(game_name, method), file=log_file)
        mimic_learner = MimicLearner(game_name=game_name,
                                     method = method,
                                     config=mimic_config,
                                     deg_model_name = model_name,
                                     local_test_flag=local_test_flag,
                                     global_model_data_path=global_model_data_path,
                                     log_file=log_file)
        # mimic_learner.test_mimic_model(action_id= int(opts.ACTION_ID), log_file=log_file)
        shell_round_number = int(opts.ROUND_NUMBER) if opts.ROUND_NUMBER is not None else None

        mimic_learner.train_mimic_model(action_id = int(opts.ACTION_ID),
                                        shell_round_number=shell_round_number,
                                        log_file=log_file,
                                        launch_time = opts.LAUNCH_TIME,
                                        data_type = 'latent',
                                        run_mcts=True)

        if log_file is not None:
            log_file.close()
    except Exception as e:
        traceback.print_exc(file=log_file)
        if log_file is not None:
            log_file.write(str(e))
            log_file.flush()
            log_file.close()
        # sys.stderr.write('finish shell round {0}'.format(shell_round_number))


if __name__ == "__main__":
    run()
    exit(0)

