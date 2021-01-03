import numpy as np
import torch
import random
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from config.mimic_config import DRLMimicConfig
from data_disentanglement.disentanglement import Disentanglement
from utils.general_utils import return_data


class flappybird_prob:
    """
    An MDP. Contains methods for initialisation, state transition.
    Can be aggregated or unaggregated.
    """

    def __init__(self, gamma=1, image_type=None):
        # assert games_directory is not None
        # self.games_directory = games_directory
        self.gamma = gamma
        self.reset = None
        self.isEpisodic = True

        # same action for all instances
        self.actions = ['the_action']

        self.dimNames = []
        self.dimSizes = []

        if image_type == 'origin':
            data_dim = 21168
        elif image_type == 'binary' or image_type == 'color':
            data_dim = 12288
        elif image_type == 'latent':
            data_dim = 10
        else:
            raise ValueError("Unknown image type {0}".format(image_type))

        for i in range(data_dim):
            if image_type == 'origin':
                self.dimNames.append('pixel_{}'.format(i))
            elif image_type == 'latent':
                self.dimNames.append('latent_{}'.format(i))
            self.dimSizes.append('continuous')

        self.stateFeatures = dict(zip(self.dimNames, self.dimSizes))
        self.nStates = len(self.stateFeatures)
        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        self.probName = ('{0}_gamma={1}_mode={2}').format(d, gamma,
                                                          'Action Feature States' if self.nStates > 12 else 'Feature States')
        # self.games_directory = games_directory
        return


def data_builder(episode_number, action_id,
                data_save_dir, dientangler,
                image_type,
                game_name,data_loader,
                action_number,
                iteration_number=0,
                disentangler_type='CVAE'):
    memory = []
    # action_number = 2
    # image_type = 'origin'

    def gather_data_values(action_value):
        action_value_items = action_value.split(',')
        action_index = int(action_value_items[0])
        action_values_list = np.zeros([action_number])
        value = 0
        if game_name == 'flappybird' or game_name == 'Enduro-v0':
            for i in range(action_number):
                action_values_list[i] = float(action_value_items[i + 1])
        elif game_name == 'Assault-v0' or game_name == 'SpaceInvaders-v0':
            value = float(action_value_items[1])
        else:
            raise ValueError('Unknown game {0}'.format(game_name))
        reward = float(action_value_items[-1])
        if reward > 1:
            reward = 1
        return action_index, action_values_list, reward, value

    with open(data_save_dir + '/' + game_name + '/action_values.txt', 'r') as f:
        action_values = f.readlines()

    [action_index_t0, action_values_list_t0,
     reward_t0, value_t0] = gather_data_values(action_values[iteration_number])
    # image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(data_save_dir,
    #                                                                          game_name,
    #                                                                          'origin',
    #                                                                          iteration_number,
    #                                                                          action_index_t0))
    # image = ttf.to_tensor(image).unsqueeze(0).to('cuda').cpu().numpy()
    x_t0 = data_loader.dataset.__getitem__(iteration_number)[0]
    # temp = x_t0.numpy()
    # temp = image - temp
    conds_t0 = data_loader.dataset.__getitem__(iteration_number)[1][:-1]
    cumu_reward_t0 = conds_t0[-1].item()
    if image_type == "latent":
        # x_t0_resized = image
        x_t0 = x_t0.to('cuda').unsqueeze(0)
        conds_t0 = conds_t0.to('cuda').unsqueeze(0)
        with torch.no_grad():
            # x_t0 = ttf.to_tensor(x_t0_resized).unsqueeze(0).to(dientangler.device)
            if disentangler_type == 'CVAE':
                random_encode_cat = torch.cat((dientangler.CVAE.state_encoder(x_t0).squeeze(-1).squeeze(-1),
                                               dientangler.CVAE.condition_encoder(conds_t0)), 1)
                z0 = dientangler.CVAE.conditional_q_nn(random_encode_cat)[:, :10].unsqueeze(-1).unsqueeze(-1)
            else:
                z0 = dientangler.VAE.encode(x_t0).squeeze()[:10]
            z0 = z0.squeeze().cpu().numpy()
    elif image_type == "origin" or image_type == "binary" or image_type == "color":
        flatten_image_t0 = np.array(x_t0).flatten()
    else:
        raise ValueError("Unknown data loader target {0}".format(image_type))

    data_length = 1000 * episode_number - iteration_number
    while len(memory) < data_length:
        [action_index_t1, action_values_list_t1,
         reward_t1, value_t1] = gather_data_values(action_values[iteration_number + 1])
        if game_name == 'flappybird' or game_name == 'Enduro-v0':
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0
        elif game_name == 'Assault-v0' or game_name == 'SpaceInvaders-v0':
            delta = value_t1 - value_t0 + reward_t0
        else:
            raise ValueError('Unknown game {0}'.format(game_name))

        # image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(data_save_dir,
        #                                                                          game_name,
        #                                                                          'origin',
        #                                                                          iteration_number + 1,
        #                                                                          action_index_t1))
        x_t1 = data_loader.dataset.__getitem__(iteration_number+1)[0]
        conds_t1 = data_loader.dataset.__getitem__(iteration_number+1)[1][:-1]
        cumu_reward_t1 = conds_t1[-1].item()
        temp = data_loader.dataset.__getitem__(iteration_number)[1][-1]
        assert abs(round(delta, 4) - round(temp.item(), 4)) < 0.001
            # print("debug")
            # print( max(action_values_list_t1) )
            # print(action_values_list_t0[action_index_t0])
            # print(reward_t0)
            # temp = data_loader.dataset.__getitem__(iteration_number)[1][-1]
        if image_type == "latent":
            # x_t1_resized = image
            x_t1 = x_t1.to('cuda').unsqueeze(0)
            conds_t1 = conds_t1.to('cuda').unsqueeze(0)
            with torch.no_grad():
                if disentangler_type == 'CVAE':
                    random_encode_cat = torch.cat((dientangler.CVAE.state_encoder(x_t1).squeeze(-1).squeeze(-1),
                                                   dientangler.CVAE.condition_encoder(conds_t1)), 1)
                    z1 = dientangler.CVAE.conditional_q_nn(random_encode_cat)[:, :10].unsqueeze(-1).unsqueeze(-1)
                else:
                    z1 = dientangler.VAE.encode(x_t1).squeeze()[:10]
                z1 = z1.squeeze().cpu().numpy()
            # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
            if action_index_t0 == action_id:
                memory.append([z0, action_index_t0, cumu_reward_t0, z1, delta])
                # print("Running builder for data{0} at iter{1}".format(len(memory), iteration_number))
            z0 = z1
        elif image_type == "origin" or image_type == "binary" or image_type == "color":
            flatten_image_t1 = np.array(x_t1).flatten()
            if action_index_t0 == action_id:
                memory.append([flatten_image_t0, action_index_t0, cumu_reward_t0, flatten_image_t1, delta])
                # print("Running builder for data{0} at iter{1}".format(len(memory), iteration_number))
            flatten_image_t0 = flatten_image_t1
        else:
            raise ValueError("Unknown data loader target {0}".format(image_type))
        action_index_t0 = action_index_t1
        action_values_list_t0 = action_values_list_t1
        cumu_reward_t0 = cumu_reward_t1
        value_t0 = value_t1
        iteration_number += 1
        reward_t0 = reward_t1

    print('loading finished')
    return memory


def write_header(writer, image_type):
    problem = flappybird_prob(image_type=image_type)

    headers = []
    headers.append('impact')
    headers = headers + problem.dimNames

    hearder_strings = ', '.join(headers)
    # [:-1] to remove last comma
    # hearder_strings = hearder_strings[:-1]

    writer.write(hearder_strings + ', action'+', cumu_reward'+'\n')


def write_data_text(data, writer):
    for i in range(len(data)):
        impact = str(data[i][-1])
        action = str(data[i][1])
        acumu_r = str(data[i][2])
        pixels = data[i][0]
        pixel_string = ', '.join(map(str, pixels))

        writer.write(impact.strip() +
                     ', ' + (pixel_string.strip()) +
                     ', ' + (action.strip()) +
                     ', ' + (acumu_r.strip()) +
                     '\n')


# do not run if called by another file


def run_static_data_generation(model_number=None, game_name=None,
                               disentangler_type = None,
                               image_type = None,
                               global_model_data_path = "/Local-Scratch/oschulte/Galen",
                               run_tmp_test=False, test_run=5):
    # game_name = 'flappybird'
    # image_type = 'latent'
    # disentangler_type = 'CVAE'
    # global_model_data_path = "/Local-Scratch/oschulte/Galen"

    if game_name is None:
        game_name = "Assault-v0"
    if disentangler_type is None:
        disentangler_type = None
    if image_type is None:
        image_type = 'binary'
    if model_number is None:
        model_number = None


    if run_tmp_test:
        tmp_msg = 'tmp_'
    else:
        tmp_msg = ''

    if game_name == 'flappybird':
        model_name = '{0}-{1}'.format(disentangler_type, model_number)
        config_game_name = 'flappybird'
        aids = [0]
    elif game_name == 'SpaceInvaders-v0':
        model_name = '{0}-{1}'.format(disentangler_type, model_number)
        config_game_name = "space_invaders_v0"
        aids = [4]
    elif game_name == 'Enduro-v0':
        model_name = '{0}-{1}'.format(disentangler_type, model_number)
        config_game_name = "enduro_v0"
        aids = [7]
    elif game_name == 'Assault-v0':
        model_name = '{0}-{1}'.format(disentangler_type, model_number)
        config_game_name = 'assault_v0'
        aids = [4]  # 2: shot, 3 right, 4 left
    elif game_name == 'Breakout-v0':
        model_name = '{0}-{1}'.format(disentangler_type, model_number)
        config_game_name = 'breakout_v0'
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    mimic_config_path = "../environment_settings/{0}_config.yaml".format(config_game_name)
    mimic_config = DRLMimicConfig.load(mimic_config_path)

    if image_type != 'latent':
        data_loader = return_data(mimic_config.DEG.Learn,
                                  global_model_data_path,
                                  mimic_config.DRL.Learn.actions,
                                  image_type=image_type)
        disentangler = None
    else:
        data_loader = return_data(mimic_config.DEG.Learn,
                                  global_model_data_path,
                                  mimic_config.DRL.Learn.actions,
                                  image_type='origin')
        disentangler = Disentanglement(mimic_config, disentangler_type, False, global_model_data_path)
        disentangler.load_checkpoint(ckptname= model_name, testing_flag=True, log_file=None)

    for aid in aids:
        data_save_dir = '/Local-Scratch/oschulte/Galen/DRL-interpreter-model/data'

        training_data_action = data_builder(episode_number=4, action_id=aid,
                                            data_save_dir=data_save_dir,
                                            dientangler=disentangler,
                                            image_type=image_type,
                                            game_name = game_name,
                                            iteration_number=0,
                                            disentangler_type=disentangler_type,
                                            data_loader=data_loader,
                                            action_number=mimic_config.DRL.Learn.actions)
        impact_file_name_training = '{5}impact_training_{4}_data_{1}_action_{2}.csv'.format(
            image_type, game_name, aid, disentangler_type, image_type, tmp_msg)
        impact_file_Writer_training = open('../LMUT_data/' + impact_file_name_training, 'w')

        print('Writing training csv for action {}...'.format(aid))
        write_header(impact_file_Writer_training, image_type=image_type)
        write_data_text(training_data_action, impact_file_Writer_training)
        impact_file_Writer_training.close()

        iteration_number = 1000 * 45
        testing_data_action = data_builder(episode_number=46, action_id=aid,
                                           data_save_dir=data_save_dir,
                                           dientangler=disentangler,
                                           image_type=image_type,
                                           game_name = game_name,
                                           iteration_number=iteration_number,
                                           disentangler_type=disentangler_type,
                                           data_loader=data_loader,
                                           action_number=mimic_config.DRL.Learn.actions)

        for i in range(test_run):
            testing_data_action_iter = testing_data_action[i*100:(i+5)*100]
            # testing_data_action_iter = random.sample(testing_data_action, 500)
            # for j in range(int(len(testing_data_action)/test_run)):
            #     testing_data_action_iter.append(testing_data_action[iter_test])
            #     iter_test += 1
            # create training and testing files
            impact_file_name_testing = '{5}impact_testing_{4}_data_{1}_action_{2}_iter{6}.csv'.format(
                image_type, game_name, aid, disentangler_type, image_type, tmp_msg, i)
            impact_file_Writer_testing = open('../LMUT_data/' + impact_file_name_testing, 'w')

            print('Writing testing csv for action {0} in iter {1}...'.format(aid, i))
            write_header(impact_file_Writer_testing, image_type=image_type)
            write_data_text(testing_data_action_iter, impact_file_Writer_testing)
            impact_file_Writer_testing.close()

if __name__ == '__main__':
    run_static_data_generation(run_tmp_test=False)
