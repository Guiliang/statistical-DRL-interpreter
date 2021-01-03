import numpy as np
import torch
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
from PIL import Image
from datetime import datetime

from config.mimic_config import DRLMimicConfig
from data_disentanglement.disentanglement import Disentanglement


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
        elif image_type == 'latent':
            data_dim = 10
        else:
            raise ValueError("Unknown image type {0}".format(image_type))

        for i in range(data_dim):
            self.dimNames.append('pixel_{}'.format(i))
            self.dimSizes.append('continuous')

        self.stateFeatures = dict(zip(self.dimNames, self.dimSizes))
        self.nStates = len(self.stateFeatures)
        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        self.probName = ('{0}_gamma={1}_mode={2}').format(d, gamma,
                                                          'Action Feature States' if self.nStates > 12 else 'Feature States')
        # self.games_directory = games_directory
        return


def data_loader(episode_number, action_id,
                data_save_dir, dientangler,
                image_type,
                game_name,
                iteration_number=0):
    memory = []
    action_number = 2
    # image_type = 'origin'

    def gather_data_values(action_value):
        action_value_items = action_value.split(',')
        action_index = int(action_value_items[0])
        action_values_list = np.zeros([action_number])
        value = 0
        if game_name == 'flappybird':
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
    image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(data_save_dir,
                                                                             game_name,
                                                                             'origin',
                                                                             iteration_number,
                                                                             action_index_t0))

    if image_type == "latent":
        x_t0_resized = image
        with torch.no_grad():
            x_t0 = ttf.to_tensor(x_t0_resized).unsqueeze(0).to(dientangler.device)
            z0 = dientangler.VAE.encode(x_t0).squeeze()[:10]
            z0 = z0.cpu().numpy()
    elif image_type == "origin":
        flatten_image_t0 = np.array(image).flatten()
    else:
        raise ValueError("Unknown data loader target {0}".format(image_type))

    data_length = 1000 * episode_number - iteration_number
    while len(memory) < data_length:
        [action_index_t1, action_values_list_t1,
         reward_t1, value_t1] = gather_data_values(action_values[iteration_number + 1])
        if game_name == 'flappybird':
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0
        elif game_name == 'Assault-v0' or game_name == 'SpaceInvaders-v0':
            delta = value_t1 - value_t0 + reward_t0
        else:
            raise ValueError('Unknown game {0}'.format(game_name))

        image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(data_save_dir,
                                                                                 game_name,
                                                                                 'origin',
                                                                                 iteration_number + 1,
                                                                                 action_index_t1))

        if image_type == "latent":
            x_t1_resized = image
            with torch.no_grad():
                x_t1 = ttf.to_tensor(x_t1_resized).unsqueeze(0).to(dientangler.device)
                z1 = dientangler.VAE.encode(x_t1).squeeze()[:10]
                z1 = z1.cpu().numpy()
            # self.memory.add(delta, (z0, action_index_t0, reward_t0, z1, delta))
            if action_index_t0 == action_id:
                memory.append([z0, action_index_t0, reward_t0, z1, delta])
                print(len(memory))
            z0 = z1
        elif image_type == "origin":
            flatten_image_t1 = np.array(image).flatten()
            if action_index_t0 == action_id:
                memory.append([flatten_image_t0, action_index_t0, reward_t0, flatten_image_t1, delta])
            flatten_image_t0 = flatten_image_t1
        else:
            raise ValueError("Unknown data loader target {0}".format(image_type))

        iteration_number += 1
        action_index_t0 = action_index_t1
        action_values_list_t0 = action_values_list_t1
        reward_t0 = reward_t1
        value_t0 = value_t1

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

    writer.write(hearder_strings + '\n')


def write_data_text(data, writer):
    for i in range(len(data)):
        impact = str(data[i][-1])
        pixels = data[i][0]
        pixel_string = ', '.join(map(str, pixels))

        writer.write(impact.strip() + ',' + (pixel_string.strip()) + '\n')


# do not run if called by another file
if __name__ == '__main__':
    game_name = 'flappybird'
    image_type = 'latent'
    global_model_data_path = "/Local-Scratch/oschulte/Galen"

    if game_name == 'flappybird':
        model_name = 'FVAE-1000000'
        config_game_name = 'flappybird'
    elif game_name == 'SpaceInvaders-v0':
        model_name = 'FVAE-1000000'
        config_game_name = "space_invaders_v0"
    elif game_name == 'Assault-v0':
        model_name = 'FVAE-1000000'
        config_game_name = 'assault_v0'
    elif game_name == 'Breakout-v0':
        model_name = 'FVAE-1000000'
        config_game_name = 'breakout_v0'
    else:
        raise ValueError("Unknown game name {0}".format(game_name))

    mimic_config_path = "./environment_settings/{0}_config.yaml".format(config_game_name)
    mimic_config = DRLMimicConfig.load(mimic_config_path)

    dientangler = Disentanglement(mimic_config, 'FVAE', False, global_model_data_path)
    dientangler.load_checkpoint(ckptname= model_name, testing_flag=True, log_file=None)

    for aid in [0]:
        data_save_dir = '/Local-Scratch/oschulte/Galen/DRL-interpreter-model/data'

        training_data_action = data_loader(episode_number=4, action_id=aid,
                                           data_save_dir=data_save_dir,
                                           dientangler=dientangler,
                                           image_type=image_type,
                                           game_name = game_name,
                                           iteration_number=0)


        iteration_number = 1000 * 45
        testing_data_action = data_loader(episode_number=45.5, action_id=aid,
                                          data_save_dir=data_save_dir,
                                          dientangler=dientangler,
                                          image_type=image_type,
                                          game_name = game_name,
                                          iteration_number=iteration_number)

        # create training and testing files
        impact_file_name_training = 'impact_training_data_{1}_action_{2}.csv'.format(image_type, game_name, aid)
        impact_file_Writer_training = open('./LMUT_data/' + impact_file_name_training, 'w')

        impact_file_name_testing = 'impact_testing_data_{1}_action_{2}.csv'.format(image_type, game_name, aid)
        impact_file_Writer_testing = open('./LMUT_data/' + impact_file_name_testing, 'w')

        print('Writing training csv for action {}...'.format(aid))
        write_header(impact_file_Writer_training, image_type=image_type)
        write_data_text(training_data_action, impact_file_Writer_training)
        impact_file_Writer_training.close()

        print('Writing testing csv for action {}...'.format(aid))
        write_header(impact_file_Writer_testing, image_type=image_type)
        write_data_text(testing_data_action, impact_file_Writer_testing)
        impact_file_Writer_testing.close()