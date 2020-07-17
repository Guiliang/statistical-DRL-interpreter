import numpy as np
from PIL import Image

def data_loader(episode_number, action_id,
                game_name = 'flappybird',
                data_save_dir='/cs/oschulte/DRL-interpreter-model/data',
                iteration_number=0):
    memory = []
    action_number = 2
    image_type = 'origin'

    def gather_data_values(action_value):
        action_value_items = action_value.split(',')
        action_index = int(action_value_items[0])
        action_values_list = np.zeros([action_number])
        value = 0
        if game_name == 'flappybird':
            for i in range(action_number):
                action_values_list[i] = float(action_value_items[i + 1])
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
                                                                             image_type,
                                                                             iteration_number,
                                                                             action_index_t0))


    flatten_image_t0 = np.array(image).flatten()
    data_length = 1000 * episode_number - iteration_number
    while len(memory) < data_length:
        [action_index_t1, action_values_list_t1,
         reward_t1, value_t1] = gather_data_values(action_values[iteration_number + 1])
        if game_name == 'flappybird':
            delta = max(action_values_list_t1) - action_values_list_t0[action_index_t0] + reward_t0
        else:
            raise ValueError('Unknown game {0}'.format(game_name))

        image = Image.open('{0}/{1}/{2}/images/{1}-{3}_action{4}_{2}.png'.format(data_save_dir,
                                                                                 game_name,
                                                                                 image_type,
                                                                                 iteration_number + 1,
                                                                                 action_index_t1))

        flatten_image_t1 = np.array(image).flatten()
        if action_index_t0 == action_id:
            memory.append([flatten_image_t0, action_index_t0, reward_t0, flatten_image_t1, delta])
        flatten_image_t0 = flatten_image_t1

        iteration_number += 1
        action_index_t0 = action_index_t1
        action_values_list_t0 = action_values_list_t1
        reward_t0 = reward_t1
        value_t0 = value_t1

    print('loading finished')

    return memory

for aid in range(2):
    training_data_action = data_loader(episode_number=4, action_id=aid,
                                       iteration_number=0)
    iteration_number = 1000 * 45
    testing_data_action = data_loader(episode_number = 45.5, action_id=aid,
                                       iteration_number=iteration_number)

