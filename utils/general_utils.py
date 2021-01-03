import scipy.io as sio
import argparse
import subprocess
"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageConditionFolder(ImageFolder):
    """
    Read and return the training data.
    """
    def __init__(self, root, cond_dir, img_path, image_type, game, action_number, transform=None):
        super(CustomImageConditionFolder, self).__init__(root, transform)
        self.indices = range(len(self)-1)
        self.game_name = game
        self.action_number = action_number
        self.img_path = img_path
        self.img_type = image_type
        with open(cond_dir, 'r') as f:
            self.action_values = f.readlines()

        rewards = []
        for action_value in self.action_values:
            reward = float(action_value.split(',')[-1])
            rewards.append(reward)

        self.cumu_reward_all = calculate_cumulative_reward(reward_all=rewards)

    def __getitem__(self, index1):
        if index1 == len(self)-1:  # handle the last index
            index1 -= 1
        index2 = random.choice(self.indices)

        [action_index_t0_i1, action_values_list_t0_i1,
         reward_t0_i1, value_t0_i1] = gather_data_values(self.action_values[index1], self.action_number, self.game_name)
        [action_index_t1_i1, action_values_list_t1_i1,
        reward_t1_i1, value_t1_i1] = gather_data_values(self.action_values[index1 + 1], self.action_number, self.game_name)

        [action_index_t0_i2, action_values_list_t0_i2,
         reward_t0_i2, value_t0_i2] = gather_data_values(self.action_values[index2], self.action_number, self.game_name)
        [action_index_t1_i2, action_values_list_t1_i2,
         reward_t1_i2, value_t1_i2] = gather_data_values(self.action_values[index2 + 1], self.action_number, self.game_name)

        # for action in range(self.action_number):
        #     check_img_path = self.img_path+'/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index1, action, self.img_type)
        #     if os.path.isfile(check_img_path):
        #         path1 = check_img_path
        #         break
        #
        # for action in range(self.action_number):
        #     check_img_path = self.img_path+'/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index2, action, self.img_type)
        #     if os.path.isfile(check_img_path):
        #         path2 = check_img_path
        #         break
        path1 = self.img_path+'/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index1, action_index_t0_i1, self.img_type)
        path2 = self.img_path + '/images/{0}-{1}_action{2}_{3}.png'.format(self.game_name, index2, action_index_t0_i2, self.img_type)
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.game_name == 'flappybird' or self.game_name == 'Enduro-v0' or self.game_name == 'Enduro-v1':
            delta_i1 = max(action_values_list_t1_i1) - action_values_list_t0_i1[action_index_t0_i1] + reward_t0_i1
            delta_i2 = max(action_values_list_t1_i2) - action_values_list_t0_i2[action_index_t0_i2] + reward_t0_i2
        elif self.game_name == 'Assault-v0' or self.game_name == 'SpaceInvaders-v0':
            delta_i1 = value_t1_i1 - value_t0_i1 + reward_t0_i1
            delta_i2 = value_t1_i2 - value_t0_i2 + reward_t0_i2
        else:
            raise ValueError('Unknown game {0}'.format(self.game_name))
        action_t0_i1 = [0 for i in range(self.action_number)]
        action_t0_i1[action_index_t0_i1] = 1
        action_t0_i2 = [0 for i in range(self.action_number)]
        action_t0_i2[action_index_t0_i2] = 1
        cond_i1 = torch.tensor(action_t0_i1+[reward_t0_i1, delta_i1])
        cond_i2 = torch.tensor(action_t0_i2+[reward_t0_i2, delta_i2])

        return img1, cond_i1, img2, cond_i2


class CustomPlayFolder:
    def __init__(self, root, game, batch_size):
        self.indices = range(len(os.listdir(root)))
        self.data_store = root
        self.batch_size = batch_size
        self.game = game
        self.root = root


    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        state_trace_length_i1, state_input_i1, \
        reward_i1, action_i1, team_id_i1=get_icehockey_game_data(data_store = self.data_store,
                                                                 dir_game=os.listdir(self.root)[index1],
                                                                 batch_size=self.batch_size)

        state_trace_length_i2, state_input_i2, \
        reward_i2, action_i2, team_id_i2=get_icehockey_game_data(data_store = self.data_store,
                                                                 dir_game=os.listdir(self.root)[index2],
                                                                 batch_size=self.batch_size)

        # print('test')
        trace_length_indices_i1 = state_trace_length_i1-np.ones([self.batch_size], dtype=np.int32)
        state_i1 = np.asarray([state_input_i1[tl_index, trace_length_indices_i1[tl_index], :] for tl_index in range(len(trace_length_indices_i1))])

        trace_length_indices_i2 = state_trace_length_i2-np.ones([self.batch_size], dtype=np.int32)
        state_i2 = np.asarray([state_input_i2[tl_index, trace_length_indices_i2[tl_index], :] for tl_index in range(len(trace_length_indices_i2))])

        return state_i1, np.concatenate([action_i1, np.expand_dims(reward_i1, -1)], axis=1), \
               state_i2, np.concatenate([action_i2, np.expand_dims(reward_i2, -1)], axis=1)



class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(config, global_model_data_path, action_number, image_type=None):
    if image_type is None:
        image_type = config.image_type

    name = config.name
    dset_dir = global_model_data_path+config.dset_dir
    batch_size = config.batch_size
    num_workers = config.num_workers
    image_length = config.image_length
    image_width = config.image_width
    # assert image_size == 64, 'currently only image size of 64 is supported'
    transform = transforms.Compose([
        # transforms.Resize((image_length, image_width)),
        transforms.ToTensor(), ])

    if name.lower() == 'flappybird':
        root = os.path.join(dset_dir, 'flappybird/'+image_type)
        cond_dir = os.path.join(dset_dir, 'flappybird/action_values.txt')
        img_path = os.path.join(dset_dir, 'flappybird/'+image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type':image_type,
                        'game': 'flappybird', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'assault-v0':
        root = os.path.join(dset_dir, 'Assault-v0/'+image_type)
        cond_dir = os.path.join(dset_dir, 'Assault-v0/action_values.txt')
        img_path = os.path.join(dset_dir, 'Assault-v0/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type':image_type,
                        'game': 'Assault-v0', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'breakout-v0':
        root = os.path.join(dset_dir, 'Breakout-v0/'+image_type)
        cond_dir = os.path.join(dset_dir, 'Breakout-v0/action_values.txt')
        img_path = os.path.join(dset_dir, 'Breakout-v0/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type':image_type,
                        'game': 'Breakout-v0', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'spaceinvaders-v0':
        root = os.path.join(dset_dir, 'SpaceInvaders-v0/'+image_type)
        cond_dir = os.path.join(dset_dir, 'SpaceInvaders-v0/action_values.txt')
        img_path = os.path.join(dset_dir, 'SpaceInvaders-v0/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type':image_type,
                        'game': 'SpaceInvaders-v0', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'enduro-v0':
        root = os.path.join(dset_dir, 'Enduro-v0/'+image_type)
        cond_dir = os.path.join(dset_dir, 'Enduro-v0/action_values.txt')
        img_path = os.path.join(dset_dir, 'Enduro-v0/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path': img_path, 'image_type':image_type,
                        'game': 'Enduro-v0', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'enduro-v1':
        root = os.path.join(dset_dir, 'Enduro-v1/'+image_type)
        cond_dir = os.path.join(dset_dir, 'Enduro-v1/action_values.txt')
        img_path = os.path.join(dset_dir, 'Enduro-v1/' + image_type)
        train_kwargs = {'root': root, 'transform': transform, 'cond_dir': cond_dir,
                        'img_path':img_path, 'image_type':image_type,
                        'game': 'Enduro-v1', 'action_number': action_number}
        dset = CustomImageConditionFolder
    elif name.lower() == 'icehockey':
        root = dset_dir
        train_kwargs = {'root': root, 'game': name, 'batch_size': batch_size}
        dset = CustomPlayFolder
        batch_size = 1
    # elif name.lower() == '3dchairs':
    #     root = os.path.join(dset_dir, '3DChairs')
    #     train_kwargs = {'root':root, 'transform':transform}
    #     dset = CustomImageFolder
    # elif name.lower() == 'dsprites':
    #     root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    #     data = np.load(root, encoding='latin1')
    #     data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    #     train_kwargs = {'data_tensor':data}
    #     dset = CustomTensorDataset
    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg: [] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay ' + str(delay) + ' -loop 0 ' + image_str + ' ' + output_gif
    subprocess.call(str1, shell=True)


def handle_dict_list(dict_list_A, dict_list_B, option):
    for key in dict_list_B.keys():
        list_B = dict_list_B.get(key)
        if key in dict_list_A.keys():
            list_A = dict_list_A.get(key)
            if option == 'add':
                list_new = list(set(list_A + list_B))
            elif option == 'substract':
                list_new = list(set(list_A) - set(list_B))
            else:
                raise ValueError('unknown option {0}'.format(option))
            dict_list_A[key] = list_new
        else:
            if option == 'add':
                dict_list_A[key] = list_B
    return dict_list_A


def compute_regression_results(predictions, labels):
    ae_sum = []
    se_sum = []

    length = len(predictions)
    for index in range(0, length):
        ae_sum.append(abs(predictions[index] - labels[index]))
        se_sum.append((predictions[index] - labels[index]) ** 2)

    mae = float(sum(ae_sum)) / length
    rmse = (float(sum(se_sum)) / length) ** 0.5

    return mae, rmse

def count_actions():
    action_values_dir = '/Local-Scratch/oschulte/Galen/DRL-interpreter-model/data/SpaceInvaders-v0/'
    action_count = {}
    with open(action_values_dir + '/action_values.txt', 'r') as f:
        action_values = f.readlines()
    for action_value in action_values[:40000]:
        action_value_items = action_value.split(',')
        action_index = int(action_value_items[0])
        if action_count.get(action_index) is not None:
            action_count[action_index]+=1
        else:
            action_count.update({action_index:1})

    print(action_count)

def get_icehockey_game_data(data_store, dir_game, batch_size=32):
    """
    return the ice hockey data for training / testing 
    :param data_store: the stored data after processing
    :param dir_game: the index of the game
    :param batch_size: the batch data size
    :return: 
    """
    game_files = os.listdir(data_store + "/" + dir_game)
    reward_name = None
    state_input_name = None
    trace_length_name = None
    team_id_name = None
    action_id_name = None
    home_away_identifier_name = None


    for filename in game_files:
        if "reward" in filename:
            reward_name = filename
        elif "state_feature_seq" in filename:
            state_input_name = filename
        elif "lt_" in filename:
            trace_length_name = filename
        elif 'team' in filename:
            team_id_name = filename
        elif 'home_away' in filename:
            home_away_identifier_name = filename
        elif 'action' in filename:
            if 'action_feature_seq' in filename:
                continue
            action_id_name = filename

    assert home_away_identifier_name is not None
    home_away_identifier = sio.loadmat(data_store + "/" + dir_game + "/" + home_away_identifier_name)
    home_away_identifier = home_away_identifier['home_away'][0]
    assert reward_name is not None
    reward = sio.loadmat(data_store + "/" + dir_game + "/" + reward_name)
    reward = reward['reward'][0]
    assert team_id_name is not None
    team_id = sio.loadmat(data_store + "/" + dir_game + "/" + team_id_name)['team']
    assert state_input_name is not None
    state_input = sio.loadmat(data_store + "/" + dir_game + "/" + state_input_name)['state_feature_seq']
    assert action_id_name is not None
    action = sio.loadmat(data_store + "/" + dir_game + "/" + action_id_name)['action']
    assert trace_length_name is not None
    state_trace_length = sio.loadmat(data_store + "/" + dir_game + "/" + trace_length_name)['lt'][0]

    random_index = random.sample(range(len(state_trace_length)), batch_size) # to return just one data point

    cumulative_reward = calculate_cumulative_reward(reward, home_away_identifier)


    for tl_index in range(len(state_trace_length)):
        if state_trace_length[tl_index] > 10:
            state_trace_length[tl_index] = 10


    return state_trace_length[random_index], state_input[random_index], \
           cumulative_reward[random_index], action[random_index], team_id[random_index]


def calculate_cumulative_reward(reward_all, home_away_identifier=None):
    """
    calculate the condition R (cumulative reward)
    :param home_away_identifier: 0 away team / 1 home team 
    :param reward_all: 0 no score /1 score
    :return: the cumulative reward
    """
    data_length  = len(reward_all)
    cumulative_reward = 0
    cumulative_reward_all = []
    for i in range(data_length):
        if reward_all[i]:
            if home_away_identifier is not None:
                if home_away_identifier[i]:
                    cumulative_reward += 1
                else:
                    cumulative_reward -= 1
            else:
                cumulative_reward += reward_all[i]
        cumulative_reward_all.append(cumulative_reward)

    return  np.asarray(cumulative_reward_all)


def gather_data_values(action_value, action_number, game_name):
    action_value_items = action_value.split(',')
    action_index = int(action_value_items[0])
    action_values_list = np.zeros([action_number])
    value = 0
    if game_name == 'flappybird' or game_name == 'Enduro-v0' or game_name == 'Enduro-v1':
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


if __name__ == "__main__":
    count_actions()