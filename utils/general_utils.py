import os
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


def return_data(config, global_model_data_path):
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
        root = os.path.join(dset_dir, 'flappybird/'+config.image_type)  # TODO: you might want to try colored?
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    elif name.lower() == 'assault-v0':
        root = os.path.join(dset_dir, 'Assault-v0/'+config.image_type)  # TODO: you might want to try colored?
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    elif name.lower() == 'breakout-v0':
        root = os.path.join(dset_dir, 'Breakout-v0/'+config.image_type)  # TODO: you might want to try colored?
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    elif name.lower() == 'spaceinvaders-v0':
        root = os.path.join(dset_dir, 'SpaceInvaders-v0/'+config.image_type)  # TODO: you might want to try colored?
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
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

if __name__ == "__main__":
    count_actions()