import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision.utils as tu
import math


# import numpy as np
# import cv2


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


def square_loss(x, y):
    return ((x - y) ** 2).mean()


def get_same_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    print('padding is {0}'.format(str(padding)))
    return padding


def calculate_conv_output_dimension(size, kernel_size, stride, dilation, padding):
    return math.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


# def handle_image_input(image, width=84, height=84):
#     image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
#     _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     # return image[None, :, :].astype(np.float32)
#     return torch.from_numpy(image[None, :, :].astype(np.float32))


def handle_image_input(img_colored,
                       if_print_img=False,
                       if_binarize=True):
    img_colored = Image.fromarray(img_colored)
    img_colored_resized = ttf.resize(img_colored, size=(84, 84))
    # img_colored = ttf.rotate(img_colored, angle=270)
    # img_colored = ttf.hflip(img_colored)
    img_gray = ttf.to_grayscale(img_colored_resized, num_output_channels=1)
    # Image._show(img_gray)
    x_t = ttf.to_tensor(img_gray)

    # Apply threshold
    max_value = torch.max(x_t)
    min_value = torch.min(x_t)
    if if_binarize:
        # x_t = x_t > (max_value - min_value) / 2  # mean value
        x_t = x_t > min_value
        x_t = x_t * 255
        x_t = x_t.float()
    if if_print_img:
        x_t_image = x_t.numpy()
        plt.figure()
        plt.imshow(x_t_image[0])

    return x_t


def build_decode_input(z):
    return  torch.from_numpy(z).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


def store_state_action_data(img_colored, action_values, reward, action_index,
                            save_image_path, action_values_file,
                            game_name, iteration_number):
    action_values_str = str(action_index)+','
    for action_value in action_values:
        action_values_str += str(action_value) + ','
    action_values_str += str(reward) + '\n'
    action_values_file.write(action_values_str)

    img_colored = Image.fromarray(img_colored)
    img_colored_save = ttf.rotate(img_colored, angle=270)
    img_colored_save = ttf.hflip(img_colored_save)
    tu.save_image(ttf.to_tensor(img_colored_save),
                  open(save_image_path + 'origin/images/' + game_name + '-' + str(iteration_number) + '_color.png',
                       'wb'))
    img_colored_save_resized = ttf.resize(img_colored_save, size=(64, 64))
    tu.save_image(ttf.to_tensor(img_colored_save_resized),
                  open(save_image_path + 'color/images/' + game_name + '-' + str(iteration_number) + '_color.png',
                       'wb'))
    img_gray_save = ttf.to_grayscale(img_colored_save_resized, num_output_channels=1)
    tu.save_image(ttf.to_tensor(img_gray_save),
                  open(save_image_path + 'gray/images/' + game_name + '-' + str(iteration_number) + '_gray.png', 'wb'))
    x_t_save = ttf.to_tensor(img_gray_save)
    min_value = torch.min(x_t_save)
    x_t_b_save = x_t_save > min_value
    x_t_b_save = x_t_b_save.float()
    tu.save_image(x_t_b_save,
                  open(save_image_path + 'binary/images/' + game_name + '-' + str(iteration_number) + '_binary.png',
                       'wb'))

def compute_latent_importance(gif_tensor, sample_dimension,
                              inter_dimension, latent_dimension,
                              image_width, image_length):
    dim_diff_dict = {}
    masked_gif_tensor = None
    for k in range(latent_dimension):
        image_dim_diff_sum = torch.zeros([3, image_length, image_width])
        image_number = 0
        for i in range(sample_dimension):
            latent_images = gif_tensor[i,:,k]
            for j in range(inter_dimension):
                for m in range(1, inter_dimension-j):
                    image_diff = latent_images[j]-latent_images[j + m]  # TODO: maybe try grey and binary image
                    image_dim_diff_sum+=image_diff.abs().cpu()
                    image_number += 1

        # print('Sum Diff of dim {0} is {1}'.format(str(k), image_dim_diff_sum.sum().numpy()))
        image_dim_diff_average = image_dim_diff_sum[0]/image_number
        dim_diff_dict.update({k: np.sum(image_dim_diff_average.numpy())})
        image_dim_mask = image_dim_diff_average > 0.1
        # tmp = torch.ones(image_dim_mask.size()) - image_dim_mask.double()
        # tmp=tmp.numpy()
        # print(tmp)
        images_dim_mask = image_dim_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sample_dimension, inter_dimension, 3, 1, 1)
        dim_gif_tensor = gif_tensor[:, :, k].cpu()
        masked_dim_gif_tensor = torch.mul(dim_gif_tensor, images_dim_mask)
        # tmp =masked_dim_gif_tensor.numpy()
        gray_mask = (torch.ones(images_dim_mask.size())-images_dim_mask.double())*0.5
        masked_dim_gif_tensor +=gray_mask
        if masked_gif_tensor is None:
            masked_gif_tensor = masked_dim_gif_tensor.unsqueeze(2)
        else:
            masked_gif_tensor = torch.cat([masked_gif_tensor, masked_dim_gif_tensor.unsqueeze(2)], 2)


    sorted_dim_imporatence = sorted(dim_diff_dict.items(), key=lambda kv: kv[1], reverse=True)
    for value in sorted_dim_imporatence:
        print('Sum Diff of dim {0} is {1}'.format(value[0], value[1]))

    return masked_gif_tensor
