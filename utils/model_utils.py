import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision.utils as tu
import math
import numpy as np
import cv2


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
                       if_binarize=True,
                       save_image_path=None,
                       iter=None):
    img_colored = Image.fromarray(img_colored)
    img_colored = ttf.resize(img_colored, size=(84, 84))
    # img_colored = ttf.rotate(img_colored, angle=270)
    # img_colored = ttf.hflip(img_colored)
    img_gray = ttf.to_grayscale(img_colored, num_output_channels=1)
    # Image._show(img_gray)
    x_t = ttf.to_tensor(img_gray)

    if save_image_path is not None:
        tu.save_image(x_t, open(save_image_path + '-' + str(iter) + '.png', 'wb'))

    # Apply threshold
    max_value = torch.max(x_t)
    min_value = torch.min(x_t)
    if if_binarize:
        # x_t = x_t > (max_value - min_value) / 2  # mean value
        x_t = x_t > min_value
        x_t = x_t*255
        x_t = x_t.float()
    if if_print_img:
        x_t_image = x_t.numpy()
        plt.figure()
        plt.imshow(x_t_image[0])
    return x_t