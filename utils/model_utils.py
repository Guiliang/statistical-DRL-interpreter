import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import math

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
    return ((x - y) ** 2).sum(1).mean()


def get_same_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    print('padding is {0}'.format(str(padding)))
    return padding


def calculate_conv_output_dimension(size, kernel_size, stride, dilation, padding):
    return math.floor((size+2*padding-dilation*(kernel_size-1)-1)/stride+1)


def handle_image_input(img_colored,
                       if_print_img=False,
                       if_binarize=False):
    img_colored = Image.fromarray(img_colored)
    img_colored = tf.resize(img_colored, size=(80, 80))
    img_colored = tf.rotate(img_colored, angle=270)
    img_colored = tf.hflip(img_colored)
    img_gray = tf.to_grayscale(img_colored, num_output_channels=1)
    # Image._show(img_gray)
    x_t = tf.to_tensor(img_gray)

    if if_print_img:
        x_t_image = x_t.numpy()
        # x_t_image = tv.transforms.ToPILImage(x_t)
        plt.imshow(x_t_image[0])

    # Apply threshold
    max_value = torch.max(x_t)
    min_value = torch.min(x_t)
    if if_binarize:
        x_t = x_t > (max_value - min_value) / 2  # mean value
        x_t = x_t.float()

    return torch.squeeze(x_t)
