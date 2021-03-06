import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
import torchvision.utils as tu
import math
from torch import nn
from torch.nn import Parameter
from torchvision.utils import save_image


# import numpy as np
# import cv2

def prediction_loss(y_predict, y_true):
    mse = nn.MSELoss(reduction='mean')
    loss = mse(y_predict, y_true)
    return loss

def recon_loss(x, x_recon, if_cross_entropy=True):
    n = x.size(0)
    if if_cross_entropy:
        loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    else:
        mse = nn.MSELoss(reduction='mean')
        loss = mse(x_recon, x)
    return loss


def kl_divergence(mu_q, logvar_q, mu_p=None, logvar_p=None):
    if mu_p is None and logvar_p is None:
        kld = -0.5 * (1 + logvar_q - mu_q ** 2 - logvar_q.exp()).sum(1).mean()
    else:
        kld = 0.5 * (logvar_p - logvar_q +
                     (logvar_q.exp() + (mu_q - mu_p)**2) / logvar_p.exp() - 1).sum(1).mean()
    return kld

def tree_construct_loss(leaf_number):
    entropy_prob = leaf_number / (2 * leaf_number - 1)
    big_o = 1/leaf_number
    structure_cost = math.log((2 * leaf_number - 1) ** 2 / ((leaf_number ** 1.5) * (leaf_number - 1) ** 0.5)) + \
                     (2 * leaf_number - 1) * (-entropy_prob * math.log(entropy_prob) - (1 - entropy_prob) * math.log(1 - entropy_prob))+\
                     big_o
    return structure_cost


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
    if game_name == "flappybird":
        img_colored_save = ttf.rotate(img_colored, angle=270)
        img_colored_save = ttf.hflip(img_colored_save)
        img_colored_save = ttf.resize(img_colored_save, size=(84, 84))
        # img_colored_save.show()
    else:
        img_colored_save = img_colored

    origin_save_dir = save_image_path + 'origin/images/{0}-{1}_action{2}_origin.png'.format(game_name,
                                                                                            iteration_number,
                                                                                            action_index)
    tu.save_image(ttf.to_tensor(img_colored_save), open(origin_save_dir, 'wb'))

    img_colored_save_resized = ttf.resize(img_colored_save, size=(64, 64))
    # img_colored_save_resized.show()
    color_save_dir = save_image_path + 'color/images/{0}-{1}_action{2}_color.png'.format(game_name,
                                                                                         iteration_number,
                                                                                         action_index)
    tu.save_image(ttf.to_tensor(img_colored_save_resized), open(color_save_dir,'wb'))
    gray_save_dir = save_image_path + 'gray/images/{0}-{1}_action{2}_gray.png'.format(game_name,
                                                                                      iteration_number,
                                                                                      action_index)

    img_gray_save = ttf.to_grayscale(img_colored_save_resized, num_output_channels=1)
    tu.save_image(ttf.to_tensor(img_gray_save), open(gray_save_dir, 'wb'))
    binary_save_dir = save_image_path + 'binary/images/{0}-{1}_action{2}_binary.png'.format(game_name,
                                                                                      iteration_number,
                                                                                      action_index)
    x_t_save = ttf.to_tensor(img_gray_save)
    min_value = torch.min(x_t_save)
    x_t_b_save = x_t_save > min_value
    x_t_b_save = x_t_b_save.float()
    tu.save_image(x_t_b_save, open(binary_save_dir, 'wb'))


def compute_diff_masked_images(images_tensor):
    [inter_dimension, _ ,image_length, image_width] = images_tensor.shape
    image_dim_diff_sum = torch.zeros([3, image_length, image_width])
    image_number = 0
    for j in range(inter_dimension):
        for m in range(1, inter_dimension - j):
            image_diff = images_tensor[j] - images_tensor[j + m]  # TODO: maybe try grey and binary image
            image_dim_diff_sum += image_diff.abs().cpu()
            image_number += 1
    # tmp = image_dim_diff_sum[0].numpy()
    image_dim_diff_average = image_dim_diff_sum[0] / image_number
    image_dim_mask = image_dim_diff_average > 0.1
    # images_dim_mask = image_dim_mask.unsqueeze(0).\
    #     repeat(inter_dimension, 1, 1)
    # blue_mask = images_dim_mask.double() * 0.5
    # masked_images_tensor = images_tensor.cpu()
    # masked_images_tensor[:,2,:,:] += blue_mask

    images_dim_mask = image_dim_mask.unsqueeze(0).unsqueeze(0).\
        repeat(inter_dimension, 3, 1, 1)
    masked_images_tensor = torch.mul(images_tensor.cpu(), images_dim_mask)
    # tmp =images_tensor.cpu().numpy()
    gray_mask = (torch.ones(images_dim_mask.size()) - images_dim_mask.double()) * 0.4
    # masked_images_tensor[:,2,:,:] += gray_mask[:,2,:,:]
    masked_images_tensor += gray_mask
    return masked_images_tensor



def compute_latent_importance(gif_tensor, sample_dimension,
                              inter_dimension, latent_dimension,
                              image_width, image_length):
    dim_diff_dict = {}
    masked_gif_tensor = None
    masked_dim_number = 0
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
        # image_dim_diff_average = image_dim_diff_sum[0]/image_number
        image_dim_diff_average = torch.mean(image_dim_diff_sum, 0, keepdim=False) / image_number
        dim_diff_dict.update({k: np.sum(image_dim_diff_average.numpy())})
        image_dim_mask = image_dim_diff_average > 0.045
        # tmp = torch.ones(image_dim_mask.size()) - image_dim_mask.double()
        # tmp=tmp.numpy()
        # print(tmp)
        images_dim_mask = image_dim_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(sample_dimension, inter_dimension, 3, 1, 1)
        dim_gif_tensor = gif_tensor[:, :, k].cpu()
        masked_dim_gif_tensor = torch.mul(dim_gif_tensor, images_dim_mask)
        # tmp =masked_dim_gif_tensor.numpy()
        gray_mask = (torch.ones(images_dim_mask.size())-images_dim_mask.double())*0.5
        masked_dim_gif_tensor +=gray_mask

        max_value = torch.max(masked_dim_gif_tensor).cpu().numpy()
        min_value = torch.min(masked_dim_gif_tensor).cpu().numpy()
        ignore_dim_flag = True if max_value == min_value else False

        if not ignore_dim_flag:
            masked_dim_number += 1
            if masked_gif_tensor is None:
                masked_gif_tensor = masked_dim_gif_tensor.unsqueeze(2)
            else:
                masked_gif_tensor = torch.cat([masked_gif_tensor, masked_dim_gif_tensor.unsqueeze(2)], 2)


    sorted_dim_importance = sorted(dim_diff_dict.items(), key=lambda kv: kv[1], reverse=True)
    for value in sorted_dim_importance:
        print('Sum Diff of dim {0} is {1}'.format(value[0], value[1]))

    return masked_gif_tensor, masked_dim_number



def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def calc_gradient_penalty(batch_size, model, real_data, gen_data, device, dataset, gp):
    datashape = model.shape
    alpha = torch.rand(batch_size, 1)
    real_data = real_data.view(batch_size, -1)

    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size)
    alpha = alpha.contiguous().view(batch_size, -1).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = model(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

    if dataset != 'mnist':
        gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp
    return gradient_penalty


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



def visualize_split(selected_action, state, data_all, decoder, device, z_dim, image_id):
    selected_state_index = int(selected_action.split('_')[0])
    selected_dim = int(selected_action.split('_')[1])
    selected_split_value = float(selected_action.split('_')[2])
    # splitted_states = state[selected_state_index:selected_state_index + 2]
    splitted_states = state
    splitted_states_avg = []
    state_features_all = []
    for state_index in range(len(splitted_states)):
        state_features = []
        state = splitted_states[state_index]
        for data_index in state:
            # z_index = np.concatenate([data_all[data_index][0],
            #                           data_all[data_index][3]], axis=0)
            z_index = data_all[data_index][0]
            state_features.append(z_index)
            state_features_all.append(z_index)
        state_features_avg = np.average(np.asarray(state_features), axis=0)
        splitted_states_avg.append(state_features_avg)
        # print(splitted_states_avg)
    state_features_all_avg = np.average(np.asarray(state_features_all), axis=0)
    x_recon_all = None
    for state_index in range(len(splitted_states_avg)):
        state_features_avg = np.copy(state_features_all_avg)
        state_features_avg[selected_dim] = splitted_states_avg[state_index][selected_dim]
        # state_features_avg = splitted_states_avg[state_index]
        z_state = build_decode_input(state_features_avg[:z_dim])
        # z_2_state = build_decode_input(state_features_avg[z_dim:])
        # z_state = torch.cat([z_1_state, z_2_state], axis=0)
        with torch.no_grad():
            x_recon = F.sigmoid(decoder(z_state.float().to(device))).data

        if x_recon_all is None:
            x_recon_all = x_recon
        else:
            x_recon_all = torch.stack([x_recon_all, x_recon], axis=0)

    # x_recon_all = torch.cat([x_recon_all[:, 0, :, :, :], x_recon_all[:, 1, :, :, :]], axis=-2)
    x_recon_all = torch.squeeze(x_recon_all, 1)
    masked_images = compute_diff_masked_images(x_recon_all)
    # masked_images = x_recon_all
    # masked_images = torch.stack(torch.split(masked_images, int(masked_images.shape[-2] / 2), dim=-2), axis=1)
    save_image(tensor=masked_images[0], fp="../mimic_learner/action_images_plots/img_{0}_split_{1}_image_left.jpg".
               format(image_id, selected_action), nrow=1, pad_value=1)
    save_image(tensor=masked_images[1], fp="../mimic_learner/action_images_plots/img_{0}_split_{1}_image_right.jpg".
               format(image_id, selected_action), nrow=1, pad_value=1)


def generate_attention_maps():
    img = Image.open('../tmp/flappybird-example.png')
    img_np = np.array(img)
    # img_np.setflags(write=1)
    # img = Image.fromarray(np.uint8(img_np))
    # img.show()
    tmp_img_np = img_np[:,:,0]

    # tmp = tmp_img_np[:, 150]
    # for i in tmp:
    #     print(i)

    map_list = [gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=10, sigma_y=10, x0=130, y0=260),
                gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=10, sigma_y=10, x0=140, y0=250),
                gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=8, sigma_y=8, x0=160, y0=260),
                gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=10, sigma_y=8, x0=150, y0=280),

                3*gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=30, sigma_y=10, x0=180, y0=127),
                # gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=10, sigma_y=10, x0=185, y0=127),
                gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=12, sigma_y=9, x0=195, y0=127),
                0.2*gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=8, sigma_y=5, x0=170, y0=120),

                3 * gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=25, sigma_y=15, x0=173, y0=330),
                gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=12, sigma_y=9, x0=155, y0=335),
                0.2 * gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=7, sigma_y=4, x0=150, y0=325),
                0.01 * gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=3, sigma_y=3, x0=153, y0=327),
                0.1 * gauss_map(img_np.shape[1], img_np.shape[0], sigma_x=3, sigma_y=3, x0=140, y0=330),
                ]

    map_all = np.zeros([img_np.shape[0], img_np.shape[1]])

    for map in map_list:
        map_all += map

    map_all = 255*map_all/(np.max(map_all)-np.min(map_all))
    img_np[:, :, 1] = tmp_img_np*0.5 + map_all*0.5
    img = Image.fromarray(np.uint8(img_np))
    img.show()
    # img.save('./flappybird-attentions.png')


def gauss_map(size_x, size_y=None, sigma_x=5, sigma_y=None, x0=None, y0=None):
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x

    assert isinstance(size_x, int)
    assert isinstance(size_y, int)

    if x0 is None:
        x0 = size_x // 2
    if y0 is None:
        y0 = size_y // 2

    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:, np.newaxis]

    x -= x0
    y -= y0

    exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

if __name__ == "__main__":
    generate_attention_maps()
