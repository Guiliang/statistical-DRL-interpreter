import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
# import sys
# cwd = os.getcwd()
# sys.path.append(cwd.replace('/explaination_method_comparison', ''))
# print (sys.path)
from data_generator.fb_game.flappy_bird import FlappyBird
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
# prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
from data_generator.nn_drl.dqn_fb import FlappyBirdDQN
from utils.model_utils import handle_image_input

occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur

class SaliencyMap:
    "check implementation from https://github.com/greydanus/visualize_atari"
    def __init__(self, filepath='../data_generator/saved_models/flappy_bird_model'):

        use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        actions_number = 2
        # self.nn = DeepQNetwork().to(self.device)
        self.model = FlappyBirdDQN().to(self.device)

        if os.path.isfile(filepath):
            if self.device == 'cuda':
                with open(filepath, 'rb') as f:
                    checkpoint = torch.load(f)
            else:
                with open(filepath, 'rb') as f:
                    checkpoint = torch.load(f, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_states']['FlappyBirdDQN'])

        self.game_env = FlappyBird()
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)

        self.radius = 5
        self.density = 5
        self.global_iter = 0

    def prepro(self, img):
        img_resized = Image.fromarray(img[35:195].mean(2)).resize(size=(80,80))
        return img_resized.astype(np.float32).reshape(1,80,80)/255.

    def get_mask(self, center, size, r):
        y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
        keep = x*x + y*y <= 1
        mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
        mask = gaussian_filter(mask, sigma=r).astype(np.float32) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask/mask.max()

    def run_through_model(self, model, img_raw, interp_func=None, mask=None, blur_memory=None):
        if mask is None:
            im = handle_image_input(img_raw)
        else:
            assert(interp_func is not None, "interp func cannot be none")
            im = handle_image_input(img_raw)
            im = interp_func(im, mask).reshape(1,84,84) # perturb input I -> I'

        tens_state = torch.cat(tuple(im for _ in range(4))).to(self.device)
        # state = Variable(tens_state.unsqueeze(0))
        return model(tens_state.unsqueeze(0))

    def score_frame(self, model, img_raw, interp_func, action_index):
        # r: radius of blur
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)
        action_values = self.run_through_model(model, img_raw, interp_func, mask=None)[0]
        select_action_index = np.argmax(action_values.cpu().detach().numpy())
        L = action_values[0]
        scores = np.zeros((int(84/self.density)+1,int(84/self.density)+1)) # saliency scores S(t,i,j)
        for i in range(0,84,self.density):
            for j in range(0,84,self.density):
                mask = self.get_mask(center=[i,j], size=[84,84], r=self.radius)
                l = self.run_through_model(model, img_raw, interp_func, mask=mask)[0][action_index]
                scores[int(i/self.density),int(j/self.density)] = (L-l).pow(2).sum().mul_(.5).data
        pmax = scores.max()
        scores = np.asarray(Image.fromarray(scores).resize(size=[84,84], resample=Image.BILINEAR)).astype(np.float32)
        return pmax * scores / scores.max(), select_action_index

    def saliency_on_atari_frame(self, saliency, atari, fudge_factor, channel=2, sigma=0):
        # sometimes saliency maps are a bit clearer if you blur them
        # slightly...sigma adjusts the radius of that blur
        pmax = saliency.max()
        S = np.asarray(Image.fromarray(saliency).resize(size=[atari.shape[1], atari.shape[0]], resample=Image.BILINEAR)).astype(np.float32)
        S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
        S -= S.min()
        S = fudge_factor * pmax * S / S.max()
        I = atari.astype('uint16')
        I[:, :, channel] += S.astype('uint16')
        I = I.clip(1, 255).astype('uint8')
        return I

    def saliency_map_perturbation(self):
        test_size = 101
        action_index = 0
        while self.global_iter < test_size:

            x_t0_colored, r_t, terminal = self.game_env.next_frame(action_index)
            x_t0_colored = x_t0_colored[:self.game_env.screen_width, :int(self.game_env.base_y)]

            if self.global_iter >= 50:
                saliency_map, action_index = self.score_frame(self.model, x_t0_colored, interp_func=occlude, action_index=action_index)
                perturbation_map = self.saliency_on_atari_frame(saliency_map, x_t0_colored, fudge_factor=10000, channel=2)
                perturbation_map = Image.fromarray(perturbation_map).rotate(angle=270)
                perturbation_map = ImageOps.mirror(perturbation_map)
                perturbation_map.save('/Users/liu/PycharmProjects/statistical-DRL-interpreter/tmp/tmp-{0}.png'.format(self.global_iter))
            else:
                action_values = self.run_through_model(self.model, x_t0_colored, occlude, mask=None)[0]
                action_index = np.argmax(action_values.cpu().detach().numpy())
            self.global_iter += 1

if __name__ == "__main__":
    SM = SaliencyMap()
    SM.saliency_map_perturbation()