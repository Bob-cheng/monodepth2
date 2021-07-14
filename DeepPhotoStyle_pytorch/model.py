from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torchvision.transforms import transforms
import copy
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
from PIL import Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from mwUpdater import MaskWeightUpdater
from dataLoader import KittiLoader



import config
import cv2
import utils

from lr_decay import PolynomialLRDecay

from image_preprocess import scene_size
from depth_model import import_depth_model
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        #print('*************: ', input.size(), self.target.size())
        if input.size() != self.target.size():
            pass
        else:
            channel, height, width = input.size()[1:4]
            self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a = batch size (=1)
    # b = number of feature maps
    # (c, d) = dimensions of a f. map (N=c*d)

    features = input.view(a*b, c*d)

    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c *d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask, content_mask):
        super(StyleLoss, self).__init__()

        self.style_mask = style_mask.detach()
        self.content_mask = content_mask.detach()

        #print(target_feature.type(), mask.type())
        _, channel_f, height, width = target_feature.size()
        channel = self.style_mask.size()[0]
        
        # ********
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)  # (w,h,2)
        grid = grid.unsqueeze_(0).to(config.device0) # (1,w,h,2)
        mask_ = F.grid_sample(self.style_mask.unsqueeze(0), grid).squeeze(0)
        # ********       
        target_feature_3d = target_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        target_feature_masked = torch.zeros(size_of_mask, dtype=torch.float).to(config.device0)
        for i in range(channel):
            target_feature_masked[i, :, :, :] = mask_[i, :, :] * target_feature_3d

        self.targets = list()
        for i in range(channel):
            if torch.mean(mask_[i, :, :]) > 0.0:
                temp = target_feature_masked[i, :, :, :]
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach()/torch.mean(mask_[i, :, :]) )
            else:
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach())
    def forward(self, input_feature):
        self.loss = 0
        _, channel_f, height, width = input_feature.size()
        #channel = self.content_mask.size()[0]
        channel = len(self.targets)
        # ****
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).to(config.device0)
        mask = F.grid_sample(self.content_mask.unsqueeze(0), grid).squeeze(0)
        # ****
        #mask = self.content_mask.data.resize_(channel, height, width).clone()
        input_feature_3d = input_feature.squeeze(0).clone() #TODO why do we need to clone() here? 
        size_of_mask = (channel, channel_f, height, width)
        input_feature_masked = torch.zeros(size_of_mask, dtype=torch.float32).to(config.device0)
        for i in range(channel):
            input_feature_masked[i, :, :, :] = mask[i, :, :] * input_feature_3d
        
        inputs_G = list()
        for i in range(channel):
            temp = input_feature_masked[i, :, :, :]
            mask_mean = torch.mean(mask[i, :, :])
            if mask_mean > 0.0:
                inputs_G.append( gram_matrix(temp.unsqueeze(0))/mask_mean)
            else:
                inputs_G.append( gram_matrix(temp.unsqueeze(0)))
        for i in range(channel):
            mask_mean = torch.mean(mask[i, :, :])
            self.loss += F.mse_loss(inputs_G[i], self.targets[i]) * mask_mean
        
        return input_feature

class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()
        self.ky = np.array([
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]]
        ])
        self.kx = np.array([
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]]
        ])
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(torch.from_numpy(self.kx).float().unsqueeze(0).to(config.device0),
                                          requires_grad=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(self.ky).float().unsqueeze(0).to(config.device0),
                                          requires_grad=False)

    def forward(self, input):
        height, width = input.size()[2:4]
        gx = self.conv_x(input)
        gy = self.conv_y(input)

        # gy = gy.squeeze(0).squeeze(0)
        # cv2.imwrite('gy.png', (gy*255.0).to('cpu').numpy().astype('uint8'))
        # exit()

        self.loss = torch.sum(gx**2 + gy**2)/2.0
        return input

class RealLoss(nn.Module):
    
    def __init__(self, laplacian_m):
        super(RealLoss, self).__init__()
        self.L = Variable(laplacian_m.detach(), requires_grad=False)

    def forward(self, input):
        channel, height, width = input.size()[1:4]
        self.loss = 0
        for i in range(channel):
            temp = input[0, i, :, :]
            temp = torch.reshape(temp, (1, height*width))
            r = torch.mm(self.L, temp.t())
            self.loss += torch.mm(temp , r)
       
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses:
content_layers_default = ['conv4_2'] 
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_mask, content_mask, laplacian_m,
                               content_layer= content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(config.device0)

    # just in order to have an iterable access to or list of content.style losses
    content_losses = []
    style_losses = []
    tv_losses = []
    #real_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn. Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    tv_loss = TVLoss()
    model.add_module("tv_loss_{}".format(0), tv_loss)
    tv_losses.append(tv_loss)
    num_pool = 1
    num_conv = 0
    content_num = 0
    style_num = 0
    for layer in cnn.children():          # cnn feature without fully connected layers
        if isinstance(layer, nn.Conv2d):
            num_conv += 1
            name = 'conv{}_{}'.format(num_pool, num_conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(num_pool, num_conv)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(num_pool)
            num_pool += 1
            num_conv = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(num_pool, num_conv)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layer:
            # add content loss
            print('xixi: ', content_img.size())
            target = model(content_img).detach()
            # print('content target size: ', target.size())
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(content_num), content_loss)
            content_losses.append(content_loss)
            content_num += 1
        if name in style_layers:
            # add style loss:
            # print('style_:', style_img.type())
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, style_mask.detach(), content_mask.detach())
            model.add_module("style_loss_{}".format(style_num), style_loss)
            style_losses.append(style_loss)
            style_num += 1

    # now we trim off the layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses, tv_losses#, real_losses


def get_input_optimizer(params, learning_rate):
    # this line to show that input is a parameter that requires a gradient
    # optimizer = optim.LBFGS([param.requires_grad_() for param in params], lr=learning_rate)
    # optimizer = optim.Adam([input_img.requires_grad_()])
    optimizer = optim.Adam([param.requires_grad_() for param in params], lr=learning_rate, betas=(0.5, 0.9))
    return optimizer
'''
def manual_grad(image, laplacian_m):
    img = image.squeeze(0)
    channel, height, width = img.size() 
    
    loss = 0
    temp = img.reshape(3, -1)
    grad = torch.mm(laplacian_m, temp.t())
    
    loss += (grad * temp.t()).sum()
    return loss, None #2.*grad.reshape(img.size())
'''
def realistic_loss_grad(image, laplacian_m):
    img = image.squeeze(0)
    channel, height, width = img.size()
    loss = 0
    grads = list()
    for i in range(channel):
        grad = torch.mm(laplacian_m, img[i, :, :].reshape(-1, 1))
        loss += torch.mm(img[i, :, :].reshape(1, -1), grad)
        grads.append(grad.reshape((height, width)))
    gradient = torch.stack(grads, dim=0).unsqueeze(0)
    return loss, 2.*gradient

def get_output_car_depth(input_img, car_img, scene_img, paint_mask, car_mask, depth_model,  method='sum'):
    # adv_car_image = input_img * paint_mask.unsqueeze(0) + car_img * (1-paint_mask.unsqueeze(0))
    adv_car_image = input_img
    adv_scene, car_scene, scene_car_mask = attach_car_to_scene_fixed(scene_img, adv_car_image, car_img, car_mask)
    adv_depth = depth_model(adv_scene)
    masked_output = adv_depth * scene_car_mask
    if method == 'sum':
        score = torch.sum(masked_output)
    elif method == 'mean':
        score = torch.sum(masked_output) / torch.sum(scene_car_mask)
    return score
        

def get_adv_loss(input_img, car_img, scene_img, paint_mask, car_mask, depth_model, args):
    batch_size = args["batch_size"]
    # compose adversarial image
    input_img_resize = utils.texture_to_car_size(input_img, car_img.size())
    adv_car_image = input_img_resize * paint_mask.unsqueeze(0) + car_img * (1-paint_mask.unsqueeze(0))
    if args['l1_norm']:
        # adv_scene, car_scene, scene_car_mask = attach_car_to_scene_fixed(scene_img, adv_car_image, car_img, car_mask)
        adv_scene, car_scene, scene_car_mask = attach_car_to_scene(scene_img, adv_car_image, car_img, car_mask, batch_size)
    else:
        adv_scene, car_scene, scene_car_mask = attach_car_to_scene(scene_img, adv_car_image, car_img, car_mask, batch_size)
    adv_depth = depth_model(adv_scene)
    car_depth = depth_model(car_scene)
    scene_depth = depth_model(scene_img)
    if args['l1_norm']:
        scene_depth_bs = scene_depth
    else:
        scene_depth_bs = torch.cat([scene_depth.clone()] * batch_size, dim=0)

    # calculate loss function
    loss_fun = torch.nn.MSELoss()

    adv_scene_loss = loss_fun(adv_depth, (1 - scene_car_mask) * scene_depth_bs)
    # adv_scene_loss = loss_fun(adv_depth, scene_depth_bs)

    adv_car_loss = -loss_fun(adv_depth * scene_car_mask, car_depth * scene_car_mask)
    w_scene = 1
    w_car = 1
    adv_loss = w_scene * adv_scene_loss + w_car * adv_car_loss
    return adv_loss

def get_lp_norm_loss(input_img, car_img, paint_mask):
    assert input_img.size() == car_img.size()
    adv_car_image = input_img * paint_mask.unsqueeze(0) + car_img * (1-paint_mask.unsqueeze(0))
    loss_fun = torch.nn.L1Loss()
    l1_loss = loss_fun(adv_car_image, car_img)
    return l1_loss

def attach_car_to_scene_fixed(scene_img, adv_car_img, car_img, car_mask):
    """
    Attach the car image and adversarial car image to the given scene with fixed position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    
    adv_scene = scene_img.clone()
    car_scene = scene_img.clone()
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
    
    for idx_Bat in range(B_Sce):
        scale = 0.7 # 600 -- 0.4, 300 -- 0.7
        # Do some transformation on the adv_car_img together with car_mask
        trans_seq = transforms.Compose([ 
            transforms.Resize([int(H * scale), int(W * scale)])
            ])

        adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        car_img_trans = trans_seq(car_img).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)
        bottom_height = int(bottom_range - scale * max(bottom_range - 10, 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
        # left = left_range // 2
        left = random.randint(50, left_range-50)

        h_index = H_Sce - H_Car - bottom_height
        w_index = left
        h_range = slice(h_index, h_index + H_Car)
        w_range = slice(w_index, w_index + W_Car)

        car_area_in_scene = adv_scene[idx_Bat, :, h_range, w_range]
        adv_scene[idx_Bat, :, h_range, w_range] = \
            adv_car_img_trans * car_mask_trans + car_area_in_scene * (1- car_mask_trans)
        car_scene[idx_Bat, :, h_range, w_range] = \
                car_img_trans * car_mask_trans + car_area_in_scene * (1 - car_mask_trans)
        scene_car_mask[idx_Bat, :, h_range, w_range] = car_mask_trans
        # utils.save_pic(adv_scene[idx_Bat,:,:,:], f'attached_adv_scene_{idx_Bat}')
        # utils.save_pic(car_scene[idx_Bat,:,:,:], f'attached_car_scene_{idx_Bat}')
        # utils.save_pic(scene_car_mask[idx_Bat,:,:,:], f'attached_scene_mask_{idx_Bat}')
    return adv_scene, car_scene, scene_car_mask

def attach_car_to_scene(scene_img, adv_car_img, car_img, car_mask, batch_size):
    """
    Attach the car image and adversarial car image to the given scene with random position. 
    The scene could have multiple images (batch size > 1)
    scene_img: B * C * H * W
    car_img:   1 * C * H * W
    car_mask:      1 * H * W
    """
    _, _, H, W = adv_car_img.size()
    if scene_img.size()[0] == batch_size:
        adv_scene = scene_img.clone()
        car_scene = scene_img.clone()
    else:
        adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
    scene_car_mask = torch.zeros(adv_scene.size()).float().to(config.device0)
    
    B_Sce, _, H_Sce, W_Sce = adv_scene.size()
    
    for idx_Bat in range(B_Sce):
        # scale = 0.7 # 600 -- 0.4, 300 -- 0.7
        scale_upper = 0.8
        scale_lower = 0.6
        scale = (scale_upper - scale_lower) * torch.rand(1) + scale_lower
        # Do some transformation on the adv_car_img together with car_mask
        trans_seq = transforms.Compose([ 
            transforms.RandomRotation(degrees=3),
            transforms.Resize([int(H * scale), int(W * scale)])
            ])
        trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

        adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
        car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

        # adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
        # car_img_trans = trans_seq(car_img).squeeze(0)
        
        car_mask_trans = trans_seq(car_mask)

        # paste it on the scene
        _, H_Car, W_Car = adv_car_img_trans.size()
        
        left_range = W_Sce - W_Car
        bottom_range = int((H_Sce - H_Car)/2)

        bottom_height = int(bottom_range - scale * max(bottom_range - 10, 0))  # random.randint(min(10, bottom_range), bottom_range) # 20 
        left = random.randint(50, left_range-50)
        h_index = H_Sce - H_Car - bottom_height
        w_index = left
        h_range = slice(h_index, h_index + H_Car)
        w_range = slice(w_index, w_index + W_Car)

        car_area_in_scene = adv_scene[idx_Bat, :, h_range, w_range]
        adv_scene[idx_Bat, :, h_range, w_range] = \
            adv_car_img_trans * car_mask_trans + car_area_in_scene * (1- car_mask_trans)
        car_scene[idx_Bat, :, h_range, w_range] = \
                car_img_trans * car_mask_trans + car_area_in_scene * (1 - car_mask_trans)
        scene_car_mask[idx_Bat, :, h_range, w_range] = car_mask_trans
        # utils.save_pic(adv_scene[idx_Bat,:,:,:], f'attached_adv_scene_{idx_Bat}')
        # utils.save_pic(car_scene[idx_Bat,:,:,:], f'attached_car_scene_{idx_Bat}')
        # utils.save_pic(scene_car_mask[idx_Bat,:,:,:], f'attached_scene_mask_{idx_Bat}')
    return adv_scene, car_scene, scene_car_mask
    
def log_perterbation(logger, input_img, car_img, paint_mask, step):
    assert input_img.size() == car_img.size()
    adv_car_image = input_img * paint_mask.unsqueeze(0) + car_img * (1-paint_mask.unsqueeze(0))
    perterbation = adv_car_image - car_img
    perterbation_sum = torch.sum(torch.abs(perterbation), dim=1, keepdim=True)
    perterbation_norm = perterbation_sum / torch.max(perterbation_sum)
    # print(perterbation_norm.size())
    logger.add_image('Train/Perterbation', perterbation_norm[0], step)

def color_mapping(image, maptype='magma', vmax=None, vmin=None):
    image_np = image.detach().squeeze().cpu().numpy()
    # vmax = np.percentile(image_np, 98)
    vmax = image_np.max() if vmax == None else vmax
    vmin = image_np.min() if vmin == None else vmin

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=maptype)
    colormapped_im = (mapper.to_rgba(image_np)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def eval_depth_diff(img1: torch.tensor, img2: torch.tensor, depth_model, filename):
    disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7)) # width, height
    plt.subplot(321); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(323)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    plt.subplot(324)
    plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(325)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    plt.subplot(326)
    plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    # plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image, disp1, disp2

def vis_input_grad(logger: SummaryWriter,paint_mask,  input_img: torch.Tensor):
    input_img_grad = paint_mask.unsqueeze(0) * input_img.grad
    input_img_grad_l1 = torch.sum(torch.abs(input_img_grad), dim=1, keepdim=True)
    # input_img_grad_l1 = input_img_grad_l1 / torch.max(input_img_grad_l1)
    input_img_grad_l1 = input_img_grad_l1 / torch.quantile(input_img_grad_l1, 0.98)
    input_img_grad_l1 = input_img_grad_l1.clamp(0, 1)
    print(input_img_grad_l1.size())
    colormap_im = color_mapping(input_img_grad_l1)
    im = pil.fromarray(colormap_im)
    im.save('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/grad_vis.png')
    # utils.save_pic(input_img_grad_l1, 'grad_vis', '/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/')
    # logger.add_image('Debug/input_grad', input_img_grad_l1[0], 0)

def get_mask_loss(paint_mask):
    # mapped_mask = 0.5 * torch.tanh(20 * paint_mask - 2) + 0.5
    mapped_mask = paint_mask
    loss = torch.sum(torch.abs(mapped_mask))
    # loss = torch.mean(torch.abs(mapped_mask))
    return loss

def run_style_transfer(logger: SummaryWriter, cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, car_img, scene_img_1, test_scene_img_1,
                       style_mask, content_mask, paint_mask_boarders, car_mask, laplacian_m,
                       args):

    """Run the style transfer."""
    style_weight = args['style_weight']
    content_weight = args['content_weight']
    tv_weight = args['tv_weight']
    rl_weight = args['rl_weight']
    l1_weight = args['l1_weight']
    mask_weight = args['mask_weight']
    num_steps = args['steps']
    adv_weight = args['adv_weight']
    learning_rate = args["learning_rate"]

    if args['random_scene']:
        kitti_loader_train = KittiLoader(mode='train',  train_list='trainval.txt', val_list='val.txt')
        train_loader = DataLoader(kitti_loader_train, batch_size=args["batch_size"], shuffle=True, num_workers=3, pin_memory=True)
        scene_data_len = len(train_loader)
        train_loader_iter = iter(train_loader)
        
        print("Using random scene... Scene dataset size: ", scene_data_len)

    # mask weight multiplier:
    # paint_mask = utils.from_inf_to_mask(paint_mask_inf, car_mask.shape)
    paint_mask = utils.make_square_mask(car_mask.size(), paint_mask_boarders)

    mask_loss_thresh = torch.sum(torch.abs(torch.ones(paint_mask.size()))).item()/16
    mwUpdater = MaskWeightUpdater(mask_weight, mask_loss_thresh)


    print("Buliding the style transfer model..")
    model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, style_mask, content_mask, laplacian_m)
    
    # get deepth model
    depth_model = import_depth_model(scene_size).to(config.device0).eval()
    for param in depth_model.parameters():
        param.requires_grad = False
    
    # optimizer = get_input_optimizer(input_img, learning_rate)
    # optimizer = get_input_optimizer([input_img, paint_mask_inf], learning_rate)
    optimizer = get_input_optimizer([input_img, paint_mask_boarders], learning_rate)

    LR_decay = PolynomialLRDecay(optimizer, num_steps//2, learning_rate/2, 0.9)

    print("Optimizing...")
    print('*'*20)
    print("Style_weith: {} Content_weighti: {} \
           TV_loss_weight: {} Realistic_loss_weight: {}".format \
           (style_weight, content_weight, tv_weight, rl_weight))
    print('*'*20)
    run = [0]
    
    best_loss = 1e10    
    best_adv_loss = 1e10
    best_input = input_img.data 
    best_adv_input = input_img.data 

    mask_loss = torch.zeros(1)

    # calculate output gradiant on input
    # optimizer.zero_grad()
    # output_car = get_output_car_depth(input_img, car_img, scene_img, paint_mask, car_mask, depth_model)
    # output_car.backward()
    # vis_input_grad(logger,paint_mask, input_img)

    while run[0] <= num_steps:

        def closure(): 
            nonlocal best_loss
            nonlocal input_img
            nonlocal best_input
            nonlocal best_adv_loss
            nonlocal best_adv_input
            nonlocal train_loader_iter
            nonlocal paint_mask_boarders
            nonlocal mask_loss

            input_img.data.clamp_(0, 1)
            # paint_mask = utils.from_inf_to_mask(paint_mask_inf, car_mask.shape)

            # paint_mask_boarders[0:1].clamp_(0, car_mask.size()[2])
            # paint_mask_boarders[2:3].clamp_(0, car_mask.size()[1])
            # paint_mask_boarders.data += torch.normal(torch.zeros(4), torch.ones(4)*1).to(config.device0)
            paint_mask = utils.make_square_mask(car_mask.size(), paint_mask_boarders)
            optimizer.zero_grad()

            style_score = torch.zeros(1)
            content_score = torch.zeros(1)
            tv_score = torch.zeros(1)
            rl_score = torch.zeros(1)
            l1_loss = torch.zeros(1)
            adv_loss = torch.zeros(1)

            if not args['l1_norm']:
                model(input_img)
                for sl in style_losses:
                    style_score += sl.loss

                for cl in content_losses:
                    content_score += cl.loss

                for tl in tv_losses:
                    tv_score += tl.loss
            
            style_score *= style_weight
            content_score *= content_weight
            tv_score *= tv_weight 
            
            if args['random_scene']:
                try:
                    scene_img, _ = next(train_loader_iter)
                    if scene_img.size()[0] != args['batch_size']:
                        raise StopIteration
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    scene_img, _ = next(train_loader_iter)
                scene_img = scene_img.to(config.device0)
            else:
                scene_img = scene_img_1

            adv_loss = get_adv_loss(input_img, car_img, scene_img, paint_mask, car_mask, depth_model, args)
            # adv_weight = 1000000
            adv_loss *= adv_weight

            if args['l1_norm']:
                l1_loss = get_lp_norm_loss(input_img, car_img, paint_mask)
                l1_loss *= l1_weight
                mask_loss = get_mask_loss(paint_mask)
                # mask_loss *= mask_weight
                mw = mwUpdater.step(mask_loss)
                mask_loss *= mw
                loss = l1_loss + adv_loss + mask_loss
                loss.backward()
            else:
                manual_grad = False
                # Two stage optimaztion pipline    
                if run[0] > num_steps // 2:
                    rl_score, part_grid = realistic_loss_grad(input_img, laplacian_m)
                    rl_score *= rl_weight
                    part_grid *= rl_weight
                    if manual_grad:
                        # Realistic loss relate sparse matrix computing, 
                        # which do not support autogard in pytorch, so we compute it separately.
                        loss = style_score + content_score + tv_score + adv_loss # + rl_score
                        loss.backward()
                        input_img.grad += part_grid
                        loss = loss + rl_score
                    else:
                        loss = style_score + content_score + tv_score + adv_loss + rl_score
                        loss.backward()
                else:
                    loss = style_score + content_score + tv_score + adv_loss
                    loss.backward()

            if loss < best_loss and run[0] > 0:
                # print(best_loss)
                best_loss = loss
                best_input = input_img.data.clone()
            
            if adv_loss < best_adv_loss and run[0] > 0:
                best_adv_loss = adv_loss
                best_adv_input = input_img.data.clone()

            if run[0] == num_steps // 2:
                # Store the best temp result to initialize second stage input
                input_img.data = best_input
                best_loss = 1e10
                LR_decay.step(0)
            
            # Gradient cliping deal with gradient exploding
            clip_grad_norm_(model.parameters(), 15.0)
          
            run[0] += 1
            if run[0] % 30 == 0:
                print("run {}/{}:".format(run, num_steps))
        
                print('Style Loss: {:4f} Content Loss: {:4f} TV Loss: {:4f} real loss: {:4f} adv_loss: {:4f} l1_norm_loss: {:4f} mask_loss: {:4f}'.format(
                   style_score.item(), content_score.item(), tv_score.item(), rl_score.item(), adv_loss.item(), l1_loss.item(), mask_loss.item()))

                print('Total Loss: ', loss.item())

                logger.add_scalar('Train/Style_loss', style_score.item(), run[0])
                logger.add_scalar('Train/Content_loss', content_score.item(), run[0])
                logger.add_scalar('Train/TV_loss', tv_score.item(), run[0])
                logger.add_scalar('Train/Real_loss', rl_score.item(), run[0])
                logger.add_scalar('Train/Adv_loss', adv_loss.item(), run[0])
                logger.add_scalar('Train/Total_loss', loss.item(), run[0])
                logger.add_scalar('Train/L1_norm_loss', l1_loss.item(), run[0])
                logger.add_scalar('Train/Mask_loss', mask_loss.item(), run[0])
                logger.add_scalar('Train/Mask_weight', mwUpdater.get_mask_weight(), run[0])
                logger.add_image('Train/Paint_mask', np.moveaxis(color_mapping(paint_mask, vmax=1, vmin=0), -1, 0), run[0])


                if run[0] % 300 == 0:
                    texture_img = utils.texture_to_car_size(input_img.data.clone(), car_img.size())
                    # add mask and evluate
                    saved_img = texture_img * paint_mask.unsqueeze(0) + car_img * (1-paint_mask.unsqueeze(0))
                    saved_img.data.clamp_(0, 1)
                    if args['random_scene']:
                        test_scene_img, _ = next(train_loader_iter)
                        test_scene_img = test_scene_img.to(config.device0)
                    else:
                        test_scene_img = test_scene_img_1
                    if args['l1_norm']:
                        # adv_scene_out, car_scene_out, _ = attach_car_to_scene_fixed(test_scene_img, saved_img, car_img, car_mask)
                        adv_scene_out, car_scene_out, _ = attach_car_to_scene(test_scene_img, saved_img, car_img, car_mask, args["batch_size"])
                        log_perterbation(logger, input_img, car_img, paint_mask, run[0])
                        logger.add_image('Train/Paint_mask_scaled', np.moveaxis(color_mapping(paint_mask, vmin=0), -1, 0), run[0])
                    else:
                        adv_scene_out, car_scene_out, _ = attach_car_to_scene(test_scene_img, saved_img, car_img, car_mask, args["batch_size"])
                    # utils.save_pic(adv_scene_out[[0]], run[0])
                    result_img, _, _ = eval_depth_diff(car_scene_out[[0]], adv_scene_out[[0]], depth_model, f'depth_diff_{run[0]}')
                    logger.add_image('Train/Compare', utils.image_to_tensor(result_img), run[0])
                    logger.add_image('Train/Car_scene', car_scene_out[0], run[0])
                    logger.add_image('Train/Adv_scene', adv_scene_out[0], run[0])
                    logger.add_image('Train/Adv_car', saved_img[0], run[0])
                    # logger.add_image('Train/Adv_patch', utils.extract_patch(saved_img, paint_mask)[0], run[0])
            return loss

        optimizer.step(closure)
        LR_decay.step()
        if mask_loss.item() < mask_loss_thresh * mask_weight:
            print("mask loss threshold reached!")
            break
    # a last corrention...
    input_img.data = best_input
    input_img.data.clamp_(0, 1)

    return input_img, depth_model