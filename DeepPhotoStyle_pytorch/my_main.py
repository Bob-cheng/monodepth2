import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

import argparse

# ------custom module----
import config
import utils
from image_preprocess import prepare_dir, process_content_img, process_style_img, process_scene_img
from image_preprocess import gen_content_path

sys.path.append('seg')
from seg.segmentation import *
from model import *
from merge_index import *


if __name__ == '__main__':
    torch.manual_seed(17)
    #----------init------------
    ap = argparse.ArgumentParser()

    ap.add_argument("-s", "--style_image", required=True, 
        help="name of the style image")

    ap.add_argument("-c", "--content_image", required=True,
        help="name of the content image")

    args = vars(ap.parse_args())

    style_image_name = args["style_image"]
    content_image_name = args["content_image"]

    prepare_dir()
    style_img_resize, style_mask_np = process_style_img(style_image_name)
    content_img_resize, car_mask_np, paint_mask_np = process_content_img(content_image_name)
    scene_img_crop = process_scene_img('0000000017.png')

    #-------------------------
    print('Computing Laplacian matrix of content image')
    content_image_path = os.path.join(gen_content_path, content_image_name)
    L = utils.compute_lap(content_image_path)
    print()
    
    style_mask_tensor = torch.from_numpy(style_mask_np).unsqueeze(0).float().to(config.device0)
    content_mask_tensor = torch.from_numpy(paint_mask_np).unsqueeze(0).float().to(config.device0)
    car_mask_tensor = torch.from_numpy(car_mask_np).unsqueeze(0).float().to(config.device0)
    #--------------------------
    print('Save each mask as an image for debugging')
    for i in range(style_mask_tensor.shape[0]):
        utils.save_pic( torch.stack([style_mask_tensor[i, :, :], style_mask_tensor[i, :, :], style_mask_tensor[i, :, :]], dim=0), 
                                    'style_mask_' + str(i) )
        utils.save_pic( torch.stack([content_mask_tensor[i, :, :], content_mask_tensor[i, :, :], content_mask_tensor[i, :, :]], dim=0), 
                                    'content_mask_' + str(i) )
    
    # Using GPU or CPU
    device = torch.device(config.device0)
    width_s, height_s = style_img_resize.size
    width_c, height_c = content_img_resize.size
    
    # 1*3*320*1024
    style_img   = utils.image_to_tensor(style_img_resize)[:3,:,:].unsqueeze(0).to(device, torch.float)
    content_img = utils.image_to_tensor(content_img_resize)[:3,:,:].unsqueeze(0).to(device, torch.float)
    scene_img   = utils.image_to_tensor(scene_img_crop)[:3, :, :].unsqueeze(0).to(device, torch.float)
    
    # print('content_img size: ', content_img.size())
    # utils.show_pic(style_img, 'style image')
    # utils.show_pic(content_img, 'content image')

    # -------------------------
    # Eval() means change the model in eval mode and requires_grad = False means 
    # the parameters of cnn are frozen.
    cnn = models.vgg19(pretrained=True).features.to(config.device0).eval()
    for param in cnn.parameters():
        param.requires_grad = False

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(config.device0)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(config.device0)

    # Two different initialization ways
    input_img = torch.randn(1, 3, height_c, width_c).to(config.device0)
    # input_img = content_img.clone()
    # print('input_img size: ', input_img.size())
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, scene_img,
                                style_mask_tensor, content_mask_tensor, car_mask_tensor, L)
    print('Style transfer completed')
    utils.save_pic(output, 'deep_style_tranfer')
    print()

    #--------------------------
    print('Postprocessing......')
    utils.post_process(output, content_image_path)
    print('Done!')




