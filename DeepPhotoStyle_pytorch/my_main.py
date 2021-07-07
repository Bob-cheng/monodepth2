import sys, socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt

import argparse

from tensorboardX import SummaryWriter
import datetime
# ------custom module----
import config
import utils
from image_preprocess import prepare_dir, process_content_img, process_style_img, process_scene_img, process_car_img
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
    ap.add_argument("-v", "--vehicle", required=True, type=str, help="The name of the vehicle image")
    ap.add_argument("-pm", "--paint-mask", required=True, type=str, help="The number of the paint mask, e.g. '01'/'02'/'03' ")
    ap.add_argument("--gpu", type=str, help="specify a GPU to use")

    ap.add_argument("--style-weight",   "-sw", default=1000000,  type=float, help="Style similarity weight")
    ap.add_argument("--content-weight", "-cw", default=100,      type=float, help="Content similarity weight")
    ap.add_argument("--tv-weight",      "-tw", default=0.0001,   type=float, help="Transform variant weight")
    ap.add_argument("--rl-weight",      "-rw", default=1,        type=float, help="Reality weight")
    ap.add_argument("--adv-weight",     "-aw", default=1000000,  type=float, help="Adversarial weight")
    ap.add_argument("--mask-weight", "-mw", default=1, type=float, help='weight for paint mask')
    ap.add_argument("--l1-weight", "-l1w", default=1, type=float, help="l1 loss weight for perterbation")
    ap.add_argument("--steps",  default=3000, type=int, help="total training steps")
    ap.add_argument("--learning-rate",  "-lr", default=1, type=float, help="leanring rate")
    ap.add_argument("--batch-size",     "-bs", default=1, type=int, help="optimization batch size")
    ap.add_argument("--l1-norm", dest='l1_norm', action='store_true', help="Wheather to use L1 Norm to find sensitive area")
    ap.add_argument("--random-scene", "-rs", action='store_true', help="Test whether we use different scene to train")

    args = vars(ap.parse_args())

    print(str(args))

    style_image_name = args["style_image"]
    content_image_name = args["content_image"]
    if args['gpu'] != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']


    #-------------------------
    prepare_dir()
    style_img_resize, style_mask_np = process_style_img(style_image_name)
    content_img_resize, content_mask_np = process_content_img(content_image_name)

    # the following could be converted to data loader
    car_img_resize, car_mask_np, paint_mask_np = process_car_img(args['vehicle'], paintMask_no=args['paint_mask'])
    scene_img_crop = process_scene_img('VW01.png')
    test_scene_img = process_scene_img('VW01.png')

    print('Computing Laplacian matrix of content image')
    content_image_path = os.path.join(gen_content_path, content_image_name)
    L = utils.compute_lap(content_image_path)
    print()


    width_s, height_s = style_img_resize.size
    width_c, height_c = content_img_resize.size

    style_mask_tensor   = torch.from_numpy(style_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    car_mask_tensor     = torch.from_numpy(car_mask_np  ).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    paint_mask_tensor = torch.from_numpy(paint_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)
    content_mask_tensor = torch.from_numpy(content_mask_np).unsqueeze(0).float().to(config.device0).requires_grad_(False)

    paint_mask_np_inf = np.arctanh((paint_mask_np - 0.5) * (2 - 1e-7))
    paint_mask_inf = torch.from_numpy(paint_mask_np_inf).unsqueeze(0).float().to(config.device0).requires_grad_(True)
    # paint_mask_inf = utils.from_mask_to_inf(paint_mask_tensor).detach().requires_grad_(True)

    # test
    # content_mask_tensor = car_mask_tensor

    # 1*3*320*1024
    style_img   = utils.image_to_tensor(style_img_resize)[:3,:,:].unsqueeze(0).to(config.device0, torch.float)
    content_img = utils.image_to_tensor(content_img_resize)[:3,:,:].unsqueeze(0).to(config.device0, torch.float)
    car_img = utils.image_to_tensor(car_img_resize)[:3,:,:].unsqueeze(0).to(config.device0, torch.float)
    scene_img   = utils.image_to_tensor(scene_img_crop)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)
    test_scene_img = utils.image_to_tensor(test_scene_img)[:3, :, :].unsqueeze(0).to(config.device0, torch.float)

    # Logger
    log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
    os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    logger.add_text('args/CLI_params', str(args), 0)

    logger.add_image('input/imgs/style_image',   style_img[0], 0)
    logger.add_image('input/imgs/car_img',       car_img[0], 0)
    logger.add_image('input/imgs/content_img',   content_img[0], 0)
    logger.add_image('input/masks/style_mask',    style_mask_tensor, 0)
    logger.add_image('input/masks/car_mask',      car_mask_tensor, 0)
    logger.add_image('input/masks/paint_mask',    paint_mask_tensor, 0)
    logger.add_image('input/masks/content_mask',  content_mask_tensor, 0)
    
    # print('Save each mask as an image for debugging')
    # for i in range(style_mask_tensor.shape[0]):
    #     utils.save_pic( torch.stack([style_mask_tensor[i, :, :], style_mask_tensor[i, :, :], style_mask_tensor[i, :, :]], dim=0), 
    #                                 'style_mask_' + str(i) )
    #     utils.save_pic( torch.stack([content_mask_tensor[i, :, :], content_mask_tensor[i, :, :], content_mask_tensor[i, :, :]], dim=0), 
    #                                 'content_mask_' + str(i) )

    # -------------------------
    # Eval() means change the model in eval mode and requires_grad = False means 
    # the parameters of cnn are frozen.
    cnn = models.vgg19(pretrained=True).features.to(config.device0).eval()
    for param in cnn.parameters():
        param.requires_grad = False

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(config.device0)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(config.device0)

    # Two different initialization ways
    if args['l1_norm']:
        input_img = content_img.clone()
    else:
        input_img = torch.randn(1, 3, height_c, width_c).to(config.device0)
    # 

    output, depth_model = run_style_transfer(logger, cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, car_img, scene_img, test_scene_img,
                                style_mask_tensor, content_mask_tensor, paint_mask_inf, car_mask_tensor, L,
                                args)
    print('Style transfer completed')
    # utils.save_pic(output, 'deep_style_tranfer')
    logger.add_image('Output/whole_texture_transfer', output[0], 0)
    print()

    # Evaluate with another new scene
    output = utils.texture_to_car_size(output, car_img.size())
    adv_car_output = output * paint_mask_tensor.unsqueeze(0) + car_img * (1-paint_mask_tensor.unsqueeze(0))
    adv_scene_out, car_scene_out, _ = attach_car_to_scene(test_scene_img, adv_car_output, car_img, car_mask_tensor, args["batch_size"])
    # utils.save_pic(adv_scene_out, f'adv_scene_output')
    
    utils.save_pic(adv_scene_out[[0]], f'adv_scene_output', log_dir=log_dir)
    utils.save_pic(car_scene_out[[0]], f'car_scene_output', log_dir=log_dir)
    utils.save_pic(adv_car_output[[0]], f'adv_car_output', log_dir=log_dir)

    logger.add_image('Output/Adv_scene', adv_scene_out[0], 0)
    logger.add_image('Output/Car_scene', car_scene_out[0], 0)
    logger.add_image('Output/Adv_car', adv_car_output[0], 0)
    # take the first image without squeeze dimension
    eval_img, car_scene_disp, adv_scene_disp = eval_depth_diff(car_scene_out[[0]], adv_scene_out[[0]], depth_model, 'depth_diff_result')
    scene_disp = depth_model(test_scene_img[[0]]).detach().cpu().squeeze().numpy() 
    np.save(os.path.join(log_dir, 'eval_car_scene_disp.npy'), car_scene_disp)
    np.save(os.path.join(log_dir, 'eval_adv_scene_disp.npy'), adv_scene_disp)
    np.save(os.path.join(log_dir, 'eval_scene_disp.npy'), scene_disp)

    logger.add_image('Output/Compare', utils.image_to_tensor(eval_img), 0)

    #--------------------------
    # print('Postprocessing......')
    # utils.post_process(output, content_image_path)
    print('Done!')




