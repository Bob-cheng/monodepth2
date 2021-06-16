#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image as pil
from torchvision import transforms
import torch
from depth_model import import_depth_model
import os

#%% 
def resize_and_crop_img(img, size_WH, bottom_gap):
    original_w, original_h = img.size
    scale = 1024 / original_w
    img = img.resize((1024, int(original_h * scale)))
    now_w, now_h = img.size
    left = 0
    bottom = now_h - bottom_gap
    top = bottom - size_WH[1]
    img = img.crop((0, top, size_WH[0], bottom))
    return img

def read_scene_img(img_path, bottom_gap):
    img1 = pil.open(img_path).convert('RGB')
    img1 = resize_and_crop_img(img1, scene_size, bottom_gap)
    img1 = transforms.ToTensor()(img1).unsqueeze(0).to(torch.device("cuda"))
    return img1

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
    # fig.canvas.draw()
    plt.savefig(filename)
    # pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    # plt.close()
    # return pil_image, disp1, disp2
#%%
scene_size = (1024, 320)
depth_model = import_depth_model(scene_size).to(torch.device("cuda")).eval()

#%%
for param in depth_model.parameters():
    param.requires_grad = False
img1_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/physical_test/S1_C1_P1.jpg"
img2_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/physical_test/S1_C1_P1_adv.jpg"

bottom_gap = 140

img1 = read_scene_img(img1_path, bottom_gap)
img2 = read_scene_img(img2_path, bottom_gap)

img1_extsplit = os.path.splitext(img1_path)
compare_img_path = img1_extsplit[0] + "_compare.jpg"

eval_depth_diff(img1, img2, depth_model, compare_img_path)

# compare_img.save(compare_img_path)
# plt.imshow(compare_img)


#%%
