#%%
from PIL import Image as pil
import os
import math

folder_name = 'BMW_mono_rob_disp_-2_X'

patch_image_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/generated_patch/{}/full_image.png".format(folder_name)
parts = [2,3]

#%%

def segment_image(image_path, parts):
    dir_name = os.path.dirname(patch_image_path)
    img = pil.open(patch_image_path)
    w,h = img.size
    split_w = math.ceil( w / parts[1] )
    split_h = math.ceil( h / parts[0] )
    for i in range(parts[0]):
        for j in range(parts[1]):
            left = j * split_w
            right = min(left + split_w, w)
            top = i * split_h
            bottom = min(top + split_h, h)
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(os.path.join(dir_name, "parts_{}{}.png".format(i, j)))

segment_image(patch_image_path, parts)

