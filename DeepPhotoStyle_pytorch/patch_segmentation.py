#%%
from PIL import Image as pil
import os
import math



patch_image_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/generated_patch/BMW_VW/full_image.png"
parts = 5

#%%

def segment_image(image_path, parts : int):
    dir_name = os.path.dirname(patch_image_path)
    img = pil.open(patch_image_path)
    w,h = img.size
    split_w = math.ceil( w / parts )
    for i in range(parts):
        left = i * split_w
        right = min(left + split_w, w)
        top = 0
        bottom = h
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(os.path.join(dir_name, "parts_{}.png".format(i)))

segment_image(patch_image_path, parts)

