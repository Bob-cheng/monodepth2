#%%
import os
import PIL.Image as pil
from PIL import ImageOps
import numpy as np

src_content_path = os.path.join(os.getcwd(), 'asset', 'src_img', 'content')
src_style_path = os.path.join(os.getcwd(), 'asset', 'src_img', 'style')
src_scene_path = os.path.join(os.getcwd(), 'asset', 'src_img', 'scene')
gen_content_path = os.path.join(os.getcwd(), 'asset', 'gen_img', 'content')
gen_style_path = os.path.join(os.getcwd(), 'asset', 'gen_img', 'style')
gen_scene_path = os.path.join(os.getcwd(), 'asset', 'gen_img', 'scene')

car_img_width = 300
scene_size = (1024, 320) # width, height

#%%
def prepare_dir():
    global src_content_path
    global src_style_path
    global gen_content_path
    global gen_style_path
    if not os.path.exists(gen_content_path):
        os.makedirs(gen_content_path)
    if not os.path.exists(gen_style_path):
        os.makedirs(gen_style_path)
    if not os.path.exists(gen_scene_path):
        os.makedirs(gen_scene_path)
    
def process_img(img_name, output_w, image_type: str):
    if image_type == 'style':
        img_path = os.path.join(src_style_path, img_name)
        img_out_path = os.path.join(gen_style_path, img_name)
    elif image_type == 'content':
        img_path = os.path.join(src_content_path, img_name)
        img_out_path = os.path.join(gen_content_path, img_name)
    if not os.path.exists(img_path):
        raise RuntimeError("image '%s' doesn't exist" % img_path)
    style_img = pil.open(img_path)
    original_w, original_h = style_img.size
    print("Image original size (w, h): (%d, %d)" % (original_w, original_h))

    output_h = int(output_w / original_w * original_h)
    style_img_resize = style_img.resize((output_w, output_h))
    style_img_resize.save(img_out_path)
    print("Output image size", style_img_resize.size)
    return style_img_resize, output_w, output_h

def process_mask(mask_name, output_w, output_h, image_type: str):
    if image_type == 'style':
        mask_path = os.path.join(src_style_path, mask_name)
        mask_out_path = os.path.join(gen_style_path, mask_name)
    elif image_type == 'content':
        mask_path = os.path.join(src_content_path, mask_name)
        mask_out_path = os.path.join(gen_content_path, mask_name)
    if not os.path.exists(mask_path):
        img_mask_np = np.ones((output_h, output_w), dtype=int)
    else:
        img_mask = ImageOps.grayscale(pil.open(mask_path))
        img_mask_np = np.array(img_mask.resize((output_w, output_h)))/255.0
        img_mask_np[img_mask_np > 0.5] = 1
        img_mask_np[img_mask_np <= 0.5] = 0
        img_mask_np = img_mask_np.astype(int)
    pil.fromarray((img_mask_np*255).astype(np.uint8), 'L').save(mask_out_path)
    return img_mask_np

def process_style_img(img_name):
    ext_split = os.path.splitext(img_name)
    style_img_resize, w, h = process_img(img_name, car_img_width, 'style')
    img_mask_np = process_mask(ext_split[0] + '_StyleMask' + ext_split[1], w, h, 'style')
    assert style_img_resize.size[::-1] == img_mask_np.shape
    return style_img_resize, img_mask_np
    
def process_content_img(img_name):
    ext_split = os.path.splitext(img_name)
    content_img_resize, w, h = process_img(img_name, car_img_width, 'content')
    car_mask_np = process_mask(ext_split[0] + '_CarMask' + ext_split[1], w, h, 'content')
    paint_mask_np = process_mask(ext_split[0] + '_PaintMask' + ext_split[1], w, h, 'content')
    assert content_img_resize.size[::-1] == car_mask_np.shape
    return content_img_resize, car_mask_np, paint_mask_np

def process_scene_img(img_name):
    scene_img = pil.open(os.path.join(src_scene_path, img_name))
    original_w, original_h = scene_img.size
    new_w, new_h = scene_size
    left = (original_w - new_w)//2
    right = left + new_w
    top = original_h - new_h
    bottom = original_h
    scene_img_crop = scene_img.crop((left, top, right, bottom))
    assert scene_size == scene_img_crop.size
    scene_img_crop.save(os.path.join(gen_scene_path, img_name))
    return scene_img_crop



#%%
if __name__ == '__main__':
    prepare_dir()
    process_style_img("Dirty_Back.png")
    process_content_img("SUV_Back.png")
    process_scene_img("0000000248.png")
# %%