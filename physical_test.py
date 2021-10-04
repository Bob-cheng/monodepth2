#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image as pil
from numpy.core.fromnumeric import sort
from torchvision import transforms
import torch
from DeepPhotoStyle_pytorch.depth_model import import_depth_model
import sys
sys.path.append('./pseudo_lidar')
from pseudo_lidar.attack_validator_Bob import generate_point_cloud
import os
import glob

#%% 
def resize_and_crop_img(img, size_WH, bottom_gap, side_crop):
    original_w, original_h = img.size
    img = img.crop((side_crop[0], 0, original_w-side_crop[1], original_h))
    original_w, original_h = img.size
    scale = 1024 / original_w
    img = img.resize((1024, int(original_h * scale)))
    now_w, now_h = img.size
    left = 0
    bottom = now_h - bottom_gap
    top = bottom - size_WH[1]
    img = img.crop((0, top, size_WH[0], bottom))
    return img

def read_scene_img(img_path, bottom_gap, side_crop):
    img1_origin = pil.open(img_path).convert('RGB')
    img1_resize = resize_and_crop_img(img1_origin, scene_size, bottom_gap, side_crop)
    img1_tensor = transforms.ToTensor()(img1_resize).unsqueeze(0).to(torch.device("cuda"))
    return img1_tensor, img1_resize

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

def process_dir(depth_model, input_dir, proc_range=None):
    if not os.path.isdir(input_dir):
        print("please give a diretory.")
        return
    output_dir = input_dir + '_Processed'
    os.makedirs(output_dir+'/Disp', exist_ok=True)
    os.makedirs(output_dir+'/RGB', exist_ok=True)
    os.makedirs(output_dir+'/Lidar', exist_ok=True)
    paths = glob.glob(os.path.join(input_dir, '*.{}'.format('png')))
    paths = sort(paths)
     #IMG_3595: b 150 side 400/200 || IMG_3603: b 250 side 400/200 || IMG_3606/IMG_3607/3609/3604/3608: b 200 side 100/500
     #IMG_3611: b 100 side 0/0
     #IMG_3625: b 280 side 350/350
     #IMG_3624: b 250 side 300/300
     #IMG_3626: b 250 side 350/350
    bottom_gap = 200           
    side_crop = [100, 500]
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if proc_range != None and idx > proc_range:
                break
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            input_image, img_resize = read_scene_img(image_path, bottom_gap, side_crop)
            # PREDICTION
            disp_resized_np = depth_model(input_image).squeeze().cpu().numpy()
            name_dest_im_lidar = os.path.join(output_dir,'Lidar', output_name)
            calib_path = "/data/cheng443/kitti/object/training/calib/003086.txt"
            generate_point_cloud(disp_resized_np, calib_path, name_dest_im_lidar, 2, is_sparse=True)

            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            
            name_dest_im = os.path.join(output_dir,'Disp', "{}.jpeg".format(output_name))
            name_dest_im_rgb = os.path.join(output_dir,'RGB', "rgb_{}.jpeg".format(output_name))
            # im.save(name_dest_im)
            # img_resize.save(name_dest_im_rgb)

            dst = pil.new('RGB', (im.width, im.height + im.height))
            dst.paste(img_resize, (0, 0))
            dst.paste(im, (0, img_resize.height))
            dst.save(name_dest_im_rgb)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
    print('-> Done!')
    # conver video to frames: ffmpeg -ss 04:00 -t 03:00 -i videofile.mpg -r 10 %04d.jpeg
    if proc_range == None:
        os.system("ffmpeg -r 10 -i {0}/rgb_%04d.jpeg -vcodec mpeg4 -y {0}/rgb_video.mp4".format(os.path.dirname(name_dest_im_rgb)))
        # os.system("ffmpeg -r 10 -i {0}/%04d.jpeg -vcodec mpeg4 -y {0}/disp_video.mp4".format(os.path.dirname(name_dest_im)))

#%%
scene_size = (1024, 320)
depth_model = import_depth_model(scene_size).to(torch.device("cuda")).eval()

for param in depth_model.parameters():
    param.requires_grad = False

#%%
input_dir = "/data/cheng443/depth_atk/videos/10-01-2021/IMG_3604"
process_dir(depth_model, input_dir, proc_range=1)

#%%
exit(0)
img2_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/physical_test/09-30-2021/P0_D3.png"
img1_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/physical_test/09-30-2021/P2_D3.png"

bottom_gap1 = 170
bottom_gap2 = 170

side_crop = [0, 0]

img1, _ = read_scene_img(img1_path, bottom_gap1, side_crop)
img2, _ = read_scene_img(img2_path, bottom_gap2, side_crop)

img1_extsplit = os.path.splitext(img1_path)
compare_img_path = img1_extsplit[0] + "_compare.jpg"

eval_depth_diff(img1, img2, depth_model, compare_img_path)


#%%
