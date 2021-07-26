import os
import sys
import PIL.Image as pil
from PIL import ImageOps
import numpy as np
import torch
from torchvision.transforms import transforms
sys.path.append(".")
from DeepPhotoStyle_pytorch.depth_model import import_depth_model
import random
from preprocessing import kitti_util

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_original_meshgrid(disp_size, original_size):
    rows, cols = original_size
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    crop_rows, crop_cols = disp_size
    left = (cols - crop_cols) // 2
    right = left + crop_cols
    top = rows - crop_rows
    bottom = rows
    h_range = slice(top, bottom)
    w_range = slice(left, right)
    c_cropped = c[h_range, w_range]
    r_cropped = r[h_range, w_range]
    assert c_cropped.shape == disp_size
    return c_cropped, r_cropped

def generate_point_cloud(disp_map, calib_path, output_path, max_height):
    calib = kitti_util.Calibration(calib_path)
    disp_map = (disp_map).astype(np.float32)
    # print(disp_map.shape)
    lidar = project_disp_to_points(calib, disp_map, max_height)
    # pad 1 in the indensity dimension
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)
    # print(lidar.shape)
    out_fn = output_path
    # print(lidar.shape)
    np.save(out_fn, lidar)
    return out_fn

def project_disp_to_points(calib, disp, max_high):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth = calib.f_u * baseline / (disp + 1. - mask)
    # rows, cols = depth.shape
    # c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    original_size = (370, 1226)
    c, r = get_original_meshgrid(depth.shape, original_size)
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

class AttackValidator():
    def __init__(self, root_path, car_name, adv_no, scene_name, depth_model) -> None:
        self.adv_car_img_path = os.path.join(root_path, "Adv_car", f"{car_name}_{adv_no}.png")
        self.ben_car_img_path = os.path.join(root_path, "Ben_car", f"{car_name}.png")
        self.car_mask_img_path = os.path.splitext(self.ben_car_img_path)[0] + '_CarMask.png'
        self.scene_img_path = os.path.join(root_path, "Scene",  f"{scene_name}.png")
        self.device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = depth_model
        self.calib_path = "/data/cheng443/kitti/object/training/calib/003086.txt"
        self.pc_dir = os.path.join(root_path, 'PointCloud')
        self.ben_pc_path = os.path.join(root_path, 'PointCloud', f'{car_name}_{adv_no}_ben.npy')
        self.adv_pc_path = os.path.join(root_path, 'PointCloud', f'{car_name}_{adv_no}_adv.npy')
        self.scene_pc_path = os.path.join(root_path, 'PointCloud', f'{car_name}_{adv_no}_sce.npy')
        self.load_imgs()
        setup_seed(17)
    
    def load_imgs(self):
        ben_car_img = pil.open(self.ben_car_img_path)
        adv_car_img = pil.open(self.adv_car_img_path)
        scene_car_img = pil.open(self.scene_img_path)
        self.ben_car_tensor = transforms.ToTensor()(ben_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float)
        self.adv_car_tensor = transforms.ToTensor()(adv_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float)
        self.scene_tensor = transforms.ToTensor()(scene_car_img)[:3,:,:].unsqueeze(0).to(self.device0, torch.float)
        img_mask = ImageOps.grayscale(pil.open(self.car_mask_img_path))
        img_mask_np = np.array(img_mask) / 255.0
        img_mask_np[img_mask_np > 0.5] = 1
        img_mask_np[img_mask_np <= 0.5] = 0
        img_mask_np = img_mask_np.astype(int)
        self.car_mask_tensor = torch.from_numpy(img_mask_np).unsqueeze(0).float().to(self.device0).requires_grad_(False)
        print("ben car size: {} \n adv car size: {} \n scene image size: {} \n car mask size: {}".format(
            self.ben_car_tensor.size(), self.adv_car_tensor.size(), self.scene_tensor.size(), self.car_mask_tensor.size()))
    
    def get_depth_data(self):
        self.attach_car_to_scene(1)
        with torch.no_grad():
            adv_scene_disp = self.depth_model(self.adv_scene_tensor).squeeze().cpu().numpy()
            ben_scene_disp = self.depth_model(self.ben_scene_tensor).squeeze().cpu().numpy()
            scene_disp = self.depth_model(self.scene_tensor).squeeze().cpu().numpy()
            generate_point_cloud(adv_scene_disp, self.calib_path, self.adv_pc_path, 3)
            generate_point_cloud(ben_scene_disp, self.calib_path, self.ben_pc_path, 3)
            generate_point_cloud(scene_disp, self.calib_path, self.scene_pc_path, 3)
        return

    def attach_car_to_scene(self, batch_size):
        """
        Attach the car image and adversarial car image to the given scene with random position. 
        The scene could have multiple images (batch size > 1)
        scene_img: B * C * H * W
        car_img:   1 * C * H * W
        car_mask:      1 * H * W
        """
        scene_img = self.scene_tensor
        adv_car_img = self.adv_car_tensor
        car_img = self.ben_car_tensor
        car_mask = self.car_mask_tensor
        _, _, H, W = adv_car_img.size()
        if scene_img.size()[0] == batch_size:
            adv_scene = scene_img.clone()
            car_scene = scene_img.clone()
        else:
            adv_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
            car_scene = torch.cat(batch_size * [scene_img.clone()], dim=0)
        scene_car_mask = torch.zeros(adv_scene.size()).float().to(self.device0)
        
        B_Sce, _, H_Sce, W_Sce = adv_scene.size()
        
        for idx_Bat in range(B_Sce):
            # scale = 0.7 # 600 -- 0.4, 300 -- 0.7
            scale_upper = 0.5
            scale_lower = 0.3
            scale = (scale_upper - scale_lower) * torch.rand(1) + scale_lower
            # Do some transformation on the adv_car_img together with car_mask
            trans_seq = transforms.Compose([ 
                # transforms.RandomRotation(degrees=3),
                transforms.Resize([int(H * scale), int(W * scale)])
                ])
            # trans_seq_color = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1)

            # adv_car_img_trans = trans_seq_color(trans_seq(adv_car_img)).squeeze(0)
            # car_img_trans = trans_seq_color(trans_seq(car_img)).squeeze(0)

            adv_car_img_trans = trans_seq(adv_car_img).squeeze(0)
            car_img_trans = trans_seq(car_img).squeeze(0)
            
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

        self.adv_scene_tensor = adv_scene
        self.ben_scene_tensor = car_scene
            

if __name__ == '__main__':
    generated_root_path = "/home/cheng443/projects/Monodepth/monodepth2_bob/pseudo_lidar/figures/GeneratedAtks/"
    car_name = 'BMW'
    adv_no = '001'
    scene_name = '0000000090'
    depth_model = import_depth_model((1024, 320), 'monodepth2').to(torch.device("cuda")).eval()
    validator = AttackValidator(generated_root_path, car_name, adv_no, scene_name, depth_model)
    validator.get_depth_data()