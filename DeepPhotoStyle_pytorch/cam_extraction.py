import torch
import os
from torch import nn
from depth_model import DepthModelWrapper, import_depth_model
import torch.nn.functional as F
import numpy as np
import config

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def save_model(log_name, model, optimizer):
    model_dir = os.path.join(os.path.abspath(os.curdir), 'CAM_model')
    model_name = '{}_pretrained.pt'.format(log_name)
    model_path = os.path.join(model_dir, model_name)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)
    print('Model saved at: ', model_path)

class CarRecognition(torch.nn.Module):
    def __init__(self, depth_module: DepthModelWrapper, categories: int):
        super(CarRecognition, self).__init__()
        self.depth_encoder = depth_module.encoder
        self.fc = torch.nn.Linear(512, categories)
        
        # init weights  
        nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()

    def forward(self, img):
        features = self.depth_encoder(img)
        global_avg_pooling = features[-1].mean([2,3])
        prob = F.softmax(self.fc(global_avg_pooling), dim=1)
        return prob
        # for t in features:
        #     print(t.size())
        # print(global_avg_pooling.size())
        # print(prob.size())

if __name__ == "__main__":
    from PIL import Image as pil
    from torchvision import transforms
    depth_model = import_depth_model((1024, 320)).to(config.device0).eval()
    recog_model = CarRecognition(depth_model, 2).to(config.device0)
    img_rgb = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/gen_img/scene/0000000009.png').convert('RGB')
    assert img_rgb.size == (1024, 320)
    img = transforms.ToTensor()(img_rgb).unsqueeze(0).to(config.device0)
    model_path = "/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/CAM_model/Jun22_18-15-13_trojai_pretrained.pt"
    checkpoint = torch.load(model_path, map_location=config.device0)
    recog_model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        prob = recog_model(img)
        print('predict output: ', prob)

    target_layer  = recog_model.depth_encoder.encoder.layer4[-1]
    # print(type(target_layer))
    cam = GradCAM(model=recog_model, target_layer=target_layer, use_cuda=True)
    target_category  = 1
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img, target_category=target_category)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.asarray(img_rgb)/255.0, grayscale_cam)
    img_cam = pil.fromarray(np.uint8(visualization))
    img_cam.save('tem_img_cam.png')

    
