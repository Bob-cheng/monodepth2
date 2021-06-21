import torch
from depth_model import DepthModelWrapper, import_depth_model
import torch.nn.functional as F

class CarRecognition(torch.nn.Module):
    def __init__(self, depth_module: DepthModelWrapper, categories: int):
        super(CarRecognition, self).__init__()
        self.depth_encoder = depth_module.encoder
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        
        self.fc = torch.nn.Linear(512, categories)

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
    depth_model = import_depth_model((1024, 320)).to(torch.device("cuda")).eval()
    recog_model = CarRecognition(depth_model, 2).to(torch.device("cuda"))
    img = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/gen_img/scene/0000000017.png').convert('RGB')
    assert img.size == (1024, 320)
    img = transforms.ToTensor()(img).unsqueeze(0).to(torch.device("cuda"))
    with torch.no_grad():
        disp = recog_model(img)
