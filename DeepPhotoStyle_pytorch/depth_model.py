#%%
import os
import sys
import torch
import torch.nn
sys.path.append("..")
# from .. import networks # for lint perpose
import networks

depth_model_dir = os.path.join(os.path.dirname(os.getcwd()), 'models')
# print(depth_model_dir)

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def import_depth_model(scene_size, model_type='monodepth2'):
    """
    import different depth model to attack:
    possible choices: monodepth2, depthhints
    """
    if scene_size == (1024, 320):
        model_name = 'mono+stereo_1024x320'
    else:
        raise RuntimeError("scene size undefined!")
    model_path = os.path.join(depth_model_dir, model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    depth_model = DepthModelWrapper(encoder, depth_decoder)
    return depth_model


#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image as pil
    from torchvision import transforms
    depth_model = import_depth_model((1024, 320)).to(torch.device("cuda")).eval()
    img = pil.open('/home/cheng443/projects/Monodepth/Monodepth2_official/DeepPhotoStyle_pytorch/asset/gen_img/scene/0000000017.png').convert('RGB')
    assert img.size == (1024, 320)
    img = transforms.ToTensor()(img).unsqueeze(0).to(torch.device("cuda"))
    with torch.no_grad():
        disp = depth_model(img)
        print(disp.size())
        disp_np = disp.squeeze().cpu().numpy()
    
    vmax = np.percentile(disp_np, 95)
    plt.figure(figsize=(5,5))
    plt.imshow(disp_np, cmap='magma', vmax=vmax)
    plt.title('Disparity')
    plt.axis('off')
    
# %%
