# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from builtins import print
import open3d as o3d
# from open3d.web_visualizer import draw

import os
from open3d._ml3d.vis import boundingbox
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import numpy as np
from open3d.ml.torch import pipelines
from open3d.ml.vis import LabelLUT

import matplotlib.pyplot as plt

def load_parameters(ckpt_folder, pipeline):
    os.makedirs(ckpt_folder, exist_ok=True)
    # ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
    ckpt_path = ckpt_folder + "ckpt_00070.pth"
    pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
    if not os.path.exists(ckpt_path):
        print('check point not found, download online')
        cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
        os.system(cmd)

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)
    return pipeline

def make_obj_pred(pipeline, pc_np):
    data = {'point': pc_np}
    result = pipeline.run_inference(data)[0]
    return result

def get_pointpillars_pipeline(pretrain=True):
    cfg_file = "/home/cheng443/projects/Monodepth/Monodepth2_official/Open3D_ML/ml3d/configs/pointpillars_kitti.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.PointPillars(**cfg.model)
    dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)
    if pretrain:
        # pipeline = load_parameters('/home/cheng443/projects/Monodepth/monodepth2_bob/Open3D_ML/logs', pipeline)
        pipeline = load_parameters('/data/cheng443/open3d_ml/logs/PointPillars_KITTI_torch/checkpoint/', pipeline)
    return pipeline, model, dataset

def visualize_dataset(dataset, split, indices, lut=None):
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = ml3d.vis.Visualizer()
    if lut != None:
        vis.set_lut("labels", lut)
        vis.set_lut("pred", lut)
    vis.visualize_dataset(dataset, split, indices)

def visualize_frame(i, pc_np, bboxes=None, bboxes_array=None, lut=None):
    vis_data = []
    for i, pc in enumerate(pc_np):
        if bboxes_array == None:
            vis_data.append({
                'name': "{:0>5d}".format(i),
                'points': pc
            })
        else:
            vis_data.append({
                'name': "{:0>5d}".format(i),
                'points': pc,
                'bounding_boxes': bboxes_array[i]
            })
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = ml3d.vis.Visualizer()
    if lut != None:
        vis.set_lut("labels", lut)
        vis.set_lut("pred", lut)
    vis.visualize(vis_data, bounding_boxes=bboxes)

def visulize_data(data, lut=None):
    o3d.visualization.webrtc_server.enable_webrtc()
    vis = ml3d.vis.Visualizer()
    if lut != None:
        vis.set_lut("labels", lut)
        vis.set_lut("pred", lut)
    if len(data) == 1:
        vis.visualize(data, bounding_boxes=data[0]['bounding_boxes'])
    else:
        vis.visualize(data)
    

def draw_z_histogram(pc_np, name):
    plt.figure()
    n, bins, patches = plt.hist(pc_np[:, 2], 30, density=True, alpha=0.75)
    plt.xlabel('z-axis')
    plt.ylabel('numbers')
    plt.xlim([-2, 0])
    plt.title('Histogram of Z')
    plt.grid(True)
    plt.savefig(f'temp_{name}.png')

def load_pc_np(file_path):
    pc = np.load(file_path).astype(np.float32) / 1000
    pc[:, 2] = pc[:, 2] - 0.73
    return pc

if __name__ == '__main__':
    pipeline, model, dataset = get_pointpillars_pipeline()

    # use data set
    training_split = dataset.get_split("training")
    i= 98 # frame index
    data = training_split.get_data(i)
    points =  data['point']
    boxes_gt = data['bounding_boxes']
    # result = pipeline.run_inference(data)[0]
    result = make_obj_pred(pipeline, points)
    bboxes = []
    bboxes.extend(boxes_gt)
    bboxes.extend(result)
    # visualize_frame(i, points,boundingbox=bboxes)

    # use customized data
    adv_scene_pc_path = '/home/cheng443/projects/Monodepth/monodepth2_bob/pseudo_lidar/visualization/eval_adv_scene_disp.npy'
    ben_scene_pc_path = '/home/cheng443/projects/Monodepth/monodepth2_bob/pseudo_lidar/visualization/eval_car_scene_disp.npy'
    orig_scene_pc_path = '/home/cheng443/projects/Monodepth/monodepth2_bob/pseudo_lidar/visualization/eval_scene_disp.npy'
    adv_pc = load_pc_np(adv_scene_pc_path)
    ben_pc = load_pc_np(ben_scene_pc_path)
    ori_pc = load_pc_np(orig_scene_pc_path)
    result = make_obj_pred(pipeline, adv_pc)
    result = make_obj_pred(pipeline, ben_pc)
    # visualize_frame(0, ben_pc, bboxes=result)
    # visualize_frame(0, [ben_pc, points])

    # the histogram of the data
    # draw_z_histogram(ben_pc, 'ben_pc')
    # draw_z_histogram(points, 'dataset')

    visualize_frame(0, [adv_pc, points], bboxes=result)
    visualize_frame(0, [ben_pc, points], bboxes=result)


    # lut = LabelLUT()
    # for key in sorted(dataset.label_to_names.keys()):
    #     lut.add_label(dataset.label_to_names[key], key)
    




    




# %%
# evaluate performance on the test set; this will write logs to './logs'.
# pipeline.run_test()


