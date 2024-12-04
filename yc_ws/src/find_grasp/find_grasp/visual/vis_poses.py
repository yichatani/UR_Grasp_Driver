"""
    Visualization examples of predicted grasps and suctions on GraspNet-1Billion. 
"""

import os
import sys
import yaml
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
import scipy.io as scio
import json

import torch
from torch.utils.data import DataLoader
from graspnetAPI.grasp import Grasp, GraspGroup
from suctionnetAPI.utils.utils import plot_sucker
from suctionnetAPI.utils.rotation import viewpoint_to_matrix
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.graspnet import GraspNet, pred_grasp_decode, pred_suction_decode
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels


parser = argparse.ArgumentParser()
# variable in shell
parser.add_argument('--epoch', default=29)
parser.add_argument('--log_dir', default='../logs/log')
parser.add_argument('--batch_size', type=int, default=29, help='Batch Size during inference [default: 1]')
parser.add_argument('--num_points', type=int, default=100000, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--scene_id', type = int)
parser.add_argument('--anno_id', type = int)


cfgs = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Load_Real_Data(root_path, scene_id):
    '''
        Input: test_scenes path, test scene id
        output: a data dict
    '''

    # Init path
    rgb_path = os.path.join(root_path, 'rgb.png')
    depth_path = os.path.join(root_path, 'depth.png')
    camera_info_path = os.path.join(root_path, 'Intrinsic_Calibration.yaml')

    # read RGBD image
    color = np.array(Image.open(rgb_path)).astype(np.float32)
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    if color.shape[-1] == 4:
        color = color[:, :, :3]

    # camera intrinsics
    with open(camera_info_path, 'r') as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    intrinsics = np.asarray(camera_info['camera_matrix']['data']).reshape(3, 3)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    width, height = camera_info['image_width'], camera_info['image_height']
    s = 1000.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, s)

    # depth to point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    color = color / 255
    
    # sample points
    depth_mask = (depth > 0)
    cloud_masked = cloud[depth_mask]
    color_masked = color[depth_mask]

    if len(cloud_masked) >= cfgs.num_points:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_points - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # wrap into a dict
    ret_dict = {'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled, dtype=np.float32),
                'point_clouds': cloud_sampled.astype(np.float32),
                'color': color_sampled.astype(np.float32),
                
                }
    return ret_dict


def Load_Real_Data_SCENE(root_path, anno_id):
    '''
        Input: test_scenes path, test scene id
        output: a data dict
    '''

    # Init path
    path = anno_id + '.png'
    camera_path = anno_id + '.mat'
    print(path)
    rgb_path = os.path.join(root_path, 'rgb', path)
    depth_path = os.path.join(root_path, 'depth', path)
    camera_info_path = os.path.join(root_path, 'meta', camera_path)
    # read RGBD image
    color = np.array(Image.open(rgb_path)).astype(np.float32)
    depth = np.array(Image.open(depth_path)).astype(np.float32)
    if color.shape[-1] == 4:
        color = color[:, :, :3]

    # camera intrinsics
    meta = scio.loadmat(camera_info_path)
    intrinsics = meta['intrinsic_matrix']

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # width, height = meta['image_width'], meta['image_height']
    s = 1000.0
    factor_depth = meta['factor_depth']
    camera = CameraInfo(1280.0, 720.0, fx, fy, cx, cy, 1000)

    # depth to point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    color = color / 255
    
    # sample points
    depth_mask = (depth > 0)
    cloud_masked = cloud[depth_mask]
    color_masked = color[depth_mask]

    if len(cloud_masked) >= cfgs.num_points:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_points - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # wrap into a dict
    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'color': color_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled, dtype=np.float32),
                }
    return ret_dict


def Load_Real_Data_meta_dataset(scene_id, anno_id):

    # colorpath = "/home/zhy/Grasp_pointcloud/new_structure/Dataset/MetaGraspnet/scene" + str(scene_id) + '/' + str(anno_id) + "_rgb.png"
    # depthpath = "/home/zhy/Grasp_pointcloud/new_structure/Dataset/MetaGraspnet/scene" + str(scene_id) + '/' + str(anno_id) + "_depth.png"
    root = '/media/zhy/Data2TB1/MetaGraspNet/mnt/data1/data_ifl/scene'
    colorpath = root + str(scene_id) + '/' + str(anno_id) + "_rgb.png"
    depthpath = root + str(scene_id) + '/' + str(anno_id) + ".npz"
    # depthpath = "/home/zhy/Grasp_pointcloud/new_structure/mnt/data1/data_ifl_real/scene" + str(scene_id) + '/' + str(anno_id) + "_depth.png"

    with np.load(depthpath) as data:
        depth_np = data['depth']

    # inpait NaN values as zeros
    for i in range(depth_np.shape[0]):
        for j in range(depth_np.shape[1]):
            if np.isnan(depth_np[i,j]):
                depth_np[i,j] = 0.0
    depth = depth_np

    # read RGBD image
    color = np.array(Image.open(colorpath)).astype(np.float32)
    # depth = np.array(Image.open(depthpath)).astype(np.float32)
    # depth = depth[:, :, 0]

    with open(root + str(scene_id) + '/' + str(anno_id) + '_camera_params.json') as f:
        fil = json.load(f)
        fx, fy = fil['fx'], fil['fy']
        # width, height = fil['resolution']['width'], fil['resolution']['height']
        width, height = fil['resolution']['width'], fil['resolution']['height']
    factor_depth = 100


    # width, height = 640, 480
    camera_info = {"fx": 1784.49072265625, "fy": 1786.48681640625, "cx": 975.0308837890625, "cy": 598.6246337890625, "width": 1944, "height": 1200}
    camera = CameraInfo(width, height, fx, fy, width/2, height/2, factor_depth)

    # depth to point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    color = color / 255
    
    # sample points
    depth_mask = (depth > 0)
    cloud_masked = cloud[depth_mask]
    color_masked = color[depth_mask]

    if len(cloud_masked) >= cfgs.num_points:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_points - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # wrap into a dict
    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'color': color_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled, dtype=np.float32),
                }
    return ret_dict
def Load_Real_Data_meta_dataset_new(scene_id, anno_id):
    
    root = '/media/zhy/Data2TB1/MetaGraspNet/mnt/data1/data_ifl/scene'
    colorpath = root + str(scene_id) + '/' + str(anno_id) + "_rgb.png"
    depthpath = root + str(scene_id) + '/' + str(anno_id) + ".npz"
    # depthpath = "/home/zhy/Grasp_pointcloud/new_structure/mnt/data1/data_ifl_real/scene" + str(scene_id) + '/' + str(anno_id) + "_depth.png"

    with np.load(depthpath) as data:
        depth_np = data['depth']

    # inpait NaN values as zeros
    for i in range(depth_np.shape[0]):
        for j in range(depth_np.shape[1]):
            if np.isnan(depth_np[i,j]):
                depth_np[i,j] = 0.0
    depth = depth_np
    height, width = depth_np.shape
    depth_raw = o3d.geometry.Image(depth_np)
    color_raw = o3d.io.read_image(colorpath)

    # read RGBD image
    color = np.array(Image.open(colorpath)).astype(np.float32)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_raw, 
        depth=depth_raw,
        depth_scale=100, # important! cm -> m
        depth_trunc=1,
        convert_rgb_to_intensity=False)

    with open(root + str(scene_id) + '/' + str(anno_id) + '_camera_params.json') as json_file:
        f = json.load(json_file)
    fx = float(f['fx'])
    fy = float(f['fy'])


    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    camera_intrinsics.set_intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=width/2,
        cy=height/2)

    print("intrinsic matrix : \n", camera_intrinsics.intrinsic_matrix)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics)

    # Convert Open3D point cloud points to NumPy array
    points_np = np.asarray(pcd.points)

    # Convert to float32
    points_np = points_np.astype(np.float32)

    ret_dict = {'point_clouds': points_np,
                
                }
    return ret_dict

def inference(root_path, scene_id):
    # Load data
    dict_data = Load_Real_Data(root_path, scene_id)

    # convert to batch(GPU) for network inference
    batch_data = minkowski_collate_fn([dict_data])
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)

    # Load model config & Init the model
    with open('../models/model_config.yaml', 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    net = GraspNet(model_config, is_training=False)
    net.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(cfgs.log_dir, 'epoch_{}.tar'.format(cfgs.epoch))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))

    # inference
    net.eval()
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_grasp_decode(end_points)
        grasp_preds = grasp_preds[0].detach().cpu().numpy() # (M, 17)
        suction_preds = pred_suction_decode(end_points) # [M, 7]

    # add Affordance to grasp_preds, (M, 19)
    graspness_preds = end_points['scores_graspable'].reshape(-1, 1).detach().cpu().numpy()
    suctioness_preds = end_points['scores_suctionable'].reshape(-1, 1).detach().cpu().numpy()
    grasp_preds = np.concatenate((grasp_preds, graspness_preds, suctioness_preds), axis=1)

    return end_points, grasp_preds, suction_preds

def view_grasps(grasps, end_points):
    
    point_cloud = end_points['point_clouds']
    if point_cloud.shape[0] == 1:
        point_cloud = point_cloud.reshape(-1, 3) 
    point_color = end_points['color']
    if point_color.shape[0] == 1:
        point_color = point_color.reshape(-1, 3)
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    cloud_o3d.colors = o3d.utility.Vector3dVector(point_color)




    grasp_group = GraspGroup(grasps)
    # collision detection
    mfcdetector = ModelFreeCollisionDetector(point_cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.005, collision_thresh=0.001)


    grasp_group = grasp_group[~collision_mask]
    grasp_group.nms()
    grasp_group_len = len(grasp_group)
    


    # grasp_group.sort_by_affordance()
    grasp_group.sort_by_score()
    grasp_group_top  = grasp_group
    # grasp_group_top = grasp_group[:int(grasp_group_len/3)]
    # grasp_group_top = grasp_group_top.random_sample(30)
    # grasp_group_mid = grasp_group[int(grasp_group_len/3) : int(grasp_group_len/3*2)]
    # grasp_group_mid = grasp_group_mid.random_sample(20)
    # grasp_group_down = grasp_group[int(grasp_group_len/3*2) : grasp_group_len]
    # grasp_group_down = grasp_group_down.random_sample(0)
    # grasp_group = grasp_group_top.add(grasp_group_mid)
    # grasp_group = grasp_group.add(grasp_group_down)


    grippers = grasp_group.to_open3d_geometry_list()
    # for gripper in grippers:
        # gripper.scale(1000, center = gripper.get_center())
        # gripper.paint_uniform_color([1,0,0])
    # print("Gripper bounding box:", grippers[1].get_axis_aligned_bounding_box())


    o3d.visualization.draw_geometries([cloud_o3d, *grippers])




if __name__ == '__main__':

    scene_id = 0 # [120, 130, 140, 170]
    root_path = os.path.join(ROOT_DIR, 'test_scenes', '0140')
    end_points, grasp_preds, suction_preds = inference(root_path, scene_id) # (1024, 19)
    vis_type = 'grasp' # choices=['grasp', 'suction', 'bar']


    # scene pc & color
    point_cloud = end_points['point_clouds']
    point_color = end_points['color']
    point_cloud = point_cloud.squeeze(0).detach().cpu().numpy()
    point_color = point_color.squeeze(0).detach().cpu().numpy()
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    cloud_o3d.colors = o3d.utility.Vector3dVector(point_color)

    # vis top-50 grasp & suction poses
    if vis_type == 'grasp':
        ### grasp poses
        grasp_group = GraspGroup(grasp_preds)
        # collision detection
        mfcdetector = ModelFreeCollisionDetector(point_cloud, voxel_size=0.01)
        collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.5, collision_thresh=0.1)
        # # filter out table corner --- x: (-0.57, 0.18)
        # print('min: ', max(point_cloud[:, 0]))
        # exit()
        # poses_xyz = grasp_group.translations
        #
        # region_mask = poses_xyz
        # print('poses_xyz: ', poses_xyz)
        # exit()


        grasp_group = grasp_group[~collision_mask]
        grasp_group.nms()
        grasp_group_len = len(grasp_group)
        print('grasp_group len: ', grasp_group_len)

        # grasp_group.sort_by_affordance()
        grasp_group.sort_by_score()
        # grasp_group_top = grasp_group[:int(grasp_group_len/3)]
        # grasp_group_top = grasp_group_top.random_sample(30)
        # grasp_group_mid = grasp_group[int(grasp_group_len/3) : int(grasp_group_len/3*2)]
        # grasp_group_mid = grasp_group_mid.random_sample(20)
        # grasp_group_down = grasp_group[int(grasp_group_len/3*2) : grasp_group_len]
        # grasp_group_down = grasp_group_down.random_sample(0)
        # grasp_group = grasp_group_top.add(grasp_group_mid)
        # grasp_group = grasp_group.add(grasp_group_down)
        
        # ~/anaconda3/envs/graspnet/lib/python3.8/site-packages/graspnetAPI/grasp.py
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])
        # o3d.visualization.draw_geometries([cloud_o3d, *grippers], width=1280, height=720, zoom=0.94,
        #                                   front=[0.0022559312318344528, -0.27565991919135979, -0.96125257852745982],
        #                                   lookat=[-0.27096993769844963, -0.10005978214040652, 0.73612768448269661],
        #                                   up=[-0.022186405724433977, 0.9610046146489547, -0.27564087872528042],
        #                                   )

        # # save open3d snapshot
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(cloud_o3d)
        # vis.update_geometry(cloud_o3d)
        # for i in range(len(grippers)):
        #     vis.add_geometry(grippers[i])
        #     vis.update_geometry(grippers[i])
        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image('snapshot.png')
        # vis.destroy_window()


    if vis_type == 'suction':
        ### suction poses
        suction_points = suction_preds[:, 4:7]
        suction_normals = suction_preds[:, 1:4]
        suction_scores = suction_preds[:, 0]
        suction_group_len = suction_scores.shape[0]

        # visualize suctionable point and their normals
        suckers = []
        radius, height = 0.004, 0.05

        suction_group_top = suction_scores.argsort()[-int(suction_group_len/3):]
        suction_group_top = np.random.choice(suction_group_top, size=25)
        suction_group_mid = suction_scores.argsort()[-int(suction_group_len/3*2):-int(suction_group_len/3)]
        suction_group_mid = np.random.choice(suction_group_mid, size=25)
        suction_group_down = suction_scores.argsort()[:int(suction_group_len/3)]
        suction_group_down = np.random.choice(suction_group_down, size=25)

        point_inds = np.concatenate((suction_group_top, suction_group_mid, suction_group_down), axis=0)
        for point_ind in point_inds:
            target_point = suction_points[point_ind]
            target_normal = suction_normals[point_ind]
            target_score = suction_scores[point_ind]

            R = viewpoint_to_matrix(target_normal.reshape(-1))
            t = target_point
            # ~/anaconda3/envs/graspnet/lib/python3.8/site-packages/suctionnetAPI/utils/utils.py
            sucker = plot_sucker(R, t, target_score, radius, height)
            suckers.append(sucker)
        # o3d.visualization.draw_geometries([cloud_o3d, *suckers])
        o3d.visualization.draw_geometries([cloud_o3d, *suckers], width=1280, height=720, zoom=0.38,
                        front = [ -0.043067006037024846, -0.31040479322240361, -0.94962839960458423 ],
                        lookat = [ 0.091706350445747375, 0.077784041076898569, 0.52100000530481339 ],
                        up = [ -0.01688716848396896, 0.95060067379094393, -0.30995674299617609 ],
                        )


    # vis blue-red gradient color map
    if vis_type == 'bar':
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcol
        import matplotlib.cm as cm
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("Value", ["b", "r"])
        # Make a normalizer that will map the time values from [start_time,end_time+1] -> [0,1]
        cnorm = mcol.Normalize(vmin=0, vmax=1)
        # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        cpick.set_array([])
        plt.colorbar(cpick, label="Value")
        plt.show()
