"""
    3D attention map visualization
"""

import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import yaml
import colorsys
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from graspnet import GraspNet, pred_grasp_decode
from graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels

parser = argparse.ArgumentParser()
# variable in shell
parser.add_argument('--model_epoch', default=29)
parser.add_argument('--log_dir', default='../logs/log')
parser.add_argument('--dataset_root', default='/media/zhy/Data2TB1/GraspNet1B')
parser.add_argument('--suction_labels_root', default='/media/wws/Elements/Graspnet_1B/suction_additional_label')
parser.add_argument('--split', default='train', choices=['test', 'test_seen', 'test_similar', 'test_novel'])

# default in shell
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
cfgs = parser.parse_args()

# load model config
with open('../models/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def load_data(split):
    grasp_labels = load_grasp_labels(cfgs.dataset_root)
    DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, suction_labels_root=cfgs.suction_labels_root,
                                    camera=cfgs.camera, split='train', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                    remove_outlier=True, augment=False, load_label=True)
    scene_list = DATASET.scene_list()
    dataloader = DataLoader(DATASET, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    
    # Init the model
    net = GraspNet(model_config, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(cfgs.log_dir, 'epoch_{}.tar'.format(cfgs.model_epoch))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))

    net.eval()

    # batch data
    for i in range(10):
        batch_data = next(iter(dataloader))
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)

    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)

    cloud = end_points['point_clouds'] # (1, N, 3)
    color = end_points['color'] # (1, N, 3)
    seg = end_points['objectness_label'] # (1, N)
    features = end_points['features'] # (1, C, N)

    cloud = cloud.squeeze(0).detach().cpu().numpy() # (N, 3)
    color = color.squeeze(0).detach().cpu().numpy() # (N, 3)
    seg = seg.squeeze(0).detach().cpu().numpy() # (N,)
    features = features.squeeze(0).permute(1, 0).detach().cpu().numpy() # (N, C)
    features[seg == 0] = 0.

    return cloud, color, features


def postprocess(color, features):

    output = np.abs(features)
    output = np.sum(output, axis=-1).squeeze() # (N, )
    output = (output - output.min()) / (output.max() - output.min())
    top_idx = output.argsort()[-int(cfgs.num_point * 0.05):] # top 2% points
    color[top_idx] = np.asarray([1, 0, 0])

    return color


if __name__ == '__main__':

    cloud, color, features = load_data(split='train')
    color = postprocess(color, features)

    # opend3d visualization
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud)
    o3d_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([o3d_cloud])