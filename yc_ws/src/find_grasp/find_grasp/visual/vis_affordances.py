'''
    Visual Graspness, Sealness, Wrenchness, Flatness
    RGB Table website: https://tool.oschina.net/commons?type=3
'''

import os
import sys
BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(BASEDIR, 'utils')
sys.path.append(BASEDIR)
sys.path.append(UTILS)

import argparse
import colorsys
import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from dataset.graspnet_dataset import GraspNetDataset, load_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/zhy/Data2TB1/GraspNet1B')
parser.add_argument('--suction_labels_root', default='/media/zhy/Data2TB1/GraspNet1B/suction_additional_label')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=961200, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--affordance', default='origin', choices=['graspness', 'sealness', 'wrenchness', \
                                                                  'flatness', 'suctioness', 'origin', 'bar'])
cfgs = parser.parse_args()


def load_data(idx):
    grasp_labels = load_grasp_labels(cfgs.dataset_root)
    DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, suction_labels_root=cfgs.suction_labels_root,
                                camera=cfgs.camera, split='train', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=False, load_label=True)

    return DATASET[idx]


def scale_brightness(rgb, scale_l, scale_s):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    r, g, b = colorsys.hls_to_rgb(h, l * scale_l, s * scale_s)
    return np.array([r, g, b]) # (1, 3)


if __name__ == '__main__':

    vis_idx = 10 * 256 + 0
    end_points = load_data(vis_idx)

    # graspness, sealness, wrenchness
    cloud = end_points['point_clouds']
    seg = end_points['objectness_label'].reshape(-1)
    graspness = end_points['graspness_label'].reshape(-1)
    # sealness = end_points['sealness_label'].reshape(-1)
    # wrenchness = end_points['wrenchness_label'].reshape(-1)
    # flatness = end_points['flatness_label'].reshape(-1)
    # suctioness = np.clip(sealness, 0, 1) * wrenchness
    
    # remove background
    graspness[seg == 0] = 0.
    # sealness[seg == 0] = 0.
    # wrenchness[seg == 0] = 0.
    # # flatness[seg == 0] = 0.
    # suctioness[seg == 0] = 0.

    # color brightness
    color_scene = end_points['color'] # 0~1
    color_scene_scaled = np.zeros((cfgs.num_point, 3))
    if cfgs.affordance == 'graspness':
        color_scene[seg == 0] *= 0.6
        for i in range(cfgs.num_point):
            if graspness[i] > 0.1:
                color_scene_scaled[i] = np.asarray([0., 1., 0.]) * graspness[i] * 2.8
            else:
                color_scene_scaled[i] = color_scene[i] * 0.75
    # elif cfgs.affordance == 'sealness':
    #     color_scene[seg == 0] *= 0.6
    #     for i in range(cfgs.num_point):
    #         if sealness[i] > 0.1:
    #             color_scene_scaled[i] = np.asarray([0., 0., 1.]) * sealness[i]
    #         else:
    #             color_scene_scaled[i] = color_scene[i] * 0.75
    # elif cfgs.affordance == 'wrenchness':
    #     for i in range(cfgs.num_point):
    #         if wrenchness[i] > 0.2:
    #             color_scene_scaled[i] = scale_brightness(np.asarray([0., 255., 0.])/256, scale_l=1, scale_s=wrenchness[i])
    #         else:
    #             color_scene_scaled[i] = scale_brightness(list(color_scene[i]), scale_l=1, scale_s=0.6)
    # elif cfgs.affordance == 'flatness':
    #     for i in range(cfgs.num_point):
    #         if flatness[i] > 0.3:
    #             color_scene_scaled[i] = scale_brightness(np.asarray([0., 255., 0.])/256, scale_l=1, scale_s=flatness[i])
            # else:
            #     color_scene_scaled[i] = scale_brightness(list(color_scene[i]), scale_l=1, scale_s=0.6)
    # elif cfgs.affordance == 'suctioness':
    #     color_scene[seg == 0] *= 0.6
    #     for i in range(cfgs.num_point):
    #         if suctioness[i] > 0.2:
    #             color_scene_scaled[i] = np.asarray([0., 1., 0.]) * suctioness[i] * 1.5
    #         else:
    #             color_scene_scaled[i] = color_scene[i] * 0.75
    elif cfgs.affordance == 'origin':
        color_scene_scaled = color_scene
    elif cfgs.affordance == 'bar':
        # vis blue-red gradient color map
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("Value", ["black", "lime"])
        # Make a normalizer that will map the time values from [start_time,end_time+1] -> [0,1]
        cnorm = mcol.Normalize(vmin=0, vmax=1)
        # Turn these into an object that can be used to map time values to colors and can be passed to plt.colorbar()
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        cpick.set_array([])
        plt.colorbar(cpick, label="Value")
        plt.show()


    # opend3d visualization
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud)
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_scene_scaled)
    o3d.visualization.draw_geometries([o3d_cloud], width=1920, height=1080, zoom=0.68,
                                    front=[-0.00052776547683580741, -0.075447714894348428, -0.99714961955607373],
                                    lookat=[-0.060554241206703469, -0.048188423178108473, 0.41146838455222845],
                                    up=[-0.015415495828539348, 0.99703188508616503, -0.075430647683077681],
                                    )

