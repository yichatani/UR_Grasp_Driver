"""
    t-SNE visualization of the output from point cloud backbone
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import sys
import yaml
import argparse
import numpy as np
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
parser.add_argument('--model_epoch', default=35)
parser.add_argument('--log_dir', default='../logs/log')
parser.add_argument('--dataset_root', default='/media/zhy/Data2TB1/GraspNet1B')
parser.add_argument('--suction_labels_root', default='/media/wws/Elements/Graspnet_1B/suction_additional_label')
parser.add_argument('--split', default='train', choices=['test', 'test_seen', 'test_similar', 'test_novel'])

# default in shell
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 15000]')
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

    features = end_points['features'] # (1, C, N)
    graspness = end_points['graspness_score'] # (1, N)
    seg = end_points['objectness_label'] # (1, N)

    features = features.squeeze(0).permute(1, 0).detach().cpu().numpy() # (N, C)
    graspness = graspness.squeeze(0).detach().cpu().numpy() # (N,)
    seg = seg.squeeze(0).detach().cpu().numpy() # (N,)

    features = features[seg != 0]
    graspness = graspness[seg != 0]

    return features, graspness


if __name__ == '__main__':

    features, graspness = load_data(split='train')
    graspness = (graspness - graspness.min()) / (graspness.max() - graspness.min())
    T_SNE = TSNE(n_components=2, learning_rate='auto', init='pca')
    t_sne = T_SNE.fit_transform(features)

    # positive index
    pos_idx = np.where(graspness > 0.3)[0]
    neg_idx = np.where(graspness < 0.05)[0]
    print('pos_idx: ', pos_idx.shape)
    print('neg_idx: ', neg_idx.shape)
    if neg_idx.shape[0] > pos_idx.shape[0]:
        neg_idx = np.random.choice(neg_idx, size=pos_idx.shape[0])
    print('neg_idx: ', neg_idx.shape)

    # Scatter plot
    plt.scatter(t_sne[pos_idx, 0], t_sne[pos_idx, 1], c='r', cmap='hot', alpha=0.8)
    plt.scatter(t_sne[neg_idx, 0], t_sne[neg_idx, 1], c='b', cmap='hot', alpha=0.2)

    # color = np.zeros((t_sne.shape[0], 3))
    # for i in range(t_sne.shape[0]):
    #     color[i] = (graspness[i], 0.3, 1 - graspness[i])
    # plt.scatter(t_sne[:, 0], t_sne[:, 1], c=color, cmap='hot', alpha=0.6)

    # Display the plot
    plt.show()