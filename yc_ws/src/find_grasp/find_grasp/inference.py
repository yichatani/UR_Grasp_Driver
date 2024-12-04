"""
    Input: epoch_{epoch}.tar
    Inference:
        log_dir/dump_{epoch}_{split}
    Evaluate:
        log_dir/dump_{epoch}_{split}/ap_{epoch}_{split}.npy
"""
import os
import sys
import yaml
import time
import torch
import json
import argparse
from PIL import Image
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader
# from suctionnetAPI import SuctionNetEval
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from utils.data_utils import CameraInfo,create_point_cloud_from_depth_image

from models.graspnet import GraspNet, pred_grasp_decode, pred_suction_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels
from utils.collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
# variable in shell
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--dataset_root', default='/media/zhy/Data2TB1/GraspNet1B', required=False)
parser.add_argument('--suction_labels_root', default=None, required=False)
parser.add_argument('--log_dir', default=os.path.join(ROOT_DIR + '/logs/log/combined'), required=False)
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--epoch_range', type=list, default=[79], help='epochs to infer&eval')
parser.add_argument('--split', default='test_seen', choices=['test', 'test_seen', 'test_similar', 'test_novel']) # corresp to evaluate func

# default in shell
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 15000]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--my_data', default=True, help='If the data belongs to yourself')
cfgs = parser.parse_args()

my_color_path = '/home/zhy/yc_dir/realsense_yc/captured_images' 
my_depth_path = '/home/zhy/yc_dir/realsense_yc/captured_images'

# load model config
with open(os.path.join(ROOT_DIR + '/models/model_config.yaml'), 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(epoch):
    # grasp_labels = load_grasp_labels(cfgs.dataset_root)
    test_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=None, suction_labels_root=cfgs.suction_labels_root,
                                camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=False, load_label=False)
    scene_list = test_dataset.scene_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

    # Init the model
    net = GraspNet(model_config, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(cfgs.log_dir, 'epoch_{}.tar'.format(epoch))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))

    net.eval()
    for batch_idx, source_batch in enumerate(tqdm(test_dataloader)):
        # def. save path
        data_idx = batch_idx
        save_grasp_dir = os.path.join(cfgs.log_dir, 'Eval_Results_' + str(epoch), 'Grasp', cfgs.split, 
                                        scene_list[data_idx], cfgs.camera)
        save_suction_dir = os.path.join(cfgs.log_dir, 'Eval_Results_' + str(epoch), 'Suction', cfgs.split, 
                                        scene_list[data_idx], cfgs.camera, 'suction')
        os.makedirs(save_grasp_dir, exist_ok=True)
        os.makedirs(save_suction_dir, exist_ok=True)
        save_grasp_path = os.path.join(save_grasp_dir, str(data_idx % 256).zfill(4) + '.npy')
        save_suction_path = os.path.join(save_suction_dir, str(data_idx % 256).zfill(4) + '.npz')
        if os.path.exists(save_grasp_path) and os.path.exists(save_suction_path):
            continue

        # data, label to device
        for key in source_batch:
            if 'list' in key:
                for i in range(len(source_batch[key])):
                    for j in range(len(source_batch[key][i])):
                        source_batch[key][i][j] = source_batch[key][i][j].to(device)
            else:
                source_batch[key] = source_batch[key].to(device)

        # Forward pass -> grasp_preds, suction_preds
        with torch.no_grad():
            source_end_points = net(source_batch)
            print(source_end_points)
            grasp_preds = pred_grasp_decode(source_end_points)
            # suction_preds = pred_suction_decode(source_end_points)

        # Dump results for grasps evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]
            # save grasps
            gg.save_npy(save_grasp_path)

        # # Dump results for suctions evaluation
        # np.savez(save_suction_path, arr_0=suction_preds)

def get_my_data():
    img_num = 22
    depth = np.array(Image.open(os.path.join(my_depth_path, f'depth_image_{str(img_num).zfill(2)}.png')))
    color = np.array(Image.open(os.path.join(my_color_path, f'color_image_{str(img_num).zfill(2)}.png')))
    with open('/home/zhy/yc_dir/realsense_yc/results.json') as f:
        fil = json.load(f)
        labels = fil["Intrinsics_labels"]
        values = fil["Intrinsics_values"]
        intrinsics = dict(zip(labels, values))
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        cx = intrinsics.get("cx")
        cy = intrinsics.get("cy")
        width = intrinsics.get("width")
        height = intrinsics.get("height")
        print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}, width: {width}, height: {height}")
    factor_depth = 1000
    camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    color = color / 255
    cloud = cloud.reshape(-1, 3) 
    xmin, ymin, zmin = -0.29 , -0.40 , 0
    xmax, ymax, zmax = 0.50, 0.40, 1.0
    mask_x = ((cloud[:, 0] > xmin) & (cloud[:, 0] < xmax))
    mask_y = ((cloud[:, 1] > ymin) & (cloud[:, 1] < ymax))
    mask_z = ((cloud[:, 2] > zmin) & (cloud[:, 2] < zmax))
    workspace_mask = (mask_x & mask_y & mask_z)
    #depth_mask = (depth > 0)
    mask = workspace_mask
    print("Shape of cloud:", cloud.shape)
    print("Shape of mask:", mask.shape)
    cloud_masked = cloud[mask]
    # scloud_masked= cloud_masked.reshape(720, 1280, 3)
    mask = mask.reshape(720,1280)
    color_masked = color[mask]
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                'color': color_sampled.astype(np.float32),
                }
    return ret_dict



def inference_my_data(epoch):
    # test_dataset = GraspNetDataset(cfgs.dataset_root, grasp_labels=None, suction_labels_root=cfgs.suction_labels_root,
    #                                camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
    #                                remove_outlier=False, augment=False, load_label=False)
    sample_data = get_my_data()
    pointcloud = sample_data['point_clouds']
    color = sample_data['color']
    net = GraspNet(model_config, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    checkpoint_path = os.path.join(cfgs.log_dir, 'epoch_{}.tar'.format(epoch))
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    sample_data = minkowski_collate_fn([sample_data])
    for key in sample_data:
        if 'list' in key:
            for i in range(len(sample_data[key])):
                for j in range(len(sample_data[key][i])):
                    sample_data[key][i][j] = sample_data[key][i][j].to(device)
        else:
            sample_data[key] = sample_data[key].to(device)

    net.eval()
    with torch.no_grad():
        source_end_points = net(sample_data)
        grasp_preds = pred_grasp_decode(source_end_points)  

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    gg = gg.sort_by_score()

    indices_to_remove = []
    # xmin, ymin, zmin = -0.29 , -0.40 , 0
    # xmax, ymax, zmax = 0.50, 0.40, 1.0
    for i, grasp in enumerate(gg):
        if (grasp.translation[0] <= -0.2 or grasp.translation[0] >= 0.35) or \
            (grasp.translation[1] <= -0.2 or grasp.translation[1] >= 0.1):
            indices_to_remove.append(i)

    gg = gg.remove(indices_to_remove)    

    grippers = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(color, dtype=np.float32))
    o3d.visualization.draw_geometries([cloud, *grippers[:20]])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("screenshot.png")
    # vis.destroy_window()

    # o3d.io.write_point_cloud("/home/zhy/graspnet_point_cloud/vis_grasp_files/pointcloud_output1.ply", cloud)
    # combined_grippers = o3d.geometry.TriangleMesh()
    # for gripper in grippers[:10]:
    #     combined_grippers += gripper
    # o3d.io.write_triangle_mesh("/home/zhy/graspnet_point_cloud/vis_grasp_files/all_grippers1.ply", combined_grippers)

def evaluate_grasp(epoch):
    dump_dir = os.path.join(cfgs.log_dir, 'Eval_Results_' + str(epoch), 'Grasp', cfgs.split)
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)

    # evaluate splits
    if cfgs.split == 'test_seen':
        res, ap = ge.eval_seen(dump_folder=dump_dir, proc=6)
    if cfgs.split == 'test_similar':
        res, ap = ge.eval_similar(dump_folder=dump_dir, proc=6)
    if cfgs.split == 'test_novel':
        res, ap = ge.eval_novel(dump_folder=dump_dir, proc=6)
    if cfgs.split == 'test':
        res, ap = ge.eval_all(dump_folder=dump_dir, proc=6)
    save_dir = os.path.join(dump_dir, 'ap_{}_{}.npy'.format(epoch, cfgs.split))
    np.save(save_dir, res)


def evaluate_suction(epoch):
    dump_dir = os.path.join(cfgs.log_dir, 'Eval_Results_' + str(epoch), 'Suction')
    se = SuctionNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)

    # evaluate splits
    if cfgs.split == 'test_seen':
        res, ap_top50, ap_top1 = se.eval_seen(dump_folder=dump_dir, proc=30)
    if cfgs.split == 'test_similar':
        res, ap_top50, ap_top1 = se.eval_similar(dump_folder=dump_dir, proc=30)
    if cfgs.split == 'test_novel':
        res, ap_top50, ap_top1 = se.eval_novel(dump_folder=dump_dir, proc=30)
    if cfgs.split == 'test':
        res, ap_top50, ap_top1 = se.eval_all(dump_folder=dump_dir, proc=30)
    save_dir = os.path.join(dump_dir, 'ap_{}_{}.npy'.format(epoch, cfgs.split))
    np.save(save_dir, res)

if __name__ == '__main__':
    if cfgs.my_data:
        inference_my_data(epoch=1)
    else:
        print('*' * 50)
        print('Eval {} dataset'.format(cfgs.split))
        print('*' * 50)

        for epoch in cfgs.epoch_range:
            if cfgs.infer:
                inference(epoch)
            if cfgs.eval:
                evaluate_grasp(epoch)
                # evaluate_suction(epoch)
        print('Finish !')


    # # show Table
    # grasp_ap = np.load(os.path.join(cfgs.log_dir, 'dump_20_test_similar/ap_20_test_similar.npy'))
    # AP = [100*np.mean(np.mean(np.mean(grasp_ap, axis=2), axis=1), axis=0),
    #       100*np.mean(np.mean(np.mean(grasp_ap[0:30], axis=2), axis=1), axis=0),
    #       100*np.mean(np.mean(np.mean(grasp_ap[30:60], axis=2), axis=1), axis=0),
    #       100*np.mean(np.mean(np.mean(grasp_ap[60:90], axis=2), axis=1), axis=0)]
    # mAP = np.mean(np.array(AP), axis=1)

    # AP = np.round_(AP, decimals = 2)
    # mAP = np.round_(mAP, decimals = 2)
    # print('\nEvaluation Result:\n----------\n{}, AP={}, AP0.8={}, AP0.4={}'.format('realsense', mAP[1], AP[1][3], AP[1][1]))
