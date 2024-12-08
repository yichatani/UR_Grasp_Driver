# import rtde_control
# import rtde_receive
# import numpy as np
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, PointCloud2
# from cv_bridge import CvBridge
# import os
# from datetime import datetime
# import threading
# from queue import Queue
# import time

# class NumpyDataLoader:
#     def __init__(self, episode_dir: str):
#         """
        
#         Args:
#             episode_dir
#         """
#         self.episode_dir = episode_dir
        
#     def load_episode_data(self, load_vision=True):
#         """
#         Args:
#             load_vision: 
#         """
#         data = np.load(os.path.join(self.episode_dir, 'episode_data.npz'))
#         episode_data = {
#             'timestamps': data['timestamps'],
#             'robot_state': {
#                 'tcp_poses': data['tcp_poses'],
#                 'joint_positions': data['joint_positions'],
#                 'joint_velocities': data['joint_velocities']
#             },
#             'actions': {
#                 'target_poses': data['target_poses'],
#                 'target_joint_velocities': data['target_joint_velocities'],
#                 'gripper_states': data['gripper_states']
#             }
#         }
        
#         if load_vision:
#             episode_data['vision'] = {
#                 'rgb_images': np.load(os.path.join(self.episode_dir, 'rgb_images.npy')),
#                 'depth_images': np.load(os.path.join(self.episode_dir, 'depth_images.npy')),
#                 'pointclouds': np.load(os.path.join(self.episode_dir, 'pointclouds.npy'))
#             }
            
#         return episode_data
    
#     def load_frame(self, frame_idx: int):
#         """load one frame"""
#         data = np.load(os.path.join(self.episode_dir, 'episode_data.npz'))
#         frame_data = {
#             'timestamp': data['timestamps'][frame_idx],
#             'robot_state': {
#                 'tcp_pose': data['tcp_poses'][frame_idx],
#                 'joint_position': data['joint_positions'][frame_idx],
#                 'joint_velocity': data['joint_velocities'][frame_idx]
#             },
#             'action': {
#                 'target_pose': data['target_poses'][frame_idx],
#                 'target_joint_velocity': data['target_joint_velocities'][frame_idx],
#                 'gripper_state': data['gripper_states'][frame_idx]
#             }
#         }
        
#         frame_data['vision'] = {
#             'rgb_image': np.load(os.path.join(self.episode_dir, 'rgb_images.npy'))[frame_idx],
#             'depth_image': np.load(os.path.join(self.episode_dir, 'depth_images.npy'))[frame_idx],
#             'pointcloud': np.load(os.path.join(self.episode_dir, 'pointclouds.npy'))[frame_idx]
#         }
        
#         return frame_data

#     def load_state_action_pairs(self):
#         data = np.load(os.path.join(self.episode_dir, 'episode_data.npz'))
        
#         states = np.concatenate([
#             data['tcp_poses'],
#             data['joint_positions'],
#             data['joint_velocities']
#         ], axis=1)
        
#         actions = np.concatenate([
#             data['target_poses'],
#             data['target_joint_velocities'],
#             data['gripper_states'].reshape(-1, 1)
#         ], axis=1)
        
#         return states, actions

import os
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_episode(self, episode_id):
        episode_path = os.path.join(self.data_dir, f'episode_{episode_id:04d}')
        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"Episode directory {episode_path} does not exist.")

        # Load data from the compressed npz file
        episode_data = np.load(os.path.join(episode_path, 'episode_data.npz'))
        data = {
            'timestamps': episode_data['timestamps'],
            # Current state
            'current_tcp_poses': episode_data['current_tcp_poses'],
            'current_joint_positions': episode_data['current_joint_positions'],
            'current_joint_velocities': episode_data['current_joint_velocities'],
            # Actions
            'target_tcp_poses': episode_data['target_tcp_poses'],
            'target_joint_positions': episode_data['target_joint_positions'],
            'target_joint_velocities': episode_data['target_joint_velocities'],
        }
        
        # Load optional observation data
        for key in ['rgb_images', 'depth_images', 'pointclouds']:
            file_path = os.path.join(episode_path, f'{key}.npy')
            if os.path.exists(file_path):
                data[key] = np.load(file_path)
            else:
                data[key] = None

        # Load metadata
        metadata_path = os.path.join(episode_path, 'metadata.npy')
        if os.path.exists(metadata_path):
            data['metadata'] = np.load(metadata_path, allow_pickle=True).item()
        else:
            data['metadata'] = {}

        return data

    def list_episodes(self):
        """List all episodes available in the data directory."""
        episodes = [
            name for name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, name)) and name.startswith('episode_')
        ]
        return sorted(episodes)

if __name__ == "__main__":
    data_loader = DataLoader(data_dir="./collected_data")

    # List available episodes
    episodes = data_loader.list_episodes()
    print("Available episodes:", episodes)

    # Load a specific episode
    try:
        episode_id = int(input("Enter the episode ID to load: "))
        data = data_loader.load_episode(episode_id)
        print("Loaded episode metadata:", data.get('metadata', {}))
        print("Timestamps:", data['timestamps'])
        if data['rgb_images'] is not None:
            print(f"RGB images shape: {data['rgb_images'].shape}")
        if data['depth_images'] is not None:
            print(f"Depth images shape: {data['depth_images'].shape}")
        if data['pointclouds'] is not None:
            print(f"Pointclouds shape: {data['pointclouds'].shape}")
    except FileNotFoundError as e:
        print(e)
