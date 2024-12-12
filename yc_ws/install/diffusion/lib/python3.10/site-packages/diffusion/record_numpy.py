import rtde_receive
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import os
from datetime import datetime
import threading
from queue import Queue
import time
import signal


class DataCollector:
    def __init__(self, robot_ip, output_dir,episode_n):
        self.end_episode_n = episode_n
        self.buffer = {
            'timestamps': [],
            'current_tcp_poses': [],      
            'current_joint_positions': [], 
            'current_joint_velocities': [],
            'next_tcp_poses': [],         
            'next_joint_positions': [],    
            'next_joint_velocities': [],  
            'gripper_states': [],
            'rgb_images': [],
            'depth_images': [],
            'pointclouds': []
        }
        self.previous_state = None  # To save the previous state
        self.output_dir = output_dir
        self.data_queue = Queue()
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.is_running = False

    def record_step(self):
        """timestamps"""
        timestamp = time.time()
        
        # robot state
        try:
            current_tcp_pose = self.rtde_r.getActualTCPPose()
            current_joint_pos = self.rtde_r.getActualQ()
            current_joint_vel = self.rtde_r.getActualQd()
        except Exception as e:
            print(f"Error retrieving robot state: {e}")
            return
        current_state = {
            'timestamp': timestamp,
            'tcp_pose': current_tcp_pose,
            'joint_position': current_joint_pos,
            'joint_velocity': current_joint_vel
        }
        ################   need add gripper_state
        
        # handle vision
        vision_data = {}
        while not self.data_queue.empty():
            data_type, data, ts = self.data_queue.get()
            vision_data[data_type] = (data, ts)
        
        if self.previous_state is not None:
            self.buffer['timestamps'].append(self.previous_state['timestamp'])
            # States
            self.buffer['current_tcp_poses'].append(self.previous_state['tcp_pose'])
            self.buffer['current_joint_positions'].append(self.previous_state['joint_position'])
            self.buffer['current_joint_velocities'].append(self.previous_state['joint_velocity'])
            
            # Actions
            self.buffer['next_tcp_poses'].append(current_tcp_pose)
            self.buffer['next_joint_positions'].append(current_joint_pos)
            self.buffer['next_joint_velocities'].append(current_joint_vel)
            
            # Observations
            self.buffer['rgb_images'].append(vision_data.get('rgb', (None, timestamp))[0])
            self.buffer['depth_images'].append(vision_data.get('depth', (None, timestamp))[0])
            self.buffer['pointclouds'].append(vision_data.get('pointcloud', (None, timestamp))[0])
        
        # update
        self.previous_state = current_state
    
    def vision_thread(self):
        while self.is_running:
            # Get RGB, Depth, PointCloud
            rgb = self.get_rgb_image()
            depth = self.get_depth_image()
            pointcloud = self.get_pointcloud()

            # Add to queue
            self.data_queue.put(('rgb', rgb, time.time()))
            self.data_queue.put(('depth', depth, time.time()))
            self.data_queue.put(('pointcloud', pointcloud, time.time()))


    def end_episode(self):
        # leave the last step away, for no next step
        episode_dir = os.path.join(self.output_dir, f'episode_{self.end_episode_n:04d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        # save as numpy
        np.savez_compressed(
            os.path.join(episode_dir, 'episode_data.npz'),
            timestamps=np.array(self.buffer['timestamps']),
            # current state
            current_tcp_poses=np.array(self.buffer['current_tcp_poses']),
            current_joint_positions=np.array(self.buffer['current_joint_positions']),
            current_joint_velocities=np.array(self.buffer['current_joint_velocities']),
            # action
            target_tcp_poses=np.array(self.buffer['next_tcp_poses']),
            target_joint_positions=np.array(self.buffer['next_joint_positions']),
            target_joint_velocities=np.array(self.buffer['next_joint_velocities'])
        )
        
        # observation
        if any(self.buffer['rgb_images']):
            np.save(os.path.join(episode_dir, 'rgb_images.npy'), 
                   np.array(self.buffer['rgb_images']))
            np.save(os.path.join(episode_dir, 'depth_images.npy'), 
                   np.array(self.buffer['depth_images']))
            np.save(os.path.join(episode_dir, 'pointclouds.npy'), 
                   np.array(self.buffer['pointclouds']))
        
        # save metadata
        metadata = {
            'episode_id': self.current_episode,
            'start_time': self.buffer['timestamps'][0],
            'end_time': self.buffer['timestamps'][-1],
            'num_frames': len(self.buffer['timestamps'])
        }
        np.save(os.path.join(episode_dir, 'metadata.npy'), metadata)

    def run(self):
        """record"""
        self.is_running = True
        while self.is_running:
            self.record_step()
            time.sleep(0.1)  # control frequency



if __name__ == "__main__":
    collector = DataCollector(
        robot_ip="192.168.1.101",
        output_dir="./collected_data"
    )
    collector_thread = threading.Thread(target=collector.run)
    collector_thread.start()

    def stop_handler(signum, frame):
        print("\nStopping data collection...")
        collector.is_running = False 

    # catch SIGINT（Ctrl+C）
    signal.signal(signal.SIGINT, stop_handler)
    
    # waite thread going end
    try:
        collector_thread.join()
    except KeyboardInterrupt:
        collector.is_running = False

    collector.end_episode()
    print("Data collection complete. Data saved.")









