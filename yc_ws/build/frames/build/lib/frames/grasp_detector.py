
# grasp_detector.py

import argparse
import os
import time

import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import random

# Ensure these modules are available in your PYTHONPATH
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup



# Initialize configuration similar to argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')

# Bounding box coordinates from the mask
x_min, y_min, x_max, y_max = 264, 47, 924, 526

# Define depth range in meters
z_min = 0.0  # Minimum depth (meters)
z_max = .8  # Maximum depth (meters)

# Function to crop and filter the depth image
def crop_and_filter_depth_in_meters(depth_image, bbox, z_min, z_max):
    """
    Crop and filter the depth image based on bounding box and depth range in meters.

    Parameters:
    - depth_image: The depth image (numpy array, assumed in millimeters).
    - bbox: Tuple containing (x_min, y_min, x_max, y_max).
    - z_min: Minimum depth value in meters.
    - z_max: Maximum depth value in meters.

    Returns:
    - Cropped and filtered depth image (in meters).
    """
    x_min, y_min, x_max, y_max = bbox

    # Crop the depth image
    cropped_depth = depth_image[y_min:y_max, x_min:x_max]

    # Convert depth to meters
    cropped_depth_meters = cropped_depth / 1000.0

    # Apply depth range filtering in meters
    depth_mask = (cropped_depth_meters >= z_min) & (cropped_depth_meters <= z_max)
    filtered_depth = np.where(depth_mask, cropped_depth_meters, 0)  # Set out-of-range values to 0

    return filtered_depth

# Function to crop the color image
def crop_color_image(color_image, bbox):
    """
    Crop the color image based on the bounding box.

    Parameters:
    - color_image: The color image (numpy array).
    - bbox: Tuple containing (x_min, y_min, x_max, y_max).

    Returns:
    - Cropped color image.
    """
    x_min, y_min, x_max, y_max = bbox
    return color_image[y_min:y_max, x_min:x_max]

def compute_workspace_from_bbox(x, y, w, h, depth_image, fx, fy, cx, cy, scale):
    """
    Compute the 3D workspace limits (xmin, xmax, ymin, ymax, zmin, zmax) from a bounding box.

    Parameters:
    - x, y, w, h: Bounding box in pixels (top-left corner and size).
    - depth_image: Depth image in millimeters.
    - fx, fy: Focal lengths of the camera.
    - cx, cy: Principal point offsets of the camera.
    - scale: Depth scaling factor (e.g., 1000 for millimeters to meters).

    Returns:
    - Workspace limits: [xmin, xmax, ymin, ymax, zmin, zmax].
    """
    # Crop depth image to bounding box
    cropped_depth = depth_image[y:y + h, x:x + w] / scale  # Convert to meters

    # Ignore invalid depth values (e.g., zeros)
    valid_depth_mask = (cropped_depth > 0)
    valid_depth = cropped_depth[valid_depth_mask]

    # Compute zmin and zmax
    zmin = valid_depth.min()
    zmax = valid_depth.max()

    # Generate meshgrid for pixel coordinates within the bounding box
    u = np.arange(x, x + w)
    v = np.arange(y, y + h)
    u, v = np.meshgrid(u, v)

    # Crop u, v using the valid depth mask
    u = u[valid_depth_mask]
    v = v[valid_depth_mask]

    # Compute 3D coordinates
    x_real = (u - cx) * valid_depth / fx
    y_real = (v - cy) * valid_depth / fy

    # Compute xmin, xmax, ymin, ymax
    xmin, xmax = x_real.min(), x_real.max()
    ymin, ymax = y_real.min(), y_real.max()

    return [xmin, xmax, ymin, ymax, zmin, zmax]


class GraspDetector:
    def __init__(self, coordinator):
        """
        Initializes the GraspDetector with the given configurations.

        Args:
            checkpoint_path (str): Path to the AnyGrasp model checkpoint.
            max_gripper_width (float): Maximum gripper width in meters (default: 0.1m).
            gripper_height (float): Gripper height in meters (default: 0.03m).
            top_down_grasp (bool): Whether to output top-down grasps (default: False).
            debug (bool): Enable debug mode with visualizations (default: True).
        """

        self.coordinator = coordinator  # Just to use the coordinator node for logging

        self.cfgs = parser.parse_args()
        self.cfgs.max_gripper_width = max(0, min(0.1, self.cfgs.max_gripper_width))

        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()

        print()
        print()
        self.coordinator.get_logger().info('AnyGrasp model loaded.\n')

        # Camera intrinsics
        self.fx = 642.4262770792711
        self.fy = 642.3461744750167
        self.cx = 647.5434733474444
        self.cy = 373.3602344467871





        # self.fx, self.fy = 927.17, 927.37
        # self.cx, self.cy = 651.32, 349.62

        self.scale = 1000.0


        #original
        # self.xmin, self.xmax = -0.19, 0.12
        # self.ymin, self.ymax = 0.02, 0.15
        # self.zmin, self.zmax = 0.0, 1.0

        # self.xmin, self.ymin, self.xmax, self.ymax =  0.264, 0.047, 0.924, 0.526
        # self.zmin = 0.0  # Minimum depth (meters)
        # self.zmax = .8  # Maximum depth (meters)


        self.xmin, self.xmax = -0.25, 0.20
        self.ymin, self.ymax = -.2, 0.4
        self.zmin, self.zmax = -0.1, 1.0

        # self.xmin, self.xmax = -0.26, 0.26  # x-axis limits (half the width of the table)
        # self.ymin, self.ymax = -0.325, 0.325  # y-axis limits (half the length of the table)
        # self.zmin, self.zmax = 0.0, 1.0    # z-axis limits (from the table surface to 1 meter above)

        
        # Workspace limits (in meters)
        #self.lims = [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0]
        self.lims = [self.xmin, self.xmax, self.ymin, self.ymax ,self.zmin, self.zmax]


        self.selected_grasp = None

    def detect_grasps(self, color_image, depth_image):
        """
        Processes the color and depth images to generate grasp poses.

        """

        #self.process_and_visualize_point_cloud(color_image, depth_image)


        # x = 345
        # y = 160
        # w = 605
        # h = 330


        x, y, w, h = 264, 47, 660, 479  # Bounding box (pixels)


        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = compute_workspace_from_bbox(x, y, w, h, depth_image, self.fx, self.fy, self.cx, self.cy, self.scale)


        
        # color_image = color_image[y:y + h, x:x + w]
        # depth_image = depth_image[y:y + h, x:x + w]
        #self.process_and_visualize_point_cloud(color_image, depth_image)
        
        # Normalize color image
        colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Ensure depth image is float32
        depths = depth_image

        #colors, depths = crop_color_image(colors), crop_and_filter_depth_in_meters(depths)

        # get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        # set your workspace to crop point cloud
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)
        #print(points.min(axis=0), points.max(axis=0))

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=True, dense_grasp=False, collision_detection=False)

        if (gg is not None):
            if len(gg) == 0:
                self.coordinator.get_logger().warn('No Grasp detected after collision detection!')                
            else: 
                gg = gg.nms().sort_by_score()
                gg_pick = gg[0:20]
                if gg_pick[0].score > 0.3:
                    self.selected_grasp = gg_pick[0]
                    self.coordinator.get_logger().info(f'Selected Grasp score: {self.selected_grasp.score}\n')
                    self.visualize_grasps(cloud, gg)

        return self.selected_grasp
    

    def visualize_grasps(self, cloud, gg):
        """
        Visualize grasps and handle key presses for specific actions.

        Args:
            cloud: The point cloud to visualize.
            gg: The detected grasps to overlay on the visualization.
        """
        if self.cfgs.debug:
            # Transform cloud and grippers
            trans_mat = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)

            # First visualization without key handling
            o3d.visualization.draw_geometries([*grippers, cloud])

            # Second visualization with key handling
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()

            # Add cloud and the first gripper for visualization
            vis.add_geometry(cloud)
            vis.add_geometry(grippers[0])

            # Define key press actions
            def on_s_press(vis):
                self.coordinator.get_logger().info("Key 's' pressed: Skip this grasp ...\n")
                self.selected_grasp = None
                vis.close()  # Close the visualization window
                return False  # Stop visualization
            
                        # Define key press actions
            def on_e_press(vis):
                self.coordinator.get_logger().info("Key 'e' pressed: Execute this grasp ...\n")
                vis.close()  # Close the visualization window
                return False  # Stop visualization

            # Register key callbacks
            vis.register_key_callback(ord('E'), on_e_press)
            vis.register_key_callback(ord('S'), on_s_press)

            # Run the visualizer
            vis.run()
            vis.destroy_window()













    def process_and_visualize_point_cloud(self, color_image, depth_image):
        if color_image is not None and depth_image is not None:

            # Normalize color image for visualization
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Create point cloud from depth image
            xmap, ymap = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
            points_z = depth_image / self.scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            # Filter points based on workspace limits
            mask = (points_z > 0) & (points_z < self.zmax)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = color_image_rgb[mask]

            # Create Open3D point cloud
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)

            # Apply transformation if necessary (for correct orientation)
            trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            cloud.transform(trans_mat)

            # Visualize the point cloud and handle key events
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name='Point Cloud Visualization')

            vis.add_geometry(cloud)

            # Add key callback to close visualization when 'q' is pressed
            def close_callback(vis):
                self.get_logger().info('Key "q" pressed. Exiting visualization...')
                vis.destroy_window()

            vis.register_key_callback(ord('Q'), close_callback)

            #self.get_logger().info('Visualizing point cloud with Open3D...')
            vis.run()
            vis.destroy_window()






    def generate_point_cloud(self, depth_image):
        """
        Generates a point cloud from the depth image using camera intrinsics.

        Args:
            depth_image (numpy.ndarray): Depth image as a NumPy array (H x W).

        Returns:
            numpy.ndarray: Array of 3D points (N x 3).
        """
        H, W = depth_image.shape
        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
        points_z = depth_image / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        # Apply mask for valid depth
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)

        return points