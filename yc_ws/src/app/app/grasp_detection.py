# coordinator_pkg/coordinator_pkg/grasp_detector.py
import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from geometry_msgs.msg import PoseStamped
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup  # Ensure graspnetAPI is in your PYTHONPATH
from scipy.spatial.transform import Rotation as R

class GraspDetector2:
    def __init__(self, checkpoint_path, max_gripper_width=0.1, gripper_height=0.03, top_down_grasp=False, debug=False, frame_id='camera_link'):
        self.checkpoint_path = checkpoint_path
        self.max_gripper_width = max(0, min(0.1, max_gripper_width))
        self.gripper_height = gripper_height
        self.top_down_grasp = top_down_grasp
        self.debug = debug
        self.frame_id = frame_id

        # Initialize the AnyGrasp model
        self.anygrasp = AnyGrasp(
            max_gripper_width=self.max_gripper_width,
            gripper_height=self.gripper_height,
            top_down_grasp=self.top_down_grasp,
            debug=self.debug
        )
        self.anygrasp.load_net(self.checkpoint_path)

        # Camera intrinsics
        self.fx = 642.4262770792711
        self.fy = 642.3461744750167
        self.cx = 647.5434733474444
        self.cy = 373.3602344467871
        self.scale = 1000.0  # Scale for depth image conversion

        # Workspace limits
        self.lims = [-0.19, 0.12, 0.02, 0.15, 0.0, 1.0]

    def process_images(self, color_image: np.ndarray, depth_image: np.ndarray) -> PoseStamped:
        """
        Processes the provided color and depth images to detect a grasp pose.

        Args:
            color_image (np.ndarray): The color image in BGR8 encoding.
            depth_image (np.ndarray): The depth image.

        Returns:
            PoseStamped: The detected grasp pose.
        """
        # Generate point cloud from depth image
        points = self._generate_point_cloud(depth_image)

        # Generate color array corresponding to the valid points
        colors = self._get_colors_from_masked_images(color_image, depth_image)

        # Detect grasps
        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=True, dense_grasp=False, collision_detection=False)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0]  # Choose the top grasp

        print('Grasp pick:', gg_pick)

        # Optionally visualize
        if self.debug:
            self._visualize_grasp(cloud, gg_pick)

        # Convert GraspGroup to PoseStamped
        grasp_pose = self._grasp_to_pose(gg_pick)

        return grasp_pose

    def detect_grasp(self) -> PoseStamped:
        """
        Placeholder method if needed for separation.

        Returns:
            PoseStamped: The detected grasp pose.
        """
        # This method can be used if you separate processing and detection steps
        return None

    def _generate_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Generates a point cloud from the depth image.

        Args:
            depth_image (np.ndarray): The depth image.

        Returns:
            np.ndarray: Array of 3D points.
        """
        xmap, ymap = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        points_z = depth_image / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)

        return points

    def _get_colors_from_masked_images(self, color_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Extracts colors corresponding to valid depth points.

        Args:
            color_image (np.ndarray): The color image.
            depth_image (np.ndarray): The depth image.

        Returns:
            np.ndarray: Array of colors corresponding to valid points.
        """
        mask = (depth_image > 0) & (depth_image < 1)
        colors = color_image[mask].astype(np.float32) / 255.0  # Normalize to [0,1]
        return colors

    def _grasp_to_pose(self, grasp):
        """
        Converts a grasp to a PoseStamped message.

        Args:
            grasp: The grasp object.

        Returns:
            PoseStamped: The grasp pose.
        """
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id  # Use the configured frame_id
        pose.pose.position.x = grasp.translation[0]
        pose.pose.position.y = grasp.translation[1]
        pose.pose.position.z = grasp.translation[2]

        # Convert rotation matrix to quaternion
        quaternion = self._rotation_matrix_to_quaternion(grasp.rotation)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        return pose

    def _rotation_matrix_to_quaternion(self, R_matrix):
        """
        Converts a rotation matrix to a quaternion.

        Args:
            R_matrix (np.ndarray): 3x3 rotation matrix.

        Returns:
            tuple: Quaternion (x, y, z, w).
        """
        r = R.from_matrix(R_matrix)
        quat = r.as_quat()  # Returns in (x, y, z, w) format
        return quat

    def _visualize_grasp(self, cloud, grasp):
        """
        Visualizes the grasp using Open3D.

        Args:
            cloud: Open3D point cloud.
            grasp: Grasp object.
        """
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = grasp.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
