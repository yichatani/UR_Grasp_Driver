# app/grasp_detector_controller.py
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R  # Correct
from geometry_msgs.msg import PoseStamped
#from .anygrasp_sdk.grasp_detection.demo import GraspDetector
from app.grasp_detection import GraspDetector2
import rclpy
from rclpy.node import Node

class GraspDetectorController:
    def __init__(self, node: Node, checkpoint_path: str):
        """
        Initializes the GraspDetectorController.

        Args:
            node (rclpy.node.Node): The ROS2 node instance for logging and other utilities.
            checkpoint_path (str): Path to the trained AnyGrasp model checkpoint.
        """
        self.node = node
        self.grasp_detector = GraspDetector2(
            checkpoint_path=checkpoint_path,
            max_gripper_width=0.1,
            gripper_height=0.03,
            top_down_grasp=True,
            debug=False  # Set to True for visualization
        )

        # Lock for thread-safe operations
        self.lock = threading.Lock()

    def update_images(self, color_image: np.ndarray, depth_image: np.ndarray) -> PoseStamped:
        """
        Updates the GraspDetector with the latest images and retrieves the grasp pose.

        Args:
            color_image (np.ndarray): Latest color image.
            depth_image (np.ndarray): Latest depth image.

        Returns:
            PoseStamped: The detected grasp pose, or None if no grasp was detected.
        """
        with self.lock:
            gg_pick = self.grasp_detector.process_images(color_image, depth_image)
            if gg_pick is not None and len(gg_pick) > 0:
                grasp_pose = self._grasp_to_pose(gg_pick[0])  # Use the top grasp
                self.node.get_logger().info(f"GraspDetectorController: Detected grasp pose:\n{grasp_pose}")
                return grasp_pose
            else:
                self.node.get_logger().warning("GraspDetectorController: No grasp pose detected.")
                return None

    def _grasp_to_pose(self, grasp):
        """
        Converts a grasp to a PoseStamped message.

        Args:
            grasp: The grasp object.

        Returns:
            PoseStamped: The grasp pose.
        """
        pose = PoseStamped()
        pose.header.frame_id = "camera_link"  # This should match the camera frame
        pose.pose.position.x = grasp.translation[0]
        pose.pose.position.y = grasp.translation[1]
        pose.pose.position.z = grasp.translation[2]

        # Convert rotation matrix to quaternion
        quaternion = self._rotation_matrix_to_quaternion(grasp.rotation_matrix)
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
