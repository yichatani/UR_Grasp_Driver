# app/camera_interface_controller.py
import threading
from app.camera_interface import CameraInterface
import numpy as np
import rclpy
from rclpy.node import Node

class CameraInterfaceController:
    def __init__(self, node: Node, color_topic: str = '/Realsense_D455/color/image_raw', depth_topic: str = '/Realsense_D455/aligned_depth_to_color/image_raw'):
        """
        Initializes the CameraInterfaceController.
        
        Args:
            node (rclpy.node.Node): The ROS2 node instance for logging and other utilities.
            color_topic (str): The ROS2 topic name for color images.
            depth_topic (str): The ROS2 topic name for depth images.
        """
        self.node = node
        self.camera_interface = CameraInterface(node, color_topic, depth_topic)
        self.lock = threading.Lock()
        self.node.get_logger().info("CameraInterfaceController: Initialized and ready.")

    def get_latest_images(self):
        """
        Retrieves the latest synchronized color and depth images.

        Returns:
            tuple: (color_image, depth_image) as NumPy arrays, or (None, None) if not available.
        """
        with self.lock:
            color_image = self.camera_interface.get_latest_color_image()
            depth_image = self.camera_interface.get_latest_depth_image()
            return color_image, depth_image
