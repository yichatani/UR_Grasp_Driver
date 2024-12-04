import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading

class CameraController:
    def __init__(self, node):
        self.node = node
        self.br = CvBridge()

        self.color_image = None
        self.depth_image = None

        # Subscribers for color and depth images
        self.color_subscriber = self.node.create_subscription(
            Image,
            '/Realsense_D455/color/image_raw',
            self.color_callback,
            10  
        )

        self.depth_subscriber = self.node.create_subscription(
            Image,
            '/Realsense_D455/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10  
        )

        self.lock = threading.Lock()

        self.node.get_logger().info('CameraController initialized and subscribers are active.')

    def color_callback(self, msg):

        try:
            color_img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.color_image = color_img
            self.node.get_logger().debug('Received color image.')
        except Exception as e:
            self.node.get_logger().error(f'Failed to convert color image: {e}')

    def depth_callback(self, msg):
        
        try:
            depth_img = self.br.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.depth_image = depth_img
            self.node.get_logger().debug('Received depth image.')
        except Exception as e:
            self.node.get_logger().error(f'Failed to convert depth image: {e}')

    def get_color_image(self):
        """Retrieve the latest color image."""
        with self.lock:
            return self.color_image.copy() if self.color_image is not None else None

    def get_depth_image(self):
        """Retrieve the latest depth image."""
        with self.lock:
            return self.depth_image.copy() if self.depth_image is not None else None
