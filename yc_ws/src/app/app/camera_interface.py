# app/camera_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np

class CameraInterface:
    def __init__(self, node: Node, color_topic: str, depth_topic: str, queue_size: int = 10, slop: float = 0.1):
        """
        Initializes the CameraInterface.
        
        Args:
            node (rclpy.node.Node): The ROS2 node instance.
            color_topic (str): The ROS2 topic name for color images.
            depth_topic (str): The ROS2 topic name for depth images.
            queue_size (int): Queue size for message synchronization.
            slop (float): Allowed time difference for approximate synchronization.
        """
        self.node = node
        self.bridge = CvBridge()
        self.latest_color_image = None
        self.latest_depth_image = None

        # Subscribers for color and depth images
        self.color_sub = message_filters.Subscriber(node, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(node, Image, depth_topic)

        # Synchronize the image topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=queue_size,
            slop=slop
        )
        self.ts.registerCallback(self.callback)

        self.node.get_logger().info(f"CameraInterface: Subscribed to '{color_topic}' and '{depth_topic}' with queue_size={queue_size} and slop={slop}.")

    def callback(self, color_msg: Image, depth_msg: Image):
        """
        Callback function for synchronized image topics.

        Args:
            color_msg (sensor_msgs.msg.Image): The color image message.
            depth_msg (sensor_msgs.msg.Image): The depth image message.
        """
        try:
            # Convert ROS Image messages to OpenCV images
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Update the latest images
            self.latest_color_image = color_image
            self.latest_depth_image = depth_image

            self.node.get_logger().debug("CameraInterface: Received and converted synchronized images.")
        except CvBridgeError as e:
            self.node.get_logger().error(f"CameraInterface: CvBridge Error: {e}")

    def get_latest_color_image(self) -> np.ndarray:
        """
        Retrieves the latest color image.

        Returns:
            np.ndarray: The latest color image, or None if not available.
        """
        return self.latest_color_image

    def get_latest_depth_image(self) -> np.ndarray:
        """
        Retrieves the latest depth image.

        Returns:
            np.ndarray: The latest depth image, or None if not available.
        """
        return self.latest_depth_image
