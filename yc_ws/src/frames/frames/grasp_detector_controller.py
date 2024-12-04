import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import threading
import numpy as np
import random

from camera_controller import CameraController

class GraspDetector(Node):
    def __init__(self):
        super().__init__('grasp_detector')

        # Initialize CameraController
        self.camera_controller = CameraController(self)

        # Publisher for grasp poses
        self.grasp_pose_publisher = self.create_publisher(Pose, '/grasp_detector/grasp_pose', 10)

        # Timer to trigger grasp detection periodically (e.g., every 1 second)
        self.timer_period = 1.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('GraspDetector initialized and timer started.')

    def timer_callback(self):
        """Periodic callback to detect and publish a grasp pose."""
        # Retrieve the latest images from CameraController
        color_image = self.camera_controller.get_color_image()
        depth_image = self.camera_controller.get_depth_image()

        if color_image is None or depth_image is None:
            self.get_logger().warn('Waiting for color and depth images...')
            return

        self.get_logger().info('Received color and depth images. Processing for grasp detection.')

        # Placeholder for grasp detection logic
        translation, rotation_matrix = self.generate_grasp_pose(color_image, depth_image)

        if translation is None or rotation_matrix is None:
            self.get_logger().warn('No grasp pose detected.')
            return

        # Convert rotation matrix to quaternion
        #quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        quaternion = [
        0.2594052377364779,
        0.6452711484893212,
        0.4281667286508839,
        0.5770678643266786
    ]
        # Create Pose message
        pose = Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        self.get_logger().info(f'Publishing Grasp Pose: Position={translation}, Orientation={quaternion}')

        # Publish the grasp pose
        self.grasp_pose_publisher.publish(pose)

    def generate_grasp_pose(self, color_image, depth_image):
        """
        A method for generating a single grasp pose.
        Need to replace this method with actual grasp detection logic using AnyGrasp SDK.
        Returns:
            translation (np.ndarray): [x, y, z] in meters
            rotation_matrix (np.ndarray): 3x3 rotation matrix
        """
        # For demonstration, we'll create a dummy grasp pose
        height, width = color_image.shape[:2]

        # Generate random pixel coordinates
        x_pixel = random.randint(0, width - 1)
        y_pixel = random.randint(0, height - 1)

        # Retrieve depth at the pixel
        depth = depth_image[y_pixel, x_pixel]

        if depth == 0:
            self.get_logger().warn(f'Depth at ({x_pixel}, {y_pixel}) is zero. Skipping this grasp.')
            return None, None

        # Convert pixel coordinates to real-world coordinates (simple pinhole model)
        fx = 600  # Focal length in x (replace with actual)
        fy = 600  # Focal length in y (replace with actual)
        cx = width / 2
        cy = height / 2

        z = depth / 1000.0  # Convert mm to meters if depth is in mm
        x = (x_pixel - cx) * z / fx
        y = (y_pixel - cy) * z / fy

        translation = np.array([x, y, z])

        # Define a dummy rotation matrix (identity)
        rotation_matrix = np.eye(3)

        return translation, rotation_matrix

    # def rotation_matrix_to_quaternion(self, rotation_matrix):
    #     """
    #     Convert a 3x3 rotation matrix to a quaternion.
    #     Args:
    #         rotation_matrix (np.ndarray): 3x3 rotation matrix
    #     Returns:
    #         tuple: (x, y, z, w) quaternion
    #     """
    #     # Convert the rotation matrix to a 4x4 homogeneous matrix
    #     homogeneous_matrix = np.eye(4)
    #     homogeneous_matrix[:3, :3] = rotation_matrix

    #     # Convert to quaternion
    #     quaternion = quaternion_from_matrix(homogeneous_matrix)

    #     return quaternion

    # def quaternion_to_euler(self, q):
    #     """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    #     x, y, z, w = q.x, q.y, q.z, q.w

    #     # Using tf_transformations for conversion
    #     from tf_transformations import euler_from_quaternion
    #     roll, pitch, yaw = euler_from_quaternion([x, y, z, w])

    #     return [roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)
    grasp_detector = GraspDetector()

    try:
        rclpy.spin(grasp_detector)
    except KeyboardInterrupt:
        grasp_detector.get_logger().info('Shutting down GraspDetector node.')
    finally:
        grasp_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()