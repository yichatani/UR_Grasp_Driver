# app2/coordinator.py
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseStamped
from camera_controller import CameraController
from grasp_detector_controller import GraspDetector

def main(args=None):
    rclpy.init(args=args)
    camera_controller = CameraController()

    try:
        rclpy.spin(camera_controller)
    except KeyboardInterrupt:
        camera_controller.get_logger().info('Shutting down CameraController node.')
    finally:
        camera_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
