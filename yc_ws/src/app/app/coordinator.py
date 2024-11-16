# app/coordinator.py
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseStamped
from .camera_interface_controller import CameraInterfaceController
from .grasp_detection_controller import GraspDetectorController

class Coordinator(Node):
    def __init__(self):
        super().__init__('coordinator')
        self.get_logger().info("Initializing Coordinator...")

        # Declare parameters with default values or allow them to be set via launch
        #self.declare_parameter('grasp_detector_checkpoint', '/path/to/checkpoint.pth')  # I need to replace with actual path
        self.declare_parameter('grasp_frame_id', 'camera_link')

        # Retrieve parameters
        #grasp_detector_checkpoint = self.get_parameter('grasp_detector_checkpoint').get_parameter_value().string_value
        #grasp_frame_id = self.get_parameter('grasp_frame_id').get_parameter_value().string_value

        # Initialize Controllers
        self.camera_interface_controller = CameraInterfaceController(self)
        #self.grasp_detector_controller = GraspDetectorController(
        #    node=self,
        #    checkpoint_path=grasp_detector_checkpoint
        #)
        #self.robot_controller = RobotController(self)
        #self.gripper_controller = GripperController(self)

        self.get_logger().info("Coordinator initialized.")

        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to execute grasp sequence periodically (e.g., every 10 seconds)
    #     self.grasp_timer = self.create_timer(10.0, self.execute_grasp_sequence)
    #     self.get_logger().info("Coordinator is ready to execute grasp sequences.")

    # def execute_grasp_sequence(self):
    #     try:
    #         self.get_logger().info("Starting grasp sequence.")

    #         # Open gripper before moving
    #         self.gripper_controller.open_gripper()

    #         # Get the latest images from CameraInterfaceController
    #         color_image, depth_image = self.camera_interface_controller.get_latest_images()

    #         if color_image is None or depth_image is None:
    #             self.get_logger().warning("Latest images not available yet.")
    #             return

    #         # Update GraspDetectorController with latest images and retrieve grasp pose
    #         grasp_pose = self.grasp_detector_controller.update_images(color_image, depth_image)

    #         if grasp_pose is None:
    #             self.get_logger().warning("No grasp pose detected.")
    #             return

    #         # Transform grasp pose to robot base frame
    #         grasp_pose_base = self.transform_pose_to_base_frame(grasp_pose)

    #         self.get_logger().info(f"Moving to grasp pose in base frame:\n{grasp_pose_base}")

    #         # Move robot to grasp pose
    #         self.robot_controller.move_to_pose(grasp_pose_base)

    #         # Close gripper to grasp
    #         self.gripper_controller.close_gripper()

    #         self.get_logger().info("Grasp executed successfully.")
    #         # Optionally, perform additional actions like lifting the object

    #     except Exception as e:
    #         self.handle_errors(e)

    # def transform_pose_to_base_frame(self, pose: PoseStamped) -> PoseStamped:
    #     self.get_logger().info("Transforming pose to robot base frame using TF2.")
    #     target_frame = 'base'  # Replace with your robot's base frame name
    #     source_frame = pose.header.frame_id

    #     try:
    #         # Wait for the transform to become available
    #         if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0)):
    #             self.get_logger().error(f"Cannot transform from {source_frame} to {target_frame}.")
    #             return pose  # Return the original pose if transform is not available

    #         # Transform the pose
    #         transformed_pose = self.tf_buffer.transform(pose, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
    #         return transformed_pose
    #     except tf2_ros.LookupException as ex:
    #         self.get_logger().error(f"Transform lookup failed: {ex}")
    #         raise
    #     except tf2_ros.ExtrapolationException as ex:
    #         self.get_logger().error(f"Transform extrapolation failed: {ex}")
    #         raise
    #     except Exception as ex:
    #         self.get_logger().error(f"Failed to transform pose: {ex}")
    #         raise

    # def handle_errors(self, error):
    #     self.get_logger().error(f"Error occurred: {error}")
    #     self.robot_controller.stop_motion()
    #     self.gripper_controller.open_gripper()

def main(args=None):
    rclpy.init(args=args)
    coordinator = Coordinator()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        coordinator.get_logger().info('Shutting down Coordinator...')
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
