# gripper_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String  # Replace String with the appropriate message type
import threading
import time
#from gripper.robotiq_controllers import 
class GripperController(Node):
    def __init__(self):
        super().__init__('gripper_controller')

        # Subscriber to robot movement status
        self.movement_subscriber = self.create_subscription(
            Bool,
            '/robot_controller/movedToPose',
            self.movement_callback,
            10
        )

        # Publisher to gripper command
        self.gripper_publisher = self.create_publisher(
            String,  # Replace String with the appropriate message type
            '/robotiq_gripper_controller/gripper_cmd',  # Replace with the actual gripper command topic
            10
        )

        self.get_logger().info('GripperController initialized and subscribed to /robot_controller/movedToPose.')

    def movement_callback(self, msg):
        """Callback function to handle robot movement status."""
        if msg.data:
            self.get_logger().info('Robot reached the target pose. Initiating gripper open and close sequence.')
            # Start a new thread to handle gripper operations
            threading.Thread(target=self.open_and_close_gripper).start()
        else:
            self.get_logger().warn('Robot failed to reach the target pose. No gripper action taken.')

    def open_and_close_gripper(self):
        """Function to open and then close the gripper."""
        # Send open command
        open_cmd = String()
        open_cmd.data = 'open'  # Need to replace with the appropriate command format
        self.gripper_publisher.publish(open_cmd)
        self.get_logger().info('Published open command to gripper.')

        # Wait for the gripper to open
        time.sleep(1)  
        # Send close command
        close_cmd = String()
        close_cmd.data = 'close'  # Need to replace with the appropriate command format
        self.gripper_publisher.publish(close_cmd)
        self.get_logger().info('Published close command to gripper.')

def main(args=None):
    rclpy.init(args=args)
    gripper_controller = GripperController()

    try:
        rclpy.spin(gripper_controller)
    except KeyboardInterrupt:
        gripper_controller.get_logger().info('Shutting down GripperController node.')
    finally:
        gripper_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
