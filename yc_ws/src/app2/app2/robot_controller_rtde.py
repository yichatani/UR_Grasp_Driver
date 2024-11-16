import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import threading
import sys
import math

import rtde_receive
import rtde_control

from std_msgs.msg import Bool

class RobotControllerRTDE(Node):
    def __init__(self, robot_ip):
        super().__init__('robot_controller_rtde')

        try:
            self.rtde_receive = rtde_receive.RTDEReceiveInterface(robot_ip)
            self.rtde_control = rtde_control.RTDEControlInterface(robot_ip)
            self.get_logger().info(f'Connected to RTDE at {robot_ip}.')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to RTDE: {e}')
            sys.exit(1)


        self.movement_publisher = self.create_publisher(Bool, '/robot_controller/movedToPose', 10)


        # Subscribe to grasp poses
        self.subscription = self.create_subscription(
            Pose,
            '/grasp_detector/grasp_pose',
            self.grasp_pose_callback,
            10
        )
        self.subscription  # Just to prevent unused variable warning

        self.lock = threading.Lock()

        self.get_logger().info('RobotControllerRTDE initialized and subscribed to /grasp_detector/grasp_pose.')

    def grasp_pose_callback(self, msg):
        """Callback function to handle incoming grasp poses."""

        with self.lock:
            print(msg)
            self.get_logger().info('Received a new grasp pose. Moving the robot.')

            target_position = [msg.position.x, msg.position.y, msg.position.z]
            q = msg.orientation
            target_orientation = self.quaternion_to_euler(q)

            target_pose = target_position + target_orientation

            self.get_logger().info(f'Target Pose (Cartesian): {target_pose}')

            # Send the pose to the robot
            try:
                #self.rtde_control.moveL(target_pose, 0.1, 0.1)        ####################3
                self.get_logger().info('Command sent to move the robot.')


                movement_status = Bool()
                movement_status.data = True
                self.movement_publisher.publish(movement_status)
                self.get_logger().info('Published movement status: True')

            except Exception as e:
                self.get_logger().error(f'Failed to move the robot: {e}')

                movement_status = Bool()
                movement_status.data = False
                self.movement_publisher.publish(movement_status)

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        x, y, z, w = q.x, q.y, q.z, q.w

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)

    robot_ip = '192.168.56.101'  
    robot_controller = RobotControllerRTDE(robot_ip)

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down RobotControllerRTDE node.')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
