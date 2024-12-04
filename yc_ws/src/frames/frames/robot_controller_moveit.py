# robot_controller_moveit2.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import threading
from std_msgs.msg import Bool

# robot_controller_moveit2.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import threading
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped



class RobotControllerMoveIt2(Node):
    def __init__(self):
        super().__init__('robot_controller_moveit2')

        # Initialize Action Client for MoveGroup
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')

        self.movement_publisher = self.create_publisher(Bool, '/robot_controller/movedToPose', 10)

        # Subscribe to grasp poses
        self.subscription = self.create_subscription(
            Pose,
            '/grasp_detector/grasp_pose',
            self.grasp_pose_callback,
            10
        )
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()

        self.get_logger().info('RobotControllerMoveIt2 initialized and subscribed to /grasp_detector/grasp_pose.')


    def grasp_pose_callback(self, msg):
        self.get_logger().info('Received a new grasp pose. Planning to move the robot.')

        with self.lock:
            # Wait for the MoveGroup action server to be available
            if not self.move_group_client.wait_for_server(timeout_sec=7.0):
                self.get_logger().error('MoveGroup action server not available!')
                return
            
            goal_msg = MoveGroup.Goal()
            
            target_pose_stamped = PoseStamped()
            target_pose_stamped.header.frame_id = "base_link"  
            target_pose_stamped.pose = msg  #

            goal_msg.request.pose_targets = [target_pose_stamped]  
            
            goal_msg.request.planner_id = "ur_manipulator"  


            self.move_group_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            ).add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        """Handle the response from the action server."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveGroup goal was rejected by the action server.')
            # Publish failure status
            movement_status = Bool()
            movement_status.data = False
            self.movement_publisher.publish(movement_status)
            return

        self.get_logger().info('MoveGroup goal accepted by the action server. Waiting for result...')
        # Get the result
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the result from the action server."""
        result = future.result().result
        error_code = result.error_code.val

        movement_status = Bool()

        if error_code == 1:
            self.get_logger().info('Robot successfully moved to the grasp pose.')
            movement_status.data = True
        else:
            self.get_logger().error(f'Failed to move the robot. Error Code: {error_code}')
            movement_status.data = False

        # Publish the movement status
        self.movement_publisher.publish(movement_status)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server."""
        feedback = feedback_msg.feedback

def main(args=None):
    rclpy.init(args=args)

    # Initialize RobotControllerMoveIt2 node
    robot_controller = RobotControllerMoveIt2()

    # Define a target pose (example values)
    target_pose = Pose()
    target_pose.position.x = 0.7196
    target_pose.position.y = -0.0126
    target_pose.position.z = 0.84
    target_pose.orientation.x = 0.2594
    target_pose.orientation.y = 0.6453
    target_pose.orientation.z = 0.4282
    target_pose.orientation.w = 0.5771

    # Manually call the grasp_pose_callback to simulate receiving a pose
    robot_controller.grasp_pose_callback(target_pose)

    try:
        # Spin to allow for callbacks
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Shutting down RobotControllerMoveIt2 node.')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# def main(args=None):
#     rclpy.init(args=args)
#     robot_controller = RobotControllerMoveIt2()

#     try:
#         rclpy.spin(robot_controller)
#     except KeyboardInterrupt:
#         robot_controller.get_logger().info('Shutting down RobotControllerMoveIt2 node.')
#     finally:
#         robot_controller.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

