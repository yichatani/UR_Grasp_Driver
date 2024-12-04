# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Pose
# from moveit_commander import MoveGroupCommander, PlanningSceneInterface

# class UR10eController(Node):
#     def __init__(self):
#         super().__init__('ur10e_controller')

#         # Initialize MoveIt 2 interfaces
#         self.move_group = MoveGroupCommander("manipulator")

#         # Define a target pose
#         target_pose = Pose()
#         target_pose.position.x = 0.4
#         target_pose.position.y = 0.0
#         target_pose.position.z = 0.4
#         target_pose.orientation.w = 1.0  # Neutral orientation

#         # Move to the target pose
#         self.move_to_pose(target_pose)

#     def move_to_pose(self, pose: Pose):
#         self.move_group.set_pose_target(pose)
#         plan = self.move_group.go(wait=True)
#         self.move_group.stop()  # Ensure no residual movement
#         self.move_group.clear_pose_targets()  # Clear targets
#         if plan:
#             self.get_logger().info("Target pose reached successfully!")
#         else:
#             self.get_logger().error("Failed to reach the target pose.")

# def main(args=None):
#     rclpy.init(args=args)
#     node = UR10eController()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup

class UR10eController(Node):
    def __init__(self):
        super().__init__('ur10e_controller')

        # Action client to send a goal to MoveIt2's MoveGroup action server
        self._action_client = ActionClient(self, MoveGroup, '/move_group')

        # Define a target pose
        target_pose = Pose()
        target_pose.position.x = 0.4
        target_pose.position.y = 0.0
        target_pose.position.z = 0.4
        target_pose.orientation.w = 1.0  # Neutral orientation

        # Move to the target pose
        self.send_goal(target_pose)

    def send_goal(self, pose):
        # Wait until action server is available
        if not self._action_client.wait_for_server(timeout_sec=7.0):
                self.get_logger().error('MoveGroup action server not available!')
                return
        # Create a goal message
        goal_msg = MoveGroup.Goal()
        goal_msg.request.goal_constraints[0].position_constraints.append(pose)

        # Send the goal to the action server
        self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return

        self.get_logger().info('Goal accepted.')
        goal_handle.result().add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback):
        self.get_logger().info(f'Received feedback: {feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info('Target pose reached successfully!')
        else:
            self.get_logger().error(f'Failed to reach the target pose, error code: {result.error_code.val}')

def main(args=None):
    rclpy.init(args=args)
    ur10e_controller = UR10eController()
    rclpy.spin(ur10e_controller)
    ur10e_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
