# gripper_controller.py

import rclpy
from rclpy.node import Node
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
import time
import threading

class GripperController:
    def __init__(self, coordinator):
        """
        Initializes the GripperController.

        Args:
            coordinator (Coordinator): Instance of Coordinator node for logging and interactions.
        """
        self.coordinator = coordinator 

        # Initialize Action Client for GripperCommand to communicate with the gripper
        self.action_client = ActionClient(self.coordinator, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        # Log that the gripper controller is initialized
        self.coordinator.get_logger().info('GripperController initialized and ready.')

    def send_gripper_command(self, position, max_effort):
        """
        Sends a GripperCommand action goal to the gripper.

        Args:
            position (float): Desired position (0.0 to 0.8). 0.0 means fully open, 0.8 is fully closed.
            max_effort (float): Maximum effort (0.0 to 255.0) that the gripper should exert.

        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        # Wait for the action server to be available, with a timeout of 5 seconds
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.coordinator.get_logger().error('Gripper action server not available.')
            return False  # If the action server is not available, return False

        # Create a goal message with the desired position and max effort
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        #goal_msg.command.max_effort = max_effort

        # Log the gripper command that is being sent
        #self.coordinator.get_logger().info(f'Sending gripper command: position={position}, max_effort={max_effort}')

        # Send the goal asynchronously and add a callback for when the goal response is received
        self.action_client.send_goal_async(goal_msg, feedback_callback=None).add_done_callback(self.goal_response_callback)
        return True  # Command successfully sent, return True

    def goal_response_callback(self, future):
        """
        Callback for the goal response.

        Args:
            future (Future): The future associated with the goal response.
        """
        goal_handle = future.result()  # Get the result of the action (goal handle)

        if not goal_handle.accepted:
            # If the goal was rejected by the action server, log the error
            self.coordinator.get_logger().error('Gripper command was rejected by the action server.')
            return

        # If the goal was accepted, log that the command was accepted and wait for the result
        self.coordinator.get_logger().info('Gripper command accepted. Waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)  # Wait for the result callback

    def get_result_callback(self, future):
        """
        Callback for the result of the action.

        Args:
            future (Future): The future associated with the result.
        """
        result = future.result().result  # Retrieve the result of the action
        # Log the result of the gripper command (position and max effort)
        #self.coordinator.get_logger().info(f'Gripper command result: position={result.position}, max_effort={result.max_effort}')

    def open_gripper(self):
        """
        Controls the gripper to open after the robot has moved to the grasp pose.
        """
        self.coordinator.get_logger().info('Opening the gripper ...')  # Log action

        # Define the gripper open position and maximum effort (can adjust based on gripper characteristics)
        open_position = 0.0  # 0.2 is a partially open position
        open_max_effort = 100.0  # Maximum effort to open the gripper

        # Send the open command to the gripper
        self.send_gripper_command(open_position, open_max_effort)

        # Wait for the gripper to open completely (adjust time based on the gripper's speed)
        time.sleep(1)  # 3 seconds is a rough estimate, adjust as needed

    def close_gripper(self):
        """
        Controls the gripper to close after the robot has moved to the grasp pose.
        """
        self.coordinator.get_logger().info('Closing the gripper to grasp the object...')  # Log action

        # Define the gripper close position (0.8 is fully closed) and maximum effort
        close_position = 0.8  # 0.8 is fully closed
        close_max_effort = 100.0  # Maximum effort to close the gripper

        # Send the close command to the gripper
        self.send_gripper_command(close_position, close_max_effort)

        # Wait for the gripper to close completely (adjust time based on the gripper's speed)
        time.sleep(3)  # 4 seconds is an estimate, adjust as necessary