# gripper_controller.py

import rclpy
from rclpy.node import Node
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
import time
import threading

class GripperController:
    def __init__(self, coordinator):

        self.coordinator = coordinator 

        # Initialize Action Client for GripperCommand to communicate with the gripper
        self.action_client = ActionClient(self.coordinator, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        self.coordinator.get_logger().info('GripperController initialized and ready.')

    def send_gripper_command(self, position, max_effort):

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.coordinator.get_logger().error('Gripper action server not available.')
            return False 

        # Create a goal message with the desired position and max effort
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        #goal_msg.command.max_effort = max_effort

        # Log the gripper command that is being sent
        #self.coordinator.get_logger().info(f'Sending gripper command: position={position}, max_effort={max_effort}')

        # Send the goal asynchronously and add a callback for when the goal response is received
        self.action_client.send_goal_async(goal_msg, feedback_callback=None).add_done_callback(self.goal_response_callback)
        return True  

    def goal_response_callback(self, future):

        goal_handle = future.result()  # Get the result of the action (goal handle)

        if not goal_handle.accepted:
            self.coordinator.get_logger().error('Gripper command was rejected by the action server.')
            return

        # If the goal was accepted, log that the command was accepted and wait for the result
        self.coordinator.get_logger().info('Gripper command accepted. Waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)  # Wait for the result callback

    def get_result_callback(self, future):

        result = future.result().result  # Retrieve the result of the action
        # Log the result of the gripper command (position and max effort)
        #self.coordinator.get_logger().info(f'Gripper command result: position={result.position}, max_effort={result.max_effort}')

    def open_gripper(self):

        self.coordinator.get_logger().info('Opening the gripper ...') 

        # Define the gripper open position and maximum effort (can adjust based on gripper characteristics)
        open_position = 0.0  # 0.2 is a partially open position
        open_max_effort = 100.0  # Maximum effort to open the gripper

        # Send the open command to the gripper
        self.send_gripper_command(open_position, open_max_effort)

        # Wait for the gripper to open completely (adjust time based on the gripper's speed)
        time.sleep(1)  # 3 seconds is a rough estimate, adjust as needed

    def close_gripper(self):

        self.coordinator.get_logger().info('Closing the gripper to grasp the object...')  

        # Define the gripper close position (0.8 is fully closed) and maximum effort
        close_position = 0.8  # 0.8 is fully closed
        close_max_effort = 100.0  # Maximum effort to close the gripper

        # Send the close command to the gripper
        self.send_gripper_command(close_position, close_max_effort)

        # Wait for the gripper to close completely (adjust time based on the gripper's speed)
        time.sleep(3)  # 4 seconds is an estimate, adjust as necessary


    def adjust_gripper_width(self, grasp_object_width):

        self.open_gripper()
        self.coordinator.get_logger().info('Adjust the gripper width to grasp the object...')  

        position = 0.8 - grasp_object_width*10

        if position <= 0.2:
            position = 0.3
        elif position >= 0.5:
            position = 0.4
        
        max_effort = 100.0  # Maximum effort to close the gripper

        # Send the close command to the gripper
        self.send_gripper_command(position, max_effort)

        # Wait for the gripper to close completely (adjust time based on the gripper's speed)
        time.sleep(2)  # 4 seconds is an estimate, adjust as necessary
