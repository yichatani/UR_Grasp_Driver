#!/usr/bin/env python3

import json

#from torch import R
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Image
import math
import cv2
from cv_bridge import CvBridge
import os
import time
from rclpy.timer import Timer
#import rospy
from datetime import timedelta
import tf2_ros
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs
#import tf
import tf2_ros
#import tf_transformations
from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
#import rospy
#from geometry_msgs.msg import Pose
import numpy as np
#from ur_kinematics import URKinematics
#from ikpy.chain import Chain
#from ikpy.link import OriginLink, URDFLink
import numpy as np
#import pybullet as p


def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (q_x, q_y, q_z, q_w).
        """
        q_x = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        q_y = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        q_z = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        q_w = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        
        return [q_x, q_y, q_z, q_w]

def deg_to_rad(degrees):
    return [angle * math.pi / 180 for angle in degrees] 



class JTCClient(Node):
    def __init__(self):
        super().__init__('jtc_client')
        self.client = ActionClient(self, FollowJointTrajectory, 'scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.joint_state_publisher = self.create_publisher(JointState, '/joint_states', 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Ensure topic name is correct
            self.image_callback,
            10  # QoS history depth
        )
        self.subscription  # Prevent unused variable warning
        self.bridge = CvBridge()  # Initialize CV Bridge to convert ROS images to OpenCV format
        self.latest_image = None
        self.count = 0
        self.get_logger().info('Image Subscriber has started.')

                # Wait for the action server to be ready
        while not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server on scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.get_logger().info('Action server ready.')
        self.parse_trajectories()
        # self.images_folder = '/home/artc/abdo_moveit/pictures/'  # Path to save images
        # if not os.path.exists(self.images_folder):
        #     os.makedirs(self.images_folder)




    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image  # Store the latest image
            self.get_logger().info('Received image data')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def take_picture(self):
        # Check if there's a latest image received
        if self.latest_image is not None:
            # Specify the folder to save the images
            save_folder = "./images"  # Replace with the path to your folder

            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                self.get_logger().info(f"Folder created: {save_folder}")

            filename = os.path.join(save_folder, f"{self.count:02}.png")
            self.count = self.count + 1


            # Save the image to the specified folder
            cv2.imwrite(filename, self.latest_image)
            self.get_logger().info(f"Picture saved as {filename}")
        else:
            self.get_logger().warning("No image data received yet!")



    def parse_trajectories(self):
        self.joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        # Define the joint trajectories
        self.TRAJECTORIES = {
                # "traj0": [
                #     {
                #         "positions": [-35.4, -120.8, 115.5, -100, -79.9, -31.6]
                #     }
                # ],
                # "traj1": [
                #     {
                #         "positions": [-26, -91, 84, -81, -87, -22]
                #     }
                # ],
                # "traj2": [
                #     {
                #         "positions": [-26, -91, 85, -83, -87, 7]
                #     }
                # ],
                # "traj3": [
                #     {
                #         "positions": [-26, -91, 84, -80, -89, -49]

                #     }
                #  ],
                # "traj4": [
                #     {
                #         "positions": [-26, -88, 82, -74, -91, -22]
                #     }
                # ],
                # "traj5": [
                #     {
                #         "positions": [-27, -93.5, 89, -93, -83, -23.5]
                #     }
                # ],
                # "traj6": [
                #     {
                #         "positions": [-29, -90, 82, -69, -72, -30.3]
                #     }
                # ],
                # "traj7": [
                #     {
                #         "positions": [-29, -89, 81, -66.6, -73.3, -36.5]
                #     }
                # ],
                # "traj8": [
                #     {
                #         "positions": [-29.3, -92.5, 84, -82, -69.5, 25]
                #     }
                # ],
                # "traj9": [
                #     {
                #         "positions": [14, -83, 72, -69, -112, 29]
                #     }
                # ],
                # "traj10": [
                #     {
                #         "positions": [13.6, -81.3, 68.5, -60, -111, 28]
                #     }
                # ],
                # "traj11": [
                #     {
                #         "positions": [29.3, -69, 68, -66, -124, 50]
                #     }
                # ],
                # "traj12": [
                #     {
                #         "positions": [29, -67, 63.4, -55.3, -119, 47]
                #     }
                # ],
                # "traj13": [
                #     {
                #         "positions": [29, -69, 67, -62.6, -127, 68.3]
                #     }
                # ],
                # "traj14": [
                #     {
                #         "positions": [-64.71, -85, 71, -75.5, -81, -59]
                #     }
                # ],
                # "traj15": [
                #     {
                #         "positions": [-70, -64.5, 41.7, -46, -74.5, -61]
                #     }
                # ],
                # "traj16": [
                #     {
                #         "positions": [-65.5, -59, 55.3, -61.3, -85, -53.5]
                #     }
                # ],
                # "traj17": [
                #     {
                #         "positions": [-65.71, -52.7, 44, -47.5, -82, -56.3]
                #     }
                # ],
                # "traj18": [
                #     {
                #         "positions": [-64.5, -57, 31, -27.5, -76.6, -57.2]
                #     }
                # ],
                # "traj19": [
                #     {
                #         "positions": [-65.5, -60, 36.3, -33, -68.5, -89]
                #     }
                # ],
                # "traj20": [
                #     {
                #         "positions": [-81, -73.5, 55.4, -34.4, -77.6, -102.3]
                #     }
                # ],
                # "traj21": [
                #     {
                #         "positions": [-84, -80, 66.5, -53.4, -57.7, -66.8]
                #     }
                # ],
                # "traj22": [
                #     {
                #         "positions": [-84, -65.8, 46.2, -50.8, -87.5, -58.7]
                #     }
                # ],
                # "traj23": [
                #     {
                #         "positions": [-19, -87.6, 74.8, -63.6, -76, -21]
                #     }
                # ],
                # "traj24": [
                #     {
                #         "positions": [-18.1, -82.8, 57.2, -65.8, -76, 34.5]
                #     }
                # ],
                # "traj25": [
                #     {
                #         "positions": [-17.7, -84, 60, -72.5, -78.4, 61]
                #     }
                # ],
                # "traj26": [
                #     {
                #         "positions": [-7, -87, 63.5, -75.2, -100.5, 81.5]
                #     }
                # ],
                # "traj27": [
                #     {
                #         "positions": [-8.2, -85, 59.3, -65.2, -93.5, -93.2]
                #     }
                # ],
                # "traj28": [
                #     {
                #         "positions": [6, -117.6, 98.7, -84.7, -104, -45]
                #     }
                # ],
                # "traj29":[        

                #     {
                #         "positions": [-35.4, -120.8, 115.5, -100, -79.9, -31.6]
                #     }
                # ]
        }



        # Iterate over the trajectory points and add missing time_from_start and velocities
        for traj_name, traj_points in self.TRAJECTORIES.items():
            for i, point in enumerate(traj_points):
                if 'time_from_start' not in point:
                    point['time_from_start'] = Duration()

                if 'velocities' not in point:
                    point['velocities'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Iterate over the trajectory points again and calculate cumulative time
        for traj_name, traj_points in self.TRAJECTORIES.items():
            for i in range(len(traj_points)):
                if i == 0:
                    traj_points[0]["time_from_start"].sec = 6
                    traj_points[0]["time_from_start"].nanosec = 0
                else:
                    traj_points[i]["time_from_start"].sec = traj_points[i - 1]["time_from_start"].sec + 8
                    traj_points[i]["time_from_start"].nanosec = traj_points[i - 1]["time_from_start"].nanosec

        #self.TRAJECTORIES[len(self.TRAJECTORIES)][len(self.TRAJECTORIES[len(self.TRAJECTORIES)])]['time_from_start'] = Duration()
        #self.TRAJECTORIES[len(self.TRAJECTORIES)][len(self.TRAJECTORIES[len(self.TRAJECTORIES)])]['velocities'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



        # # Set initial time for the first trajectory point
        # for i in range(len(self.TRAJECTORIES["traj0"])):
        #     if 'time_from_start' not in self.TRAJECTORIES["traj0"][i]:
        #         self.TRAJECTORIES["traj0"][i]['time_from_start'] = Duration()
        #     if 'velocities' not in self.TRAJECTORIES["traj0"][i]:
        #         self.TRAJECTORIES["traj0"][i]['velocities'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # # Iterate over the trajectory points and calculate cumulative time
        



        # for i in range(len(self.TRAJECTORIES["traj0"]) - 1):
        #     if i == 0:
        #         self.TRAJECTORIES["traj0"][0]["time_from_start"].sec = 1
        #         self.TRAJECTORIES["traj0"][0]["time_from_start"].nanosec = 0
        #     self.TRAJECTORIES["traj0"][i + 1]["time_from_start"].sec = self.TRAJECTORIES["traj0"][i]["time_from_start"].sec + 2 # Adding time (8 seconds in this case)
        #     self.TRAJECTORIES["traj0"][i + 1]["time_from_start"].nanosec = self.TRAJECTORIES["traj0"][i]["time_from_start"].nanosec
        #     #self.get_logger().info(f"Time for traj {i+1}: {self.TRAJECTORIES['traj0'][i+1]['time_from_start'].sec} seconds")

        #         # Set initial time for the first trajectory point
        # for i in range(len(self.TRAJECTORIES["traj1"])):
        #     if 'time_from_start' not in self.TRAJECTORIES["traj1"][i]:
        #         self.TRAJECTORIES["traj1"][i]['time_from_start'] = Duration()
        #     if 'velocities' not in self.TRAJECTORIES["traj1"][i]:
        #         self.TRAJECTORIES["traj1"][i]['velocities'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # # Iterate over the trajectory points and calculate cumulative time
        # for i in range(len(self.TRAJECTORIES["traj1"]) - 1):
        #     if i == 0:
        #         self.TRAJECTORIES["traj1"][0]["time_from_start"].sec = 1
        #         self.TRAJECTORIES["traj1"][0]["time_from_start"].nanosec = 0
        #     self.TRAJECTORIES["traj1"][i + 1]["time_from_start"].sec = self.TRAJECTORIES["traj1"][i]["time_from_start"].sec + 2 # Adding time (8 seconds in this case)
        #     self.TRAJECTORIES["traj1"][i + 1]["time_from_start"].nanosec = self.TRAJECTORIES["traj1"][i]["time_from_start"].nanosec
        #     #self.get_logger().info(f"Time for traj {i+1}: {self.TRAJECTORIES['traj0'][i+1]['time_from_start'].sec} seconds")


        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)



        # Convert the joint positions in each trajectory to radians
        for traj_name, traj_data in self.TRAJECTORIES.items():
            for waypoint in traj_data:
                #print(waypoint["positions"])
                waypoint["positions"] = deg_to_rad(waypoint["positions"])
                print(waypoint["positions"])

                # # Calculate forward kinematics
                # cartesian_position, quaternion_orientation = self.calculate_fk(waypoint["positions"])
                # if cartesian_position and quaternion_orientation:
                #     print(f'7D Pose -> Position: {cartesian_position}, Orientation (qx, qy, qz, qw): {quaternion_orientation}')


        #self.get_logger().info(self.TRAJECTORIES.items())

        # Execute each trajectory
        for traj_name, points in self.TRAJECTORIES.items():
            self.execute_trajectory(traj_name, points)
            #time.sleep(5)



    # #Use TF2 to calculate forward kinematics
        
    # def publish_joint_states(self, joint_positions):
    #     joint_state_msg = JointState()
    #     joint_state_msg.name = self.joints
    #     joint_state_msg.position = joint_positions
    #     joint_state_msg.header.stamp = self.get_clock().now().to_msg()

    #     # Publish joint states
    #     self.joint_state_publisher.publish(joint_state_msg)
    #     self.get_logger().info(f'Published joint states: {joint_positions}')

    # # Use TF2 to calculate forward kinematics using joint positions
    # def calculate_fk(self, joint_positions):
    #     # Publish joint positions to update the robot's state
    #     self.publish_joint_states(joint_positions)
        
    #     # Introduce a short delay to allow TF tree to update
    #     time.sleep(1.0)  # Adjust this delay if necessary

    #     # Now look up the transform between base and tool0 after applying joint positions
    #     try:
    #         # Wait for the transform to be available (allow time for TF to update)
    #         self.tf_buffer.can_transform('base','tool0',  rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0))

    #         # Lookup the transform from base to tool0
    #         trans: TransformStamped = self.tf_buffer.lookup_transform('base', 'tool0', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=5.0))

    #         # Cartesian position (X, Y, Z)
    #         position = (trans.transform.translation.x,
    #                     trans.transform.translation.y,
    #                     trans.transform.translation.z)

    #         # Orientation (as quaternion qx, qy, qz, qw)
    #         orientation_quat = (trans.transform.rotation.x,
    #                             trans.transform.rotation.y,
    #                             trans.transform.rotation.z,
    #                             trans.transform.rotation.w)

    #         # Return the 7D pose (X, Y, Z, qx, qy, qz, qw)
    #         return position, orientation_quat

    #     except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as ex:
    #         self.get_logger().error(f"Could not get transform: {ex}")
    #         return None, None
                



    def execute_trajectory(self, traj_name, points):
            self.get_logger().info(f'Executing trajectory {traj_name}')
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = self.joints

            for pt in points:
                point = JointTrajectoryPoint()
                point.positions = pt["positions"]
                point.velocities = pt["velocities"]

                # Create a Duration message for time_from_start
                duration_msg = Duration()
                duration_msg.sec = pt["time_from_start"].sec
                duration_msg.nanosec = pt["time_from_start"].nanosec
                point.time_from_start = duration_msg  # Set the Duration object

                if not self.is_within_limits(deg_to_rad(point.positions)):
                    self.get_logger().error(f'Trajectory {traj_name} has position limits violations.')
                    return
                goal_msg.trajectory.points.append(point)


                # Send the goal to the action server asynchronously
            send_goal_future = self.client.send_goal_async(goal_msg)
            #self.client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
                # Wait for the goal response
            rclpy.spin_until_future_complete(self, send_goal_future)
            #goal_handle = send_goal_future.result()
            self.save_poses_to_file2("pose.json")

                # Wait for the goal to complete before continuing to the next trajectory
                # goal_handle = future.result()
                # if not goal_handle.accepted:
                #     self.get_logger().error(f'Trajectory {traj_name} was rejected.')
                #     return

                # result_future = goal_handle.get_result_async()
                # result = result_future.result()

                # if result.status == GoalStatus.STATUS_SUCCEEDED:
                #     self.get_logger().info(f'Trajectory {traj_name} executed successfully.')
                # else:
                #     self.get_logger().error(f'Trajectory {traj_name} failed with status {result.status}')
                
                # pause before starting the next trajectory
            time.sleep(4)  # Adjust the pause time as needed
            #input("hit enter to continue")
            rclpy.spin_once(self)  # Spin once to process a single callback
            # Now, take a picture at this point (after receiving one image)
            self.take_picture()
        

    def is_within_limits(self, positions):
        # Implement logic to check if the positions are within the robot's limits
        lower_limits = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
        upper_limits = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]

        for pos, low, high in zip(positions, lower_limits, upper_limits):
            if pos < low or pos > high:
                return False
        return True
    
    def goal_response_callback(self, future):
        goal_handle = future.result()

        # Check if the goal was accepted by the action server
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            return

        self.get_logger().info('Goal accepted.')

        # Use the goal handle to asynchronously get the result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Trajectory executed successfully.')
        else:
            self.get_logger().error('Trajectory execution failed.')



    # def extract_poses(self):
    #     # Extract poses from the trajectory data
    #     poses = []  # List to store the poses
    #     for traj_name, traj_data in self.TRAJECTORIES.items():
    #         for waypoint in traj_data:
    #             # Extract the positions (which represent the pose in joint space)
    #             poses.append(waypoint["positions"])
        
    #     return poses

    

    def extract_poses(self):
        # Extract poses from the trajectory data and convert to quaternion
        poses = []  # List to store the poses
        for traj_name, traj_data in self.TRAJECTORIES.items():
            for waypoint in traj_data:
                # Extract the positions (which represent the pose in joint space)
                position = waypoint["positions"][:3]  # [x, y, z]
                roll = waypoint["positions"][3]  # Roll
                pitch = waypoint["positions"][4]  # Pitch
                yaw = waypoint["positions"][5]  # Yaw

                rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)

                print(rotation_matrix)
                # Calculate the quaternion
                quaternion = rotation_matrix_to_quaternion(rotation_matrix)

                # Position can be extracted or calculated; here we will assume (x, y, z) = (0, 0, 0)
                # Modify this part based on your robot's configuration and end-effector position
                #x, y, z = 0, 0, 0  # Example values, replace with actual end-effector position if known

                # Create the final 7D representation
                #seven_d_representation = np.concatenate((position, quaternion))
                # Convert Euler angles (roll, pitch, yaw) to quaternion
                #quaternion = euler_to_quaternion(roll, pitch, yaw)
                #r = R.from_euler('xyz', waypoint["positions"][3:])  # 'xyz' corresponds to roll, pitch, yaw order
                #quaternion = r.as_quat()  # Convert to quaternion

                # Append pose with quaternion
                #poses.append( position + quaternion)

                pose = np.concatenate((position, quaternion))
                poses.append(pose)
                #self.get_logger().info(poses)
        return poses

    def save_poses_to_file(self, filename):
        # Extract poses
        poses = self.extract_poses()
        # Open file for writing
        with open(filename, 'w') as file:
            # Write headers
            file.write("pose_frame_id\t\"base\"\n")
            file.write("image_frame_id\t\"Realsense_D455_color_optical_frame\"\n")
            file.write("poses\n")

            # Write each pose in the desired format
            for idx, pose in enumerate(poses):
                file.write(f"{idx}\n")  # Write the index of the pose
                for i, position in enumerate(pose):
                    file.write(f"{i}\t{position}\n")  # Write each position with its index
                file.write("\n")  # Add an empty line between poses

        print(f"Poses saved to {filename}")

    def save_poses_to_file2(self, filename):
        # Extract poses
        poses = self.extract_poses()
        poses1 = [pose.tolist() for pose in poses]  # Convert each ndarray to a list

        # Prepare the data to be saved as JSON
        data = {
            "pose_frame_id": "base",
            "image_frame_id": "Realsense_D455_color_optical_frame",
            "poses": poses1
        }

        # Open file for writing JSON data
        with open(filename, 'w') as file:
            # Use json.dump to write the data to a file in JSON format
            json.dump(data, file, indent=4)

        self.get_logger().info(f"Poses saved to {filename}")




def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    # Calculate rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    return R


# Convert rotation matrix to quaternion
def rotation_matrix_to_quaternion(R):
    q = np.zeros(4)
    q_w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q_x = (R[2, 1] - R[1, 2]) / (4 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4 * q_w)
    return np.array([q_w, q_x, q_y, q_z])

# Calculate the quaternion

def main(args=None):
    rclpy.init(args=args)
    #for i in range(2):
    jtc_client = JTCClient()
    #rclpy.spin(jtc_client)
    jtc_client.save_poses_to_file2("poses.json")
    rclpy.shutdown()



if __name__ == '__main__':
    main()
