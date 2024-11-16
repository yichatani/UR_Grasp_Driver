import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import json  
from myframe import Myframe
#import pyrealsense2 as rs  
import cv2 
import numpy as np
import os  
import threading
import keyboard  
import time
import sys
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


def quaternion_to_rotation_matrix(q):
    rotation = R.from_quat(q)
    return rotation.as_matrix()


import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
import rclpy
from rclpy.node import Node
import rtde_receive

#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
#from ur_rtde import RTDEReceiveInterface
import sys




class GraspPoseTransformer(Node):
    def __init__(self):
        super().__init__('grasp_pose_transformer')
        
        self.get_logger().info('Initializing Grasp Pose Transformer Node...')
        
        self.UR10E_IP = "192.168.56.101"
        self.tf_buffer = tf2_ros.Buffer()

        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.StaticTransformBroadcaster(self)
        
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.UR10E_IP)
            self.rtde_c = rtde_control.RTDEControlInterface(self.UR10E_IP)
            self.get_logger().info(f"Connected to UR10e at {self.UR10E_IP}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to UR10e at {self.UR10E_IP}: {e}")
            sys.exit(1)
        
        
        # Get the hand-eye Calibration Transformation (tool0 to camera)
        self.T_tool0_wrt_camera = self.get_handeye_transform()
        
        # Grasp Rotation from anygrasp as 3x3 matrix and Translation as [x, y, z]
        self.T_grasp_wrt_camera = self.define_grasp_pose()
        
        # Perform Transformation
        self.T_grasp_wrt_base = self.compute_grasp_in_base()
        
        self.display_transformation()
        self.save_transformation_to_json()

        self.cleanup()
    
    def get_handeye_transform(self):
        """
        My hand-eye calibration transformation from tool0 to camera.
        """

        # My Hand-Eye Calibration Result: tool0 to Camera (Realsense_D455_link)
        myhandeye_xyzquat = [
            -0.04793565821499937,
            -0.0578564661503661,   
            0.07338666566373077,    
            0.49568395919661706,    
            -0.4956504619966394,    
            0.5045117470947331,    
            0.5040792885659003      
        ]
        
        translation = np.array(myhandeye_xyzquat[:3])  # [x, y, z]
        quaternion = np.array(myhandeye_xyzquat[3:])    # [qx, qy, qz, qw]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        
        # Construct Homogeneous Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        
        #self.get_logger().info("Hand-Eye Calibration Transformation (tool0 to camera):")
        #self.get_logger().info(f"Translation: {translation}")
        #self.get_logger().info(f"Rotation Matrix:\n{rotation_matrix}")
        
        return T
    
    def define_grasp_pose(self):
        """
        The grasp pose relative to the camera frame.
        """
        # # From anygrasp
        # grasp_translation = np.array([0.00560609, 0.1253016,  0.48300001])  # in meters
        # grasp_rotation_matrix = np.array([
        #     [ 0.0568925 ,  0.14536799 ,-0.98774052],
        #     [-0.43903548 , 0.89219284 , 0.10601819],
        #     [ 0.89666671 , 0.42762148 , 0.11458079]])
        

        # grasp_translation = np.array([-0.089886  ,  0.02040699 , 0.53200001])  # in meters
        # grasp_rotation_matrix = np.array([[ 0.10401868, -0.67549032, -0.72999519],
        #                                   [ 0.09528963,  0.73736888, -0.66873544],
        #                                   [ 0.99000001,  0.,          0.14106737]])
        
        
#         grasp_translation = np.array([-0.01140973,  0.02011757,  0.50400001])  # in meters
#         grasp_rotation_matrix = np.array([[-0.01589504,  0.99617106,  0.08596864],
#  [-0.18111572, -0.08742573,  0.97956824],
#  [ 0.98333335,  0.,          0.18181187]])
        
        grasp_translation = np.array([-0.01171036,  0.02007274,  0.484 ])  # in meters
        grasp_rotation_matrix = np.array([[-0.01589504,  0.99617106,  0.08596864],
 [-0.18111572, -0.08742573,  0.97956824],
 [ 0.98333335,  0.,          0.18181187]])
        




        # Construct Homogeneous Transformation Matrix for Grasp Pose
        T_grasp = np.eye(4)
        T_grasp[:3, :3] = grasp_rotation_matrix
        T_grasp[:3, 3] = grasp_translation


        ori_target_F = Myframe.from_Tmat(T_grasp)
        ori_target_tf = ori_target_F.as_transform("Realsense_D455_color_optical_frame", "target_tf", self)
        
        tfs_to_send = []
        rotate_y_only_F = Myframe.from_rotation_only('y', 90)
        rotate_z_only_F = Myframe.from_rotation_only('z', 90)

        

        final_target_F = ori_target_F.pose_trans(rotate_y_only_F)
        final_target_tf = final_target_F.as_transform("Realsense_D455_color_optical_frame", "final_tf", self)

        tfs_to_send.append(ori_target_tf)
        tfs_to_send.append(final_target_tf)

        final_target2_F = final_target_F.pose_trans(rotate_z_only_F)

        final_target2_tf = final_target2_F.as_transform("Realsense_D455_color_optical_frame", "final2_tf", self)

        tfs_to_send.append(final_target2_tf)

        self.br.sendTransform(tfs_to_send)


        #exit()
        # elvin here

        # R_x = np.eye(4)
        # R_x[:3, :3] = [[1,0,0], [0,0,1], [0, -1, 0]]

        # print("R_x: ")

        # print(R_x)

        # print()

        # print("frame: ")
        # print(Myframe.from_rotation_only('x', 90).R.as_matrix())
        # print()

        # R_y = np.eye(4)
        # R_y[:3, :3] = [[0,0,-1], [0,1,0], [1, 0, 0]]

        # print("R_y: ")

        # print(R_y)

        # print()

        # R = R_x @ R_y

        # print("R: ")

        # print(R)
        # print()

        #print(final_target_F.Tm)
        #time.sleep(100)
        #exit()

        # grasp_translation = np.array([0.00560609, 0.1253016,  0.48300001])  # in meters
        # grasp_rotation_matrix = np.array([
        #             [ 0.0568925 ,  0.14536799 ,-0.98774052],
        #             [-0.43903548 , 0.89219284 , 0.10601819],
        #             [ 0.89666671 , 0.42762148 , 0.11458079]])
                
        #         # Construct Homogeneous Transformation Matrix for Grasp Pose
        # T_grasp = np.eye(4)
        # T_grasp[:3, :3] = grasp_rotation_matrix
        # T_grasp[:3, 3] = grasp_translation

        # print()
        # print("T_grasp:")

        # print(T_grasp)

        # print()
        # print("final: ")

        # final = R @ T_grasp
        # print(final)

        print("final_target_F.Tmat: ")
        #rotate_y_only_F.as_transform("Realsense_D455_color_optical_frame", "any", self)

        #self.get_logger().info("Grasp Pose Relative to Camera:")
        #self.get_logger().info(f"Translation: {grasp_translation}")
        #self.get_logger().info(f"Rotation Matrix:\n{grasp_rotation_matrix}")
        
        print(final_target2_F.Tmat)
        T_grasp = final_target2_F.Tmat
        
        return T_grasp
    
    def get_tool0_wrt_base_transform(self):

        try:

            tcp_pose = self.rtde_r.getActualTCPPose()
            self.get_logger().info(f"Current TCP Pose (meters and rotation vector): {tcp_pose}")
            
            translation = np.array(tcp_pose[:3])
            rotation_vector = np.array(tcp_pose[3:6]) 
            
            # Convert rotation vector to rotation matrix
            rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
            
            # Construct Homogeneous Transformation Matrix
            T_tool0_wrt_base = np.eye(4)
            T_tool0_wrt_base[:3, :3] = rotation_matrix
            T_tool0_wrt_base[:3, 3] = translation
            
            self.get_logger().info("Transformation from tool0 to base:")
            self.get_logger().info(f"Translation: {translation}")
            self.get_logger().info(f"Rotation Matrix:\n{rotation_matrix}")
            
            return T_tool0_wrt_base
        
        except Exception as e:
            self.get_logger().error(f"Failed to retrieve tool0 to base transform: {e}")
            sys.exit(1)
    
    def compute_grasp_in_base(self):
        """
        Finally calculate the grasp pose in the base frame using the transformation chain:
        T_grasp_wrt_base = T_tool0_wrt_base × T_camera_wrt_tool0 × T_grasp_wrt_camera
        """

        # Step 1: Get T_tool0_wrt_base
        T_tool0_wrt_base = self.get_tool0_wrt_base_transform()
        
        # Step 2: Get T_camera_wrt_tool0 by inverting T_tool0_wrt_camera
        #T_camera_wrt_tool0 = np.linalg.inv(self.T_tool0_wrt_camera)

        T_camera_wrt_tool0 = self.T_tool0_wrt_camera
        for _ in range(10):
            rclpy.spin_once(self)
            try:
                tool_to_color_opt_frame_tf = self.tf_buffer.lookup_transform("tool0", "Realsense_D455_color_optical_frame", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
                break
            except:
                print("lookup has problems")
        
        print(tool_to_color_opt_frame_tf)
        T_camera_wrt_tool0 = Myframe.from_transformstamped(tool_to_color_opt_frame_tf).Tmat
        

        #self.get_logger().info("Transformation from camera to tool0 (inverted hand-eye calibration):")
        self.get_logger().info("Transformation from camera to tool0 (hand-eye calibration):")
        self.get_logger().info(f"Transformation Matrix:\n{T_camera_wrt_tool0}")
        
        # Step 3: Get T_grasp_wrt_camera (already defined with rotation matrix)
        T_grasp_wrt_camera = self.T_grasp_wrt_camera
        self.get_logger().info("Grasp Pose Transformation Matrix (relative to camera):")
        self.get_logger().info(f"Transformation Matrix:\n{T_grasp_wrt_camera}")
        
        # Step 4: Compute T_grasp_wrt_base = T_tool0_wrt_base × T_camera_wrt_tool0 × T_grasp_wrt_camera
        justabroker = np.matmul(T_tool0_wrt_base, T_camera_wrt_tool0)
        T_grasp_wrt_base = np.matmul(justabroker, T_grasp_wrt_camera)
        
        self.get_logger().info("Final Grasp Pose in Base Frame:")
        self.get_logger().info(f"Transformation Matrix:\n{T_grasp_wrt_base}")

        T = T_grasp_wrt_base
        position_base = T[:3, 3]
        rotation_matrix = T[:3, :3]
        rotation_quat = R.from_matrix(rotation_matrix).as_quat()
        rotation_euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        

        ori_target_F = Myframe.from_Tmat(T)
        ori_target_tf = ori_target_F.as_transform("base", "finallll", self)
        
        self.br.sendTransform(ori_target_tf)
        print("\n=== Final Grasp Pose in Base Frame ===")
        print("Position (meters):", position_base)
        print("Orientation (quaternion [qx, qy, qz, qw]):", rotation_quat)

        rotation_vector = R.from_quat(rotation_quat).as_rotvec()

        # Define the target pose
        target_pose = np.concatenate((position_base, rotation_vector))

        print("Target Pose for UR10e (meters and rotation vector):", target_pose)

        
        try:
            # Move the robot to the grasp pose
            print("Moving to Grasp Pose... Press 's' to stop.")
            #self.rtde_c.moveL(target_pose, 0.01, 0.02)  # [speed, acceleration] parameters can be adjusted
            print("Movement Command Sent.")
        except Exception as e:
            print(f"Movement interrupted or failed: {e}")
        
        return T_grasp_wrt_base
    
    
    def display_transformation(self):

        T = self.T_grasp_wrt_base
        translation = T[:3, 3]
        rotation_matrix = T[:3, :3]
        rotation_quat = R.from_matrix(rotation_matrix).as_quat()
        rotation_euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        #print("\n=== Final Grasp Pose in Base Frame ===")
        #print("Position (meters):", translation)
        #print("Orientation (quaternion [qx, qy, qz, qw]):", rotation_quat)
        #print("Orientation (Euler angles [deg]):", rotation_euler)
        #print("Transformation Matrix:\n", T)
    
    def cleanup(self):
        self.get_logger().info("Shutting down Grasp Pose Transformer Node...")
        self.rtde_r.disconnect()


    def save_transformation_to_json(self):
        
        T = self.T_grasp_wrt_base
        translation = T[:3, 3].tolist()
        rotation_matrix = T[:3, :3].tolist()
        rotation_quat = R.from_matrix(T[:3, :3]).as_quat().tolist()  # [qx, qy, qz, qw]
        rotation_euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True).tolist()
        
        grasp_pose = {
            "position": translation,
            "orientation_quaternion": rotation_quat,
            "orientation_euler_deg": rotation_euler,
            "transformation_matrix": rotation_matrix + [T[:3, 3].tolist()]  # Flatten matrix
        }
        
        output_file = "final_grasp_pose.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(grasp_pose, f, indent=4)
            self.get_logger().info(f"Final grasp pose saved to {output_file}")
            #print(f"\n=== Final Grasp Pose in Base Frame ===")
            #print(json.dumps(grasp_pose, indent=4))
        except Exception as e:
            self.get_logger().error(f"Failed to save final grasp pose to JSON: {e}")


    
def main(args=None):
    rclpy.init(args=args)
    grasp_transformer = GraspPoseTransformer()
    rclpy.shutdown()

if __name__ == '__main__':
    main()