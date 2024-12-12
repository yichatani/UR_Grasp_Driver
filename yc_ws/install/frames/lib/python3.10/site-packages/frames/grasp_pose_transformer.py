# grasp_pose_transformer.py

import sys
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from myframe import Myframe
import rclpy
import tf2_ros
#from tf_transformations import quaternion_matrix, euler_matrix


class GraspPoseTransformer:
    def __init__(self, coordinator):

        self.coordinator = coordinator

        self.base_to_camera_transformation_done = False        
        self.base_to_camera = np.zeros((4,4), dtype=float)  # Transformation matrix from base to camera

    def define_grasp_pose(self, grasp_pose):
        """
        The grasp pose relative to the camera frame.
        """

        grasp_translation, grasp_rotation_matrix = grasp_pose.translation, grasp_pose.rotation_matrix

        # Construct Homogeneous Transformation Matrix for Grasp Pose
        T_grasp = np.eye(4)
        T_grasp[:3, :3] = grasp_rotation_matrix
        T_grasp[:3, 3] = grasp_translation

        ori_target_F = Myframe.from_Tmat(T_grasp)
        #ori_target_tf = ori_target_F.as_transform("Realsense_D455_color_optical_frame", "target_tf", self)
        
        rotate_y_only_F = Myframe.from_rotation_only('y', 90)
        rotate_z_only_F = Myframe.from_rotation_only('z', 90)

        final_target_F = ori_target_F.pose_trans(rotate_y_only_F)

        final_target2_F = final_target_F.pose_trans(rotate_z_only_F)

        T_grasp = final_target2_F.Tmat
        
        return T_grasp
    

    def get_base_to_camera_transformation(self):
        max_retries = 4 
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Lookup the transform from base to camera frame
                base_to_color_opt_frame_tf = self.coordinator.tf_buffer.lookup_transform(
                    "base",
                    "Realsense_D455_color_optical_frame",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                base_to_color_opt_frame = Myframe.from_transformstamped(base_to_color_opt_frame_tf).Tmat
                self.base_to_camera = base_to_color_opt_frame
                self.coordinator.get_logger().info('Transform from base to camera acquired successfully.\n')
                self.base_to_camera_transformation_done = True
                return True

            except Exception as e:
                retry_count += 1
                self.coordinator.get_logger().error(f'Failed to lookup transform: {e} (Attempt {retry_count}/{max_retries})\n')
                if retry_count < max_retries:
                    time.sleep(2)
                else:
                    self.coordinator.get_logger().error('Max retries reached. Transform lookup failed.\n')
                    return False
        return self.base_to_camera_transformation_done

        
    def trasform_graspPose_to_Base(self, selected_grasp):

        if self.base_to_camera_transformation_done is False:
            self.get_base_to_camera_transformation()

        self.coordinator.get_logger().info('Grasp detected. Transforming grasp pose to base frame...\n')

        # Transform the grasp pose to the base frame
        grasp_wrt_optical = self.define_grasp_pose(selected_grasp)
        grasp_wrt_base = np.matmul(self.base_to_camera, grasp_wrt_optical)  # Matrix multiplication for transformation

        if grasp_wrt_base is None:
            self.coordinator.get_logger().warn('Grasp pose transformation failed.\n')
            return False

        ori_target_F = Myframe.from_Tmat(grasp_wrt_base)  # Create a transformation frame from the grasp pose
        ori_target_tf = ori_target_F.as_transform("base", "finalll", self.coordinator)  # Convert to a TF message
        self.coordinator.buffer.sendTransform(ori_target_tf)  # Broadcast the transform
        target_pose = ori_target_F.as_UR_pose()  # Convert to UR robot pose format

        return target_pose