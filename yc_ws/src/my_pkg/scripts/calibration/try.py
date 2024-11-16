#!/usr/bin/env python3

import numpy as np
from spatialmath import SE3
import json

# -------------------------- Step 1: Define Calibration Data --------------------------

# Calibration data as provided
calibration_data = {
    "kinematics": {
        "shoulder": {
            "x": 0,
            "y": 0,
            "z": 0.18058515099439337,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 2.2602480301774737e-07
        },
        "upper_arm": {
            "x": 8.3485644710547946e-05,
            "y": 0,
            "z": 0,
            "roll": 1.5702545696548049,
            "pitch": 0.0,
            "yaw": 4.1047584906868243e-06
        },
        "forearm": {
            "x": -0.61220312894318329,
            "y": 0,
            "z": 0,
            "roll": 0.00010654313820385056,
            "pitch": 6.6540968284754308e-05,
            "yaw": -4.3829526419064191e-06
        },
        "wrist_1": {
            "x": -0.5713742389854275,
            "y": 0.0016578240048675445,
            "z": 0.17362039553304554,
            "roll": 3.132044388142369,
            "pitch": -3.1412222367350666,
            "yaw": -3.1415878510741835
        },
        "wrist_2": {
            "x": 1.2667172807438902e-05,
            "y": -0.11977766949035959,
            "z": 6.0743796004779998e-05,
            "roll": 1.5702891889364319,
            "pitch": 0.0,
            "yaw": 7.201977154270855e-08
        },
        "wrist_3": {
            "x": -0.00013447451577126986,
            "y": 0.11558585398615148,
            "z": -2.8027014129265135e-05,
            "roll": 1.5705538489084301,
            "pitch": 3.1415926535897931,
            "yaw": 3.1415926089038959
        }
    },
    "hash": "calib_16372075385502550731"
}

# -------------------------- Step 2: Define Trajectories --------------------------

# Trajectories with joint angles in degrees
trajectories = {
    "traj1": [{"positions": [-26, -91, 84, -81, -87, -22]}],
    "traj2": [{"positions": [-26, -91, 85, -83, -87, 7]}],
    "traj3": [{"positions": [-26, -91, 84, -80, -89, -49]}],
    "traj4": [{"positions": [-26, -88, 82, -74, -91, -22]}],
    "traj5": [{"positions": [-27, -93.5, 89, -93, -83, -23.5]}],
    "traj6": [{"positions": [-29, -90, 82, -69, -72, -30.3]}],
    "traj7": [{"positions": [-29, -89, 81, -66.6, -73.3, -36.5]}],
    "traj8": [{"positions": [-29.3, -92.5, 84, -82, -69.5, 25]}],
    "traj9": [{"positions": [14, -83, 72, -69, -112, 29]}],
    "traj10": [{"positions": [13.6, -81.3, 68.5, -60, -111, 28]}],
    "traj11": [{"positions": [29.3, -69, 68, -66, -124, 50]}],
    "traj12": [{"positions": [29, -67, 63.4, -55.3, -119, 47]}],
    "traj13": [{"positions": [29, -69, 67, -62.6, -127, 68.3]}],
    "traj14": [{"positions": [-64.71, -85, 71, -75.5, -81, -59]}],
    "traj15": [{"positions": [-70, -64.5, 41.7, -46, -74.5, -61]}],
    "traj16": [{"positions": [-65.5, -59, 55.3, -61.3, -85, -53.5]}],
    "traj17": [{"positions": [-65.71, -52.7, 44, -47.5, -82, -56.3]}],
    "traj18": [{"positions": [-64.5, -57, 31, -27.5, -76.6, -57.2]}],
    "traj19": [{"positions": [-65.5, -60, 36.3, -33, -68.5, -89]}],
    "traj20": [{"positions": [-81, -73.5, 55.4, -34.4, -77.6, -102.3]}],
    "traj21": [{"positions": [-84, -80, 66.5, -53.4, -57.7, -66.8]}],
    "traj22": [{"positions": [-84, -65.8, 46.2, -50.8, -87.5, -58.7]}],
    "traj23": [{"positions": [-19, -87.6, 74.8, -63.6, -76, -21]}],
    "traj24": [{"positions": [-18.1, -82.8, 57.2, -65.8, -76, 34.5]}],
    "traj25": [{"positions": [-17.7, -84, 60, -72.5, -78.4, 61]}],
    "traj26": [{"positions": [-7, -87, 63.5, -75.2, -100.5, 81.5]}],
    "traj27": [{"positions": [-8.2, -85, 59.3, -65.2, -93.5, -93.2]}],
    "traj28": [{"positions": [6, -117.6, 98.7, -84.7, -104, -45]}],
    "traj29": [{"positions": [-35.4, -120.8, 115.5, -100, -79.9, -31.6]}]
}

# -------------------------- Step 3: Helper Functions --------------------------

def deg2rad(degrees):
    """
    Convert a list of angles from degrees to radians.
    """
    return [np.deg2rad(angle) for angle in degrees]

def rpy_to_se3(roll, pitch, yaw, x, y, z):
    """
    Convert roll, pitch, yaw (in radians) and position to an SE3 transformation.
    """
    return SE3.RPY([roll, pitch, yaw], order='xyz') * SE3(x, y, z)

# -------------------------- Step 4: Define Kinematic Chain --------------------------

# List of joint names in order
joint_names = ['shoulder', 'upper_arm', 'forearm', 'wrist_1', 'wrist_2', 'wrist_3']

# Create a list to hold the fixed transformations between joints
fixed_transforms = []

for name in joint_names:
    joint = calibration_data['kinematics'][name]
    T_fixed = rpy_to_se3(
        roll=joint['roll'],
        pitch=joint['pitch'],
        yaw=joint['yaw'],
        x=joint['x'],
        y=joint['y'],
        z=joint['z']
    )
    fixed_transforms.append(T_fixed)

# -------------------------- Step 5: Forward Kinematics Function --------------------------

def compute_fk(joint_angles_rad, fixed_transforms):
    """
    Compute the forward kinematics for the given joint angles.

    Parameters:
    - joint_angles_rad: List of joint angles in radians.
    - fixed_transforms: List of SE3 fixed transformations between joints.

    Returns:
    - SE3 object representing the end-effector pose.
    """
    if len(joint_angles_rad) != 6:
        raise ValueError("Expected 6 joint angles.")

    T = SE3()  # Initialize as identity

    for i in range(6):
        # Rotation about Z-axis for revolute joint
        T_rot = SE3.Rz(joint_angles_rad[i])

        # Fixed transformation
        T_fixed = fixed_transforms[i]

        # Update cumulative transformation
        T = T * T_rot * T_fixed

    return T

# -------------------------- Step 6: Convert Trajectories to Poses --------------------------

# Dictionary to store all poses
poses = {}

# Iterate over each trajectory
for traj_name, traj_data in trajectories.items():
    # Assuming each trajectory has only one set of positions
    joint_angles_deg = traj_data[0]["positions"]
    joint_angles_rad = deg2rad(joint_angles_deg)

    # Compute forward kinematics
    fk_result = compute_fk(joint_angles_rad, fixed_transforms)

    # Extract position
    position = fk_result.t  # [x, y, z]

    # Extract orientation as quaternion [w, x, y, z]
    orientation = fk_result.quaternion  # [w, x, y, z]

    # Combine into a 7D pose
    pose_7d = {
        "position": position.tolist(),
        "orientation": orientation.tolist()
    }

    # Store in poses dictionary
    poses[traj_name] = pose_7d

# -------------------------- Step 7: Save Poses to JSON --------------------------

# Specify the output file name
output_file = "ur10e_poses.json"

# Save the poses to the JSON file
with open(output_file, "w") as f:
    json.dump(poses, f, indent=4)

print(f"Conversion complete. Poses saved to '{output_file}'.")
