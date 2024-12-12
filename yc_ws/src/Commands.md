# Commands


## First source the workspace
```bash
cd ~/abdo_ws
source install/setup.bash
```

## Launching the UR10e:
```bash
ros2 launch my_ur_driver my_ur.launch.py  # Modify the code to adjust your ur type and ip address
```

## Launching the Moveit:
```bash
ros2 launch my_ur_driver my_moveit.launch.py  # Modify the code to adjust your ur type
```

## Launching the robot on Simulation:
```bash
ros2 launch my_ur_driver my.launch.py use_fake_hardware:=true fake_execution:=true
```

## Launching the moveit on Simulation:
```bash
ros2 launch my_ur_driver my_moveit.launch.py use_fake_hardware:=true fake_execution:=true
```

## Runing the Realsense camera
```bash
ros2 launch my_ur_driver camera.launch.py
```

## Runing the Realsense camera with both aligned depth with color and point cloud enabled
```bash
ros2 launch my_ur_driver camera.launch.py align_depth.enable:=true pointcloud.enable:=true
```


## The gripper launch
```bash
ros2 launch robotiq_description robotiq_control.launch.py
```

## Closing the gripper
```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.6, max_effort: 50.0}}"
```

## Opening the gripper
```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.1, max_effort: 50.0}}"
```

**gripper limits: 0.06 or 0.1 (full open) - 0.6 (full close)**



## Install all ros packages or make sure they are installed
```bash
rosdep update
rosdep install --from-paths /opt/ros/humble/share --ignore-src -r -y
```

## Two ways to run python files

```bash
/usr/bin/python3.10 calibration.py
ros2 run my_pkg sub_realsense.py
```


## Gazebo Simulation
```bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py
ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py
```

## Runing Realsense camera
```bash
ros2 launch realsense2_camera rs_launch.py
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 pointcloud.enable:=true
```


#

# From official
## Launching the robot:
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.56.101
```
## Launching the Moveit!
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py  ur_type:=ur10e  
```
## Launching the robot on Simulation
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.56.101 use_fake_hardware:=true fake_execution:=true
```
## Launching the moveit on Simulation
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e use_fake_hardware:=true fake_execution:=true
```
## The Gripper Discription
```bash
#The gripper launch (NO NEED to launch again if already launched the my_ur_with_gripper.launch.py)
ros2 launch robotiq_description robotiq_control.launch.py
```




