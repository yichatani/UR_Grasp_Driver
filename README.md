# Portable UR_Grasp_Driver

The workspace is a nearly complete robot arm control system based on vision. It contains packages not only like Moveit, and RTDE, but also contains some self-built framework to better handle ros2. 

It is built under Ros2 humble.

Hopefully, it can help to build a workspace faster, rather than build from the beginning every time. 

The following is some commands you need. 

### Compile the workspace

```sh
cd yc_ws
colcon build
source install/setup.bash
```

### Launching the robot:
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.56.101
```

### Launching the Moveit!
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py  ur_type:=ur10e  
```

### Launching the robot on Simulation
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.56.101 use_fake_hardware:=true fake_execution:=true
```
### Launching the moveit on Simulation
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e use_fake_hardware:=true fake_execution:=true
```

### Runing Realsense camera
```bash
ros2 launch realsense2_camera rs_launch.py
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 pointcloud.enable:=true
```
