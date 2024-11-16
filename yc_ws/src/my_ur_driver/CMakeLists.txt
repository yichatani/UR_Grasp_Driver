
set(CMAKE_POLICY_WARNING_CMP0167 OFF)
#cmake_policy(SET CMP0167 NEW)

cmake_minimum_required(VERSION 3.8)
project(my_ur_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(ur_robot_driver REQUIRED)
find_package(moveit_ros_move_group REQUIRED)

#find_package(controller_manager REQUIRED)
#find_package(universal_robot REQUIRED)
#find_package(rclcpp REQUIRED)
#find_package(geometry_msgs REQUIRED)
#find_package(sensor_msgs REQUIRED)
#find_package(ur_robot_driver REQUIRED)  # Use ur_robot_driver
#find_package(Universal_Robots_ROS2_Driver REQUIRED)  # Use ur_robot_driver
#find_package(Universal_Robots_ROS2_Description REQUIRED)

#find_package(ur_robot_driver REQUIRED)

# Install launch files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()



# Install ur config files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/my_ur_config")
  install(DIRECTORY my_ur_config/
    DESTINATION share/${PROJECT_NAME}/my_ur_config
  )
endif()

# Install moveit config files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/my_moveit_config")
  install(DIRECTORY my_moveit_config/
    DESTINATION share/${PROJECT_NAME}/my_moveit_config
  )
endif()





if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
