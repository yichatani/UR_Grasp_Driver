cmake_minimum_required(VERSION 3.8)
project(my_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ur_robot_driver REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(move_robot src/move_robot.cpp)

#### the tester cpp
target_include_directories(move_robot PUBLIC

  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(move_robot PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(move_robot "rclcpp" "moveit_ros_planning_interface")
#####

install(TARGETS move_robot DESTINATION lib/${PROJECT_NAME})


install(PROGRAMS
  scripts/calibration/calibration.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/mine/realsense_subscriber.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/mine/transformation.py
  DESTINATION lib/${PROJECT_NAME}
)

#install(PROGRAMS
#  scripts/anygrasp_bridging/anygrasp_application.py
#  DESTINATION lib/${PROJECT_NAME}
#)


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  install(DIRECTORY scripts/
    DESTINATION share/${PROJECT_NAME}/scripts
  )
endif()


project(my_pkg)



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
