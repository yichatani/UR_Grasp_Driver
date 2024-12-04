cmake_minimum_required(VERSION 3.8)
project(find_grasp)

# 使用ament_python来构建Python功能包
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_package()
