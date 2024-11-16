#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
// #include <moveit/core/moveit_error_code.hpp> // Correct header
#include <cstdio>
#include <iostream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"

#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit/planning_scene_interface/planning_scene_interface.h"

#include "moveit/robot_model_loader/robot_model_loader.h"
#include "moveit/robot_model/robot_model.h"
#include "moveit/robot_state/robot_state.h"

#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <Eigen/Geometry>




#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>  // For quaternion conversions

int main(int argc, char** argv)
{
    const double Pi = 3.14159265358979323846264;

    // Initialize ROS 2
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr m_Node;

    m_Node = rclcpp::Node::make_shared("robot_runner", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    moveit::planning_interface::MoveGroupInterface m_Mgi = moveit::planning_interface::MoveGroupInterface(m_Node, "ur_manipulator");
    moveit::planning_interface::PlanningSceneInterface m_Psi;

    // Adjust speed
    double velocity_scaling = 0.2;  // 20% of the maximum velocity
    double acceleration_scaling = 0.05;  // 5% of the maximum acceleration
    m_Mgi.setMaxVelocityScalingFactor(velocity_scaling);
    m_Mgi.setMaxAccelerationScalingFactor(acceleration_scaling);
    m_Mgi.setPlanningTime(10.0);  // Increase planning time to 10 seconds
    m_Mgi.setStartStateToCurrentState();

    // Moving to a 6D pose target
    std::cout << "Setting 6D pose target" << std::endl;
    geometry_msgs::msg::Pose target_pose;

    // Define target position
    target_pose.position.x = -0.683;  // Adjust this value as per your target
    target_pose.position.y = 0.132;  // Adjust this value as per your target
    target_pose.position.z = 0.344;  // Adjust this value as per your target

    // Define target orientation in roll, pitch, yaw (rx, ry, rz)
    double roll = 2.2;  // Roll angle in radians
    double pitch = 2.1; // Pitch angle in radians
    double yaw = -0.07; // Yaw angle in radians

    // Convert RPY to quaternion
    tf2::Quaternion quaternion;
    quaternion.setRPY(roll, pitch, yaw);

    // Assign quaternion to pose orientation
    target_pose.orientation.x = quaternion.x();
    target_pose.orientation.y = quaternion.y();
    target_pose.orientation.z = quaternion.z();
    target_pose.orientation.w = quaternion.w();

    m_Mgi.setPoseTarget(target_pose);
    moveit::planning_interface::MoveGroupInterface::Plan Plan;
    bool success = (m_Mgi.plan(Plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (success) {
        success = (m_Mgi.execute(Plan) == moveit::core::MoveItErrorCode::SUCCESS);
    }

    if (success) {
        std::cout << "Successfully moved to 6D pose target" << std::endl;
    } else {
        std::cout << "Failed to move to 6D pose target" << std::endl;
    }

    // Shutdown ROS 2
    rclcpp::shutdown();
    return 0;
}


// int main(int argc, char** argv)
// {
//     const double Pi = 3.14159265358979323846264;
//     const double toRadians = Pi / 180.0;
//     // const double toDegrees = 180.0 / Pi;

//     // Initialize ROS 2
//     rclcpp::init(argc, argv);
//     rclcpp::Node::SharedPtr m_Node;
//     // rclcpp::Logger m_Logger;
//     std::cout << "I am aaa" << std::endl;
//     // moveit::planning_interface::MoveGroupInterface m_Mgi;
//     m_Node = rclcpp::Node::make_shared("robot_runner", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
//     moveit::planning_interface::MoveGroupInterface m_Mgi = moveit::planning_interface::MoveGroupInterface(m_Node, "ur_manipulator");
//     std::cout << "I am bbb" << std::endl;
//     moveit::planning_interface::PlanningSceneInterface m_Psi = moveit::planning_interface::PlanningSceneInterface();
//     std::cout << "I am ccc" << std::endl;
//     std::cout << "I am here" << std::endl;
//     // Create a ROS 2 node
    
//     // Allow some time for parameters to be available
//     // rclcpp::sleep_for(std::chrono::seconds(2));
    
//         // Adjust speed
//     double velocity_scaling = 0.2;  // 50% of the maximum velocity
//     double acceleration_scaling = 0.05;  // 50% of the maximum acceleration
//     m_Mgi.setMaxVelocityScalingFactor(velocity_scaling);
//     m_Mgi.setMaxAccelerationScalingFactor(acceleration_scaling);
//     m_Mgi.setPlanningTime(10.0);  // Increase planning time to 10 seconds
//     m_Mgi.setStartStateToCurrentState();


//     //moving stuff
//     std::cout << "I am aaa" << std::endl;
//     bool success = false;
//     m_Mgi.clearPoseTarget();
//     std::cout << "Going home in joint space" << std::endl;
//     std::vector<double> Joints_home(6);
//     //-26, -91, 84, -81, -87, -22
//     Joints_home[0] =  -28 * toRadians;
//     Joints_home[1] = -91.0 * toRadians;
//     Joints_home[2] =  84.0 * toRadians;
//     Joints_home[3] = -81 * toRadians;
//     Joints_home[4] = -87.0 * toRadians;
//     Joints_home[5] = -22.0 * toRadians;
    

//     m_Mgi.setJointValueTarget(Joints_home);
//     moveit::planning_interface::MoveGroupInterface::Plan Plan;
//     success = (m_Mgi.plan(Plan) == moveit::core::MoveItErrorCode::SUCCESS);
//     if (success){
//     success = (m_Mgi.execute(Plan)== moveit::core::MoveItErrorCode::SUCCESS);
//     }
//     std::cout << "finished home in joint space" << std::endl;




//     // Shutdown ROS 2
//     rclcpp::shutdown();
//     return 0;
// }





    // rclcpp::spin(node);
    // try
    // {
    //     // Create the MoveGroupInterface for the "ur_manipulator" planning group
        
    //     // // Optional: Wait until MoveGroup is ready
    //     // if (!move_group.waitForServer(std::chrono::seconds(10)))
    //     // {
    //     //     RCLCPP_ERROR(node->get_logger(), "MoveGroup server not available.");
    //     //     return 1;
    //     // }
        
    //     // // Optionally, set planning parameters
    //     // move_group.setPlanningTime(10.0);
    //     // move_group.setNumPlanningAttempts(10);
    //     // move_group.setMaxVelocityScalingFactor(0.1);
    //     // move_group.setMaxAccelerationScalingFactor(0.1);
        
    //     // Define the target pose
    //     geometry_msgs::msg::Pose target_pose;
    //     target_pose.position.x = 0.5;  // I need to djust these values as needed
    //     target_pose.position.y = 0.0;
    //     target_pose.position.z = 0.5;
    //     target_pose.orientation.w = 1.0;
    //     target_pose.orientation.x = 0.0;
    //     target_pose.orientation.y = 0.0;
    //     target_pose.orientation.z = 0.0;
        
    //     // Set the target pose
    //     move_group.setPoseTarget(target_pose);
        
    //     // Plan to the target pose
    //     moveit::planning_interface::MoveGroupInterface::Plan plan;
    //     bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        
    //     if (success)
    //     {
    //         RCLCPP_INFO(node->get_logger(), "Planning successful. Executing the plan...");
            
    //         // Execute the planned trajectory
    //         moveit::core::MoveItErrorCode exec_success = move_group.execute(plan);
    //         if (exec_success == moveit::core::MoveItErrorCode::SUCCESS)
    //         {
    //             RCLCPP_INFO(node->get_logger(), "Motion executed successfully!");
    //         }
    //         else
    //         {
    //             RCLCPP_ERROR(node->get_logger(), "Motion execution failed.");
    //         }
    //     }
    //     else
    //     {
    //         RCLCPP_ERROR(node->get_logger(), "Planning failed.");
    //     }
    // }
    // catch (const std::exception &ex)
    // {
    //     RCLCPP_FATAL(node->get_logger(), "Exception: %s", ex.what());
    //     rclcpp::shutdown();
    //     return 1;
    // }
    


// int main(int argc, char** argv)
// {
//     // Initialize ROS 2
//     rclcpp::init(argc, argv);
//     auto node = rclcpp::Node::make_shared("robot_runner");
//     moveit::planning_interface::MoveGroupInterface move_group(node, "ur_manipulator");

//     // Define the target pose using the provided translation and quaternion
//     geometry_msgs::msg::Pose target_pose;
//     target_pose.position.x = 0.0712011;       // Translation x
//     target_pose.position.y = -0.03391694;     // Translation y
//     target_pose.position.z = 0.60179048;      // Translation z
//     target_pose.orientation.x = -0.3543387;   // Quaternion x
//     target_pose.orientation.y = -0.40190282;  // Quaternion y
//     target_pose.orientation.z = -0.55839211;  // Quaternion z
//     target_pose.orientation.w = 0.63333757;    // Quaternion w

//     // Set the target pose
//     move_group.setPoseTarget(target_pose);

//     // Plan to the target pose
//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

//     if (success)
//     {
//         std::cout << "Planning successful. Executing the plan..." << std::endl;
        
//         // Execute the planned trajectory
//         moveit::core::MoveItErrorCode exec_success = move_group.execute(plan);
//         if (exec_success == moveit::core::MoveItErrorCode::SUCCESS)
//         {
//             std::cout << "Motion executed successfully!" << std::endl;
//         }
//         else
//         {
//             std::cout << "Motion execution failed." << std::endl;
//         }
//     }
//     else
//     {
//         std::cout << "Planning failed." << std::endl;
//     }

//     // Shutdown ROS 2
//     rclcpp::shutdown();
//     return 0;
// }

// int main(int argc, char** argv)
// {
//     const double Pi = 3.14159265358979323846264;
//     const double toRadians = Pi / 180.0;
//     // const double toDegrees = 180.0 / Pi;

//     // Initialize ROS 2
//     rclcpp::init(argc, argv);
//     rclcpp::Node::SharedPtr m_Node;
//     // rclcpp::Logger m_Logger;
//     std::cout << "I am aaa" << std::endl;
//     // moveit::planning_interface::MoveGroupInterface m_Mgi;
//     m_Node = rclcpp::Node::make_shared("robot_runner", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
//     moveit::planning_interface::MoveGroupInterface m_Mgi = moveit::planning_interface::MoveGroupInterface(m_Node, "ur_manipulator");
//     std::cout << "I am bbb" << std::endl;
//     moveit::planning_interface::PlanningSceneInterface m_Psi = moveit::planning_interface::PlanningSceneInterface();
//     std::cout << "I am ccc" << std::endl;
//     std::cout << "I am here" << std::endl;
//     // Create a ROS 2 node
    
//     // Allow some time for parameters to be available
//     // rclcpp::sleep_for(std::chrono::seconds(2));
    