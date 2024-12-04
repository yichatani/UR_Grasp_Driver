import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import json  


class URSubscriber(Node):
    def __init__(self):
        super().__init__('ur_subscriber')

        self.frequency = 0.5
        self.timer_period = 1.0 / self.frequency 

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # self.data = {
        #     "joint_state": None,
        #     "velocity": None,
        #     "end_pose": None,
        #     "depth_image": None,
        #     "color_image": None,
        #     #"camera_intrinsics": None,
        # }

        self.data = {
            "joint_state": {"positions": [], "velocities": []},
            "velocity": {"linear": [], "angular": []},
            "end_pose": {"translation": [], "rotation": []},
            "depth_image": None,
            "color_image": None,
        }

        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(TwistStamped, '/tool_velocity', self.velocity_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.create_subscription(Image, '/Realsense_D455/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/Realsense_D455/color/image_raw', self.color_callback, 10)
        # self.create_subscription(CameraInfo, '/camera/info', self.camera_info_callback, 10)

    def joint_callback(self, msg):
        self.data["joint_state"] = {
            "positions": np.array(msg.position).tolist(),
            "velocities": np.array(msg.velocity).tolist(),
        }
        if self.data["joint_state"] == {
            "positions": [],
            "velocities": [],
        }:
            self.data["joint_state"] = {"positions": [23455555], "velocities": [88777864836]}

    def velocity_callback(self, msg):
        self.data["velocity"] = {
            "linear": [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
            "angular": [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z],
        }
        if self.data["velocity"] == {"linear": [], "angular": []}:
            self.data["velocity"] = {"linear": [2222222], "angular": [333333333]}


    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "tool0":
                self.data["end_pose"] = {
                    "translation": [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    "rotation": [
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w,
                    ],
                }
        if  self.data["end_pose"] == {"translation": [], "rotation": []}:
            self.data["end_pose"] = {"translation": [2222,2222], "rotation": [22222,2222]}


    def depth_callback(self, msg):
        depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self.data["depth_image"] = depth_array.tolist()
        if self.data["depth_image"] == None:
            self.data["depth_image"] = [23455555, 88777864836]

    def color_callback(self, msg):
        color_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.data["color_image"] = color_array.tolist()
        if self.data["color_image"] == None:
            self.data["color_image"] = [23455555, 88777864836]

    # def camera_info_callback(self, msg):
    #     self.data["camera_intrinsics"] = msg.k

    def timer_callback(self):
        if all(self.data.values()):
            # change to JSON
            data_json = json.dumps(self.data, indent=4)
            self.get_logger().info(f"Formatted data for diffusion policy:\n{data_json}")

            # Selection：send data to diffusion policy to process
            # diffusion_policy.process(self.data)
        else:
            self.get_logger().info("Waiting for all data to be available...")


def main():
    rclpy.init()
    node = URSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
