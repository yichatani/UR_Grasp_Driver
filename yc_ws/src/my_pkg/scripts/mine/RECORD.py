import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage


class URSubscriber(Node):
    def __init__(self):
        super().__init__('ur_subscriber')

        # Set frequency
        self.frequency = 10
        self.timer_period = 1.0 / self.frequency  # Timer interval in seconds

        # Use a timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.joint_state_data = None
        self.velocity_data = None
        self.tf_data = None
        self.depth_data = None
        self.color_data = None
        self.camera_info_data = None

        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        #self.create_subscription(TwistStamped, '/tool_velocity', self.velocity_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.create_subscription(Image, '/Realsense_D455/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/Realsense_D455/color/image_raw', self.color_callback, 10)
        # self.create_subscription(CameraInfo, '/camera/info', self.camera_info_callback, 10)

    def joint_callback(self, msg):
        self.joint_state_data = msg

    def velocity_callback(self, msg):
        self.velocity_data = msg


    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "tool0":
                self.tf_data = transform
                self.get_logger().info(f"Received Tool0: {self.tf_data}")
            # self.get_logger().info(f"Received TF: {msg.transforms}")


    def depth_callback(self, msg):
        self.depth_data = msg

    def color_callback(self, msg):
        self.color_data = msg

    def camera_info_callback(self, msg):
        self.camera_info_data = msg

    def timer_callback(self):
        if self.joint_state_data:
            self.get_logger().info(f"joint_state: {self.joint_state_data.position}, velocity: {self.joint_state_data.velocity}")
        if self.velocity_data:
            self.get_logger().info(f"end_velocity: linear={self.velocity_data.twist.linear}, angular={self.velocity_data.twist.angular}")
        if self.tf_data:
            self.get_logger().info(
                f"end_pose: translation={self.tf_data.transform.translation}, rotation={self.tf_data.transform.rotation}"
            )
        if self.depth_data:
            self.get_logger().info(f"depth: {len(self.depth_data.data)} bytes")
        if self.color_data:
            self.get_logger().info(f"colors: {len(self.color_data.data)} bytes")
        if self.camera_info_data:
            self.get_logger().info(f"camera_intrisinc: {self.camera_info_data.k}")
        self.get_logger().info("\n")

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
