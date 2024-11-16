import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import time

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.client = self.create_client(Trigger, 'save_images')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Save Images service not available, waiting...')

        self.timer = self.create_timer(5.0, self.send_save_request)  # Adjust the timer as needed

    def send_save_request(self):
        req = Trigger.Request()
        future = self.client.call_async(req)
        future.add_done_callback(self.save_response_callback)

    def save_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(response.message)
            else:
                self.get_logger().warning('Failed to save images.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()

    try:
        rclpy.spin(image_saver)
    except KeyboardInterrupt:
        image_saver.get_logger().info('Shutting down ImageSaver...')
    finally:
        image_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
