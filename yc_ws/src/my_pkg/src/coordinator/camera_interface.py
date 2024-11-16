#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_srvs.srv import Trigger

class CameraInterface(Node):
    def __init__(self, node_name='camera_interface', save_directory='captured_images'):
        super().__init__(node_name)

        # Camera intrinsic parameters
        self.fx = 642.4262770792711
        self.fy = 642.3461744750167
        self.cx = 647.5434733474444
        self.cy = 373.3602344467871
        self.scale = 1000.0  # Scale for depth image conversion

        # Create a service to save images
        self.save_service = self.create_service(Trigger, 'save_images', self.save_images_callback)
        self.get_logger().info('Save Images service ready.')

        # Subscribers for color and depth images
        self.color_subscriber = Subscriber(self, Image, '/Realsense_D455/color/image_raw')
        self.depth_subscriber = Subscriber(self, Image, '/Realsense_D455/aligned_depth_to_color/image_raw')

        # Synchronize the color and depth image topics
        self.sync = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.listener_callback)

        self.br = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.image_count = 0
        self.save_directory = save_directory

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.get_logger().info(f'{node_name} initialized and synchronized.')

    def listener_callback(self, color_msg, depth_msg):
        self.get_logger().debug('Receiving synchronized image frames.')

        # Convert color and depth images using CvBridge
        try:
            self.color_image = self.br.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.depth_image = self.br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.get_logger().debug('Images successfully converted.')
        except Exception as e:
            self.get_logger().error(f'Failed to process images: {e}')

    def save_images(self):
        if self.color_image is not None and self.depth_image is not None:
            color_image_filename = self._get_next_filename('color_image', extension='.png')
            depth_image_filename = self._get_next_filename('depth_image', extension='.png')

            self.get_logger().info(f'Saving images: {color_image_filename}, {depth_image_filename}')

            # Save color and depth images
            try:
                cv2.imwrite(color_image_filename, self.color_image)
                # For depth images, it's better to save in a format that preserves depth information
                # Here, we'll save as 16-bit PNG if the depth image is in that format
                if self.depth_image.dtype == np.uint16:
                    cv2.imwrite(depth_image_filename, self.depth_image)
                else:
                    # If depth image is not 16-bit, convert it appropriately
                    depth_normalized = cv2.normalize(self.depth_image, None, 0, 65535, cv2.NORM_MINMAX)
                    depth_normalized = depth_normalized.astype(np.uint16)
                    cv2.imwrite(depth_image_filename, depth_normalized)
                self.get_logger().info(f'Saved RGB image as {color_image_filename}')
                self.get_logger().info(f'Saved depth image as {depth_image_filename}')
            except Exception as e:
                self.get_logger().error(f'Failed to save images: {e}')
        else:
            self.get_logger().warning('Images are not ready for saving.')

    def _get_next_filename(self, prefix, extension='.png'):
        while True:
            filename = os.path.join(self.save_directory, f'{prefix}_{self.image_count:04d}{extension}')
            if not os.path.exists(filename):
                self.image_count += 1
                return filename
            self.image_count += 1

    # Additional methods to provide interfaces
    def get_current_color_image(self):
        return self.color_image

    def get_current_depth_image(self):
        return self.depth_image
    
    def save_images_callback(self, request, response):
        self.save_images()
        response.success = True
        response.message = 'Images saved successfully.'
        return response

def main(args=None):
    rclpy.init(args=args)
    camera_interface = CameraInterface()

    try:
        rclpy.spin(camera_interface)
    except KeyboardInterrupt:
        camera_interface.get_logger().info('Shutting down CameraInterface...')
    finally:
        camera_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
