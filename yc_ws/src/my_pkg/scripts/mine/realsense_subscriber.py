import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import open3d as o3d
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Camera intrinsic parameters
        self.fx, self.fy = 642.72, 642.66
        self.cx, self.cy = 647.91, 373.42
        self.scale = 1000.0  # Scale for depth image conversion

        # Set workspace limits for point cloud filtering
        self.xmin, self.xmax = -0.19, 0.12
        self.ymin, self.ymax = 0.02, 0.15
        self.zmin, self.zmax = 0.0, 1.0
        self.lims = [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]

        # Subscribers for color and depth images
        self.color_subscriber = Subscriber(self, Image, '/Realsense_D455/color/image_raw')
        self.depth_subscriber = Subscriber(self, Image, '/Realsense_D455/aligned_depth_to_color/image_raw')

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
        self.save_directory = 'captured_images2'

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.get_logger().info('ImageSubscriber initialized and synchronized.')

    def listener_callback(self, color_msg, depth_msg):
        self.get_logger().info('Receiving synchronized image frames.')

        print(color_msg.header.frame_id)
        print(depth_msg.header.frame_id)

        # Convert color and depth images using CvBridge
        try:
            self.color_image = self.br.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.depth_image = self.br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.get_logger().info('Images successfully converted.')
        except Exception as e:
            self.get_logger().error(f'Failed to process images: {e}')
            return

        # Process and visualize the point cloud within the callback
        self.process_and_visualize_point_cloud()

    def process_and_visualize_point_cloud(self):
        if self.color_image is not None and self.depth_image is not None:
            # Normalize color image for visualization
            color_image_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Create point cloud from depth image
            xmap, ymap = np.meshgrid(np.arange(self.depth_image.shape[1]), np.arange(self.depth_image.shape[0]))
            points_z = self.depth_image / self.scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            # Filter points based on workspace limits
            mask = (points_z > 0) & (points_z < self.zmax)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = color_image_rgb[mask]

            # Create Open3D point cloud
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)

            # Apply transformation if necessary (for correct orientation)
            trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            cloud.transform(trans_mat)

            # Visualize the point cloud and handle key events
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name='Point Cloud Visualization')

            vis.add_geometry(cloud)

            # Add key callback to save images when 's' is pressed
            def save_callback(vis):
                self.get_logger().info('Key "s" pressed. Saving images...')
                self.save_images()

            vis.register_key_callback(ord('S'), save_callback)

            # Add key callback to close visualization when 'q' is pressed
            def close_callback(vis):
                self.get_logger().info('Key "q" pressed. Exiting visualization...')
                vis.destroy_window()

            vis.register_key_callback(ord('Q'), close_callback)

            self.get_logger().info('Visualizing point cloud with Open3D...')
            vis.run()
            vis.destroy_window()

    def save_images(self):
        if self.color_image is not None and self.depth_image is not None:
            # Determine the next available filenames for color and depth images
            color_image_filename = self._get_next_filename('color_image')
            depth_image_filename = self._get_next_filename('depth_image')

            self.get_logger().info(f'Saving images: {color_image_filename}, {depth_image_filename}')

            # Save color and depth images
            try:
                cv2.imwrite(color_image_filename, self.color_image)
                cv2.imwrite(depth_image_filename, self.depth_image)
                self.get_logger().info(f'Saved RGB image as {color_image_filename}')
                self.get_logger().info(f'Saved depth image as {depth_image_filename}')
            except Exception as e:
                self.get_logger().error(f'Failed to save images: {e}')
        else:
            self.get_logger().warning('Images are not ready for saving.')

    def _get_next_filename(self, prefix):
        """Generate the next available filename with the format: prefix_XX.png"""
        i = 0
        while True:
            filename = os.path.join(self.save_directory, f'{prefix}_{i:02d}.png')
            if not os.path.exists(filename):
                return filename
            i += 1

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info('Shutting down...')
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
