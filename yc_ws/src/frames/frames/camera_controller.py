import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import open3d as o3d
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ImageSubscriber():
    def __init__(self, coordinator):

        self.coordinator = coordinator

        # Subscribers for color and depth images
        self.color_subscriber = Subscriber(self.coordinator, Image, '/Realsense_D455/color/image_raw')
        self.depth_subscriber = Subscriber(self.coordinator, Image, '/Realsense_D455/aligned_depth_to_color/image_raw')

        self.sync = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.listener_callback)

        self.br = CvBridge()
        self.color_image = None
        self.depth_image = None
        

        self.coordinator.get_logger().info('ImageSubscriber initialized and synchronized.')

    def listener_callback(self, color_msg, depth_msg):
        #self.get_logger().info('Receiving synchronized image frames.')

        # Convert color and depth images using CvBridge
        try:
            self.color_image = self.br.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            self.depth_image = self.br.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            #self.coordinator.get_logger().info('Images successfully converted.')
        except Exception as e:
            self.get_logger().error(f'Failed to process images: {e}')
            return

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

            self.coordinator.get_logger().info('Visualizing point cloud with Open3D...')
            vis.run()
            vis.destroy_window()
    

    def get_current_images(self):
        """
        Retrieves the latest color and depth images with thread safety.

        Returns:
            tuple: (color_image, depth_image) as NumPy arrays or (None, None) if not available.
        """
        color = self.color_image.copy() if self.color_image is not None else None
        depth = self.depth_image.copy() if self.depth_image is not None else None
        #self.process_and_visualize_point_cloud()

        return color, depth























# def main(args=None):
#     rclpy.init(args=args)
#     image_subscriber = ImageSubscriber()

#     try:
#         rclpy.spin(image_subscriber)
#     except KeyboardInterrupt:
#         image_subscriber.get_logger().info('Shutting down...')
#     finally:
#         image_subscriber.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
