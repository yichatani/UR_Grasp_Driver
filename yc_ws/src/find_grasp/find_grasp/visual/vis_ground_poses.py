import cv2
import open3d as o3d
from graspnetAPI import GraspNet


sceneId, annId = 16, 0

# initialize a GraspNet instance
graspnet_root = '/media/zhy/Data2TB1/GraspNet1B'
g = GraspNet(graspnet_root, camera='realsense', split='train')

# load grasps of scene 1 with annotation id = 3, camera = realsense and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'realsense', fric_coef_thresh = 0.3)
# print('6d grasp:\n{}'.format(_6d_grasp))

# visualize the grasps using open3d
geometries = []
geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = 'realsense'))

geometries += _6d_grasp[:].to_open3d_geometry_list()
# geometries += _6d_grasp.sort_by_score()[:500].random_sample(numGrasp = 100).to_open3d_geometry_list()
o3d.visualization.draw_geometries(geometries)