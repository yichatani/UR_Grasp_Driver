import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional
import open3d as o3d


class VoxelGrid:
    """Voxel grid representation of obstacles"""
    def __init__(self, points: np.ndarray, voxel_size: float = 0.05):
        self.voxel_size = voxel_size
        voxel_coords = np.floor(points / voxel_size).astype(int)
        self.occupied_voxels = set(map(tuple, voxel_coords))

    def is_occupied(self, point: np.ndarray) -> bool:
        voxel = tuple(np.floor(point[:3] / self.voxel_size).astype(int))  # Only check XYZ
        return voxel in self.occupied_voxels

    def check_line(self, start: np.ndarray, end: np.ndarray) -> bool:
        direction = end - start
        length = np.linalg.norm(direction[:3])  # Only consider XYZ for length
        if length < 1e-6:
            return self.is_occupied(start)
        direction[:3] /= length  # Normalize translation part
        steps = int(np.ceil(length / self.voxel_size))
        
        # Interpolate both translation and rotation
        start_rot = R.from_euler('xyz', start[3:])
        end_rot = R.from_euler('xyz', end[3:])
        
        for i in range(steps + 1):
            t = i / steps
            point = start[:3] + t * (end[:3] - start[:3])  # Interpolate translation
            interpolated_rot = start_rot.slerp(t, end_rot)  # Interpolate rotation
            interpolated_pose = np.concatenate([point, interpolated_rot.as_euler('xyz')])
            if self.is_occupied(interpolated_pose):
                return True
        return False


class RRTStarPlanner:
    """RRT* path planner for 6D poses"""
    def __init__(self, 
                 obstacles: np.ndarray,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 robot_radius: float = 0.1,
                 voxel_size: float = 0.05,
                 max_step: float = 0.5,
                 rewire_radius: float = 1.0):
        self.obstacles = VoxelGrid(obstacles, voxel_size)
        self.bounds = bounds
        self.max_step = max_step
        self.rewire_radius = rewire_radius
        self.nodes = []
        self.parents = []
        self.costs = []
        self.kdtree = None

    def add_node(self, node: np.ndarray, parent_idx: int, cost: float):
        self.nodes.append(node)
        self.parents.append(parent_idx)
        self.costs.append(cost)
        if self.kdtree is None:
            self.kdtree = KDTree(np.array([node[:3]]))  # Only use XYZ for KDTree
        else:
            self.kdtree = KDTree(np.array([n[:3] for n in self.nodes]))

    def nearest_neighbor(self, point: np.ndarray) -> Tuple[int, np.ndarray]:
        idx = self.kdtree.query(point[:3])[1]  # Use XYZ for nearest neighbor search
        return idx, self.nodes[idx]

    def new_state(self, from_state: np.ndarray, to_state: np.ndarray) -> np.ndarray:
        direction = to_state[:3] - from_state[:3]
        distance = np.linalg.norm(direction)
        if distance < self.max_step:
            return to_state
        direction /= distance
        new_translation = from_state[:3] + self.max_step * direction

        # Interpolate rotation
        from_rot = R.from_euler('xyz', from_state[3:])
        to_rot = R.from_euler('xyz', to_state[3:])
        interpolated_rot = from_rot.slerp(self.max_step / distance, to_rot)

        return np.concatenate([new_translation, interpolated_rot.as_euler('xyz')])

    def is_valid_edge(self, from_state: np.ndarray, to_state: np.ndarray) -> bool:
        return not self.obstacles.check_line(from_state, to_state)

    def sample_state(self) -> np.ndarray:
        translation = np.random.uniform(self.bounds[0][:3], self.bounds[1][:3])
        rotation = np.random.uniform(-np.pi, np.pi, size=3)  # Random roll, pitch, yaw
        return np.concatenate([translation, rotation])

    def rewire(self, new_idx: int):
        new_node = self.nodes[new_idx]
        for i, node in enumerate(self.nodes):
            if i == new_idx:
                continue
            distance = np.linalg.norm(node[:3] - new_node[:3])  # Only use XYZ for rewire radius
            if distance < self.rewire_radius and self.is_valid_edge(new_node, node):
                new_cost = self.costs[new_idx] + distance
                if new_cost < self.costs[i]:
                    self.parents[i] = new_idx
                    self.costs[i] = new_cost

    def plan(self, 
             start: np.ndarray, 
             goal: np.ndarray, 
             max_iterations: int = 10000,
             goal_bias: float = 0.1) -> Optional[List[np.ndarray]]:
        self.nodes = []
        self.parents = []
        self.costs = []
        self.kdtree = None
        self.add_node(start, -1, 0.0)
        
        for _ in range(max_iterations):
            if np.random.random() < goal_bias:
                random_state = goal
            else:
                random_state = self.sample_state()

            nearest_idx, nearest_node = self.nearest_neighbor(random_state)
            new_state = self.new_state(nearest_node, random_state)
            if not self.is_valid_edge(nearest_node, new_state):
                continue

            new_cost = self.costs[nearest_idx] + np.linalg.norm(nearest_node[:3] - new_state[:3])
            self.add_node(new_state, nearest_idx, new_cost)
            self.rewire(len(self.nodes) - 1)

            if np.linalg.norm(new_state[:3] - goal[:3]) < self.max_step:
                if self.is_valid_edge(new_state, goal):
                    self.add_node(goal, len(self.nodes) - 1, new_cost + np.linalg.norm(new_state[:3] - goal[:3]))
                    return self.extract_path(len(self.nodes) - 1)

        return None

    def extract_path(self, goal_idx: int) -> List[np.ndarray]:
        path = []
        current_idx = goal_idx
        while current_idx != -1:
            path.append(self.nodes[current_idx])
            current_idx = self.parents[current_idx]
        return path[::-1]


def visualize_path(path, obstacles):
    lines = [[i, i+1] for i in range(len(path)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([p[:3] for p in path]),  # Only visualize XYZ
        lines=o3d.utility.Vector2iVector(lines)
    )
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(obstacles))
    o3d.visualization.draw_geometries([pcd, line_set])


# Example usage
if __name__ == "__main__":
    # Example point cloud
    pcd = o3d.io.read_point_cloud("obstacles.pcd")
    obstacles = np.asarray(pcd.points)

    # Define start, goal, and bounds
    start = np.array([0, 0, 0, 0, 0, 0])  # [x, y, z, roll, pitch, yaw]
    goal = np.array([5, 5, 5, 0, 0, 0])   # [x, y, z, roll, pitch, yaw]
    bounds = (np.array([-1, -1, -1, -np.pi, -np.pi, -np.pi]), 
              np.array([6, 6, 6, np.pi, np.pi, np.pi]))

    # Initialize RRT* planner
    planner = RRTStarPlanner(obstacles, bounds, robot_radius=0.1, voxel_size=0.05, max_step=0.5, rewire_radius=1.0)
    path = planner.plan(start, goal, max_iterations=5000, goal_bias=0.2)

    if path is not None:
        print("Path found with", len(path), "nodes")
        visualize_path(path, obstacles)
    else:
        print("No path found")
