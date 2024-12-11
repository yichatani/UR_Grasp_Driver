import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import copy
import open3d as o3d


class VoxelGrid:
    """Voxel grid representation of obstacles"""
    
    def __init__(self, points: np.ndarray, voxel_size: float = 0.05):
        """
        Initialize voxel grid from point cloud
        
        Args:
            points: Nx3 array of point cloud coordinates
            voxel_size: Size of each voxel
        """
        self.voxel_size = voxel_size
        
        # Convert points to voxel coordinates
        voxel_coords = np.floor(points / voxel_size).astype(int)
        
        # Create dictionary of occupied voxels
        self.occupied_voxels = set(map(tuple, voxel_coords))
        
        # Calculate bounds
        self.min_bounds = np.min(points, axis=0)
        self.max_bounds = np.max(points, axis=0)
        
    def is_occupied(self, point: np.ndarray) -> bool:
        """Check if point is in occupied voxel"""
        voxel = tuple(np.floor(point / self.voxel_size).astype(int))
        return voxel in self.occupied_voxels
        
    def check_line(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if line segment intersects any occupied voxels"""
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return self.is_occupied(start)
            
        direction = direction / length
        steps = int(np.ceil(length / self.voxel_size))
        
        for i in range(steps + 1):
            point = start + i * self.voxel_size * direction
            if self.is_occupied(point):
                return True
                
        return False


class RRTPlanner:
    """RRT path planner with point cloud obstacles"""
    
    def __init__(self, 
                 obstacles: np.ndarray,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 robot_radius: float = 0.1,
                 voxel_size: float = 0.05,
                 max_step: float = 0.5):
        """
        Initialize RRT planner
        
        Args:
            obstacles: Nx3 array of obstacle points
            bounds: Tuple of min and max bounds for sampling
            robot_radius: Radius for collision checking
            voxel_size: Size of voxels for obstacle representation
            max_step: Maximum step size for tree extension
        """
        # Expand obstacles by robot radius
        expanded_points = []
        phi = np.linspace(0, 2 * np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        phi, theta = np.meshgrid(phi, theta)
        x = robot_radius * np.sin(theta) * np.cos(phi)
        y = robot_radius * np.sin(theta) * np.sin(phi)
        z = robot_radius * np.cos(theta)
        sphere_points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        for point in obstacles:
            expanded_points.extend(point + sphere_points)
            
        self.obstacles = VoxelGrid(np.array(expanded_points), voxel_size)
        self.bounds = bounds
        self.max_step = max_step
        
        # Initialize tree with dynamic KDTree
        self.nodes = []
        self.parents = []
        self.kdtree = None  # Placeholder for KDTree
        
    def add_node(self, node: np.ndarray, parent_idx: int = -1):
        """Add node to tree and update KDTree incrementally"""
        self.nodes.append(node)
        self.parents.append(parent_idx)
        if self.kdtree is None:
            self.kdtree = KDTree(np.array([node]))
        else:
            self.kdtree = KDTree(np.array(self.nodes))  # Rebuild KDTree incrementally
        
    def nearest_neighbor(self, point: np.ndarray) -> Tuple[int, np.ndarray]:
        """Find nearest node in tree"""
        if self.kdtree is None:
            raise ValueError("KDTree is not initialized.")
        distances, idx = self.kdtree.query(point)
        return idx, self.nodes[idx]
        
    def new_state(self, from_state: np.ndarray, to_state: np.ndarray) -> np.ndarray:
        """Get new state stepping from current state towards target"""
        direction = to_state - from_state
        distance = np.linalg.norm(direction)
        
        if distance < self.max_step:
            return to_state
            
        return from_state + self.max_step * direction / distance
        
    def is_valid_edge(self, from_state: np.ndarray, to_state: np.ndarray) -> bool:
        """Check if edge is collision free"""
        return not self.obstacles.check_line(from_state, to_state)
        
    def sample_state(self) -> np.ndarray:
        """Sample random state"""
        return np.random.uniform(self.bounds[0], self.bounds[1])
        
    def plan(self, 
            start: np.ndarray, 
            goal: np.ndarray, 
            max_iterations: int = 10000,
            goal_bias: float = 0.1) -> Optional[List[np.ndarray]]:
        """
        Plan path using RRT
        
        Args:
            start: Start state
            goal: Goal state
            max_iterations: Maximum number of iterations
            goal_bias: Probability of sampling goal state
            
        Returns:
            List of states forming path, or None if no path found
        """
        # Initialize tree with start node
        self.nodes = []
        self.parents = []
        self.kdtree = None
        self.add_node(start)
        
        for _ in range(max_iterations):
            # Sample random state
            if np.random.random() < goal_bias:
                random_state = goal
            else:
                random_state = self.sample_state()
                
            # Find nearest node
            nearest_idx, nearest_node = self.nearest_neighbor(random_state)
            
            # Extend tree
            new_state = self.new_state(nearest_node, random_state)
            if self.is_valid_edge(nearest_node, new_state):
                self.add_node(new_state, nearest_idx)
                
                # Check if goal reached
                if np.linalg.norm(new_state - goal) < self.max_step:
                    if self.is_valid_edge(new_state, goal):
                        self.add_node(goal, len(self.nodes) - 1)
                        return self.extract_path(len(self.nodes) - 1)
                        
        return None
        
    def extract_path(self, goal_idx: int) -> List[np.ndarray]:
        """Extract path from tree"""
        path = []
        current_idx = goal_idx
        
        while current_idx != -1:
            path.append(self.nodes[current_idx])
            current_idx = self.parents[current_idx]
            
        return path[::-1]
        
    def smooth_path(self, path: List[np.ndarray], max_iterations: int = 100) -> List[np.ndarray]:
        """
        Smooth path using shortcut method
        
        Args:
            path: List of states forming path
            max_iterations: Maximum smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
            
        path = copy.deepcopy(path)
        
        for _ in range(max_iterations):
            # Select two random indices
            i = np.random.randint(0, len(path) - 1)
            j = np.random.randint(i + 1, len(path))
            
            # Check if shortcut possible
            if self.is_valid_edge(path[i], path[j]):
                path[i + 1:j] = []
                
        return path


def visualize_path(path, obstacles):
    lines = [[i, i+1] for i in range(len(path)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path),
        lines=o3d.utility.Vector2iVector(lines)
    )
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(obstacles))
    o3d.visualization.draw_geometries([pcd, line_set])

if __name__ == "__main__":
    pass