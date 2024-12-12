import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List
import copy

class PointCloudProcessor:
    """Point cloud preprocessing pipeline"""
    
    def __init__(self, points: Optional[np.ndarray] = None):
        """
        Initialize processor with optional point cloud data
        
        Args:
            points: Nx3 array of point coordinates
        """
        if points is not None:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
        else:
            self.pcd = None
            
    def load_from_file(self, filename: str):
        """Load point cloud from file"""
        self.pcd = o3d.io.read_point_cloud(filename)
        return self
        
    def save_to_file(self, filename: str):
        """Save point cloud to file"""
        o3d.io.write_point_cloud(filename, self.pcd)
        
    def get_points(self) -> np.ndarray:
        """Get points as numpy array"""
        return np.asarray(self.pcd.points)
        
    def voxel_downsample(self, voxel_size: float = 0.05):
        """
        Voxel downsampling to reduce point cloud density
        
        Args:
            voxel_size: Size of voxel grid for downsampling
        """
        self.pcd = self.pcd.voxel_down_sample(voxel_size)
        return self
        
    def statistical_outlier_removal(self, 
                                  nb_neighbors: int = 20,
                                  std_ratio: float = 2.0):
        """
        Remove outliers using statistical analysis
        
        Args:
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation ratio threshold
        """
        self.pcd, _ = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return self
        
    def radius_outlier_removal(self,
                             nb_points: int = 16,
                             radius: float = 0.05):
        """
        Remove outliers with too few neighbors
        
        Args:
            nb_points: Minimum number of points within radius
            radius: Radius to search for neighbors
        """
        self.pcd, _ = self.pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
        return self
        
    def estimate_normals(self,
                        radius: Optional[float] = None,
                        max_nn: int = 30):
        """
        Estimate point normals
        
        Args:
            radius: Radius to search for neighbors (auto if None)
            max_nn: Maximum number of neighbors to use
        """
        if radius is None:
            # Estimate radius from point cloud density
            bbox = self.pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            num_points = len(self.pcd.points)
            radius = np.mean(bbox_size) * np.power(num_points, -1/3) * 3
            
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )
        return self
        
    def crop_box(self,
                min_bound: np.ndarray,
                max_bound: np.ndarray):
        """
        Crop point cloud to bounding box
        
        Args:
            min_bound: Minimum corner of box
            max_bound: Maximum corner of box
        """
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
        self.pcd = self.pcd.crop(bbox)
        return self
        
    def remove_ground_plane(self,
                          height_threshold: float = 0.1,
                          num_iterations: int = 100):
        """
        Remove ground plane points using RANSAC
        
        Args:
            height_threshold: Maximum deviation from plane
            num_iterations: Number of RANSAC iterations
        """
        plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=height_threshold,
            ransac_n=3,
            num_iterations=num_iterations
        )
        outlier_cloud = self.pcd.select_by_index(inliers, invert=True)
        self.pcd = outlier_cloud
        return self
        
    def cluster_dbscan(self,
                      eps: float = 0.05,
                      min_points: int = 10) -> List[np.ndarray]:
        """
        Cluster points using DBSCAN
        
        Args:
            eps: Maximum distance between points in cluster
            min_points: Minimum points for core cluster
            
        Returns:
            List of point arrays for each cluster
        """
        labels = np.array(self.pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points
        ))
        
        max_label = labels.max()
        clusters = []
        points = np.asarray(self.pcd.points)
        
        for i in range(max_label + 1):
            cluster_points = points[labels == i]
            clusters.append(cluster_points)
            
        return clusters
        
    def estimate_noise(self,
                      num_neighbors: int = 30) -> np.ndarray:
        """
        Estimate local noise level at each point
        
        Args:
            num_neighbors: Number of neighbors for local analysis
            
        Returns:
            Array of estimated noise levels
        """
        # Build KD tree
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        points = np.asarray(self.pcd.points)
        noise_levels = np.zeros(len(points))
        
        # Compute local noise level for each point
        for i, point in enumerate(points):
            # Find neighbors
            [_, idx, _] = pcd_tree.search_knn_vector_3d(point, num_neighbors)
            neighbors = points[idx]
            
            # Compute local covariance
            cov = np.cov(neighbors.T)
            eigenvals = np.linalg.eigvals(cov)
            
            # Use smallest eigenvalue as noise estimate
            noise_levels[i] = np.sqrt(np.min(eigenvals))
            
        return noise_levels
        
    def fill_holes(self,
                  radius: float = 0.05,
                  num_neighbors: int = 30):
        """
        Fill small holes in point cloud
        
        Args:
            radius: Maximum hole radius to fill
            num_neighbors: Number of neighbors for interpolation
        """
        # Build KD tree
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        points = np.asarray(self.pcd.points)
        
        # Estimate normals if not already computed
        if not self.pcd.has_normals():
            self.estimate_normals()
        normals = np.asarray(self.pcd.normals)
        
        new_points = []
        
        # Detect and fill holes
        for i, point in enumerate(points):
            # Find neighbors
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
            
            if k < num_neighbors:
                # Potential hole - interpolate new points
                neighbors = points[idx]
                neighbor_normals = normals[idx]
                
                # Fit local plane
                plane_center = np.mean(neighbors, axis=0)
                plane_normal = np.mean(neighbor_normals, axis=0)
                
                # Add interpolated points
                num_points = num_neighbors - k
                for _ in range(num_points):
                    # Random point in hole
                    r = np.random.uniform(0, radius)
                    theta = np.random.uniform(0, 2*np.pi)
                    
                    # Project to fitted plane
                    offset = r * np.array([np.cos(theta), np.sin(theta), 0])
                    offset = offset - np.dot(offset, plane_normal) * plane_normal
                    new_point = plane_center + offset
                    
                    new_points.append(new_point)
                    
        # Add interpolated points to cloud
        if new_points:
            new_points = np.array(new_points)
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(new_points)
            self.pcd += new_pcd
            
        return self

def preprocess_point_cloud(filename: str,
                          voxel_size: float = 0.05,
                          remove_outliers: bool = True,
                          remove_ground: bool = True,
                          fill_holes: bool = True) -> np.ndarray:
    """
    Complete point cloud preprocessing pipeline
    
    Args:
        filename: Input point cloud file
        voxel_size: Voxel size for downsampling
        remove_outliers: Whether to remove outliers
        remove_ground: Whether to remove ground plane
        fill_holes: Whether to fill holes
        
    Returns:
        Processed point cloud as numpy array
    """
    processor = PointCloudProcessor()
    
    # Load and basic processing
    processor.load_from_file(filename)
    processor.voxel_downsample(voxel_size)
    
    if remove_outliers:
        processor.statistical_outlier_removal()
        processor.radius_outlier_removal()
        
    # Estimate normals for later steps
    processor.estimate_normals()
    
    if remove_ground:
        processor.remove_ground_plane()
        
    if fill_holes:
        processor.fill_holes()
        
    return processor.get_points()

# Example usage:
if __name__ == "__main__":
    # Load and process point cloud
    processor = PointCloudProcessor()
    processor.load_from_file("input.pcd")
    
    # Basic preprocessing
    processor.voxel_downsample(0.05)  # 5cm voxel size
    processor.statistical_outlier_removal()
    processor.radius_outlier_removal()
    
    # Advanced processing
    processor.estimate_normals()
    processor.remove_ground_plane()
    processor.fill_holes()
    
    # Get clusters
    clusters = processor.cluster_dbscan()
    
    # Save result
    processor.save_to_file("output.pcd")
    
    print(f"Found {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster)} points")