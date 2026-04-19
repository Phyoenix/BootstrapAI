"""
3D Gaussian Splatting - NumPy CPU Implementation
Phase 1: Basic 3D Gaussian representation and projection
No CUDA/PyTorch dependency - pure NumPy for CPU execution

Based on: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Gaussian3D:
    """
    A single 3D Gaussian primitive.
    
    Attributes:
        position: (3,) center position in world space [x, y, z]
        covariance: (3, 3) covariance matrix defining shape
        color: (3,) RGB color in [0, 1]
        opacity: float opacity value α ∈ [0, 1]
    """
    position: np.ndarray
    covariance: np.ndarray
    color: np.ndarray
    opacity: float
    
    def __post_init__(self):
        # Validate shapes
        assert self.position.shape == (3,), f"Position must be (3,), got {self.position.shape}"
        assert self.covariance.shape == (3, 3), f"Covariance must be (3,3), got {self.covariance.shape}"
        assert self.color.shape == (3,), f"Color must be (3,), got {self.color.shape}"
        assert 0 <= self.opacity <= 1, f"Opacity must be in [0,1], got {self.opacity}"
    
    @classmethod
    def from_rotation_scale(cls, position, rotation, scale, color, opacity):
        """
        Create Gaussian from rotation (quaternion or rotation matrix) and scale.
        
        Covariance = R * S * S^T * R^T
        where R is rotation matrix, S is diagonal scale matrix
        """
        if rotation.shape == (4,):  # Quaternion
            R = Rotation.from_quat(rotation).as_matrix()
        else:
            R = rotation
        
        S = np.diag(scale ** 2)
        covariance = R @ S @ R.T
        
        return cls(position, covariance, color, opacity)


class GaussianScene:
    """Collection of 3D Gaussians representing a scene."""
    
    def __init__(self):
        self.gaussians: List[Gaussian3D] = []
        self.num_gaussians = 0
        
    def add_gaussian(self, gaussian: Gaussian3D):
        """Add a single Gaussian to the scene."""
        self.gaussians.append(gaussian)
        self.num_gaussians += 1
        
    def add_random_gaussians(self, n: int, bounds: Tuple[float, float] = (-1, 1)):
        """Add n random Gaussians for testing."""
        for _ in range(n):
            # Random position
            pos = np.random.uniform(bounds[0], bounds[1], 3)
            
            # Random rotation
            rot = Rotation.random()
            
            # Random scale (small for dense sampling)
            scale = np.random.uniform(0.01, 0.1, 3)
            
            # Create covariance
            S = np.diag(scale ** 2)
            cov = rot.as_matrix() @ S @ rot.as_matrix().T
            
            # Random color and opacity
            color = np.random.uniform(0, 1, 3)
            opacity = np.random.uniform(0.1, 1.0)
            
            self.add_gaussian(Gaussian3D(pos, cov, color, opacity))
    
    def get_arrays(self) -> dict:
        """Convert scene to numpy arrays for batch processing."""
        if not self.gaussians:
            return {}
        
        positions = np.stack([g.position for g in self.gaussians])
        covariances = np.stack([g.covariance for g in self.gaussians])
        colors = np.stack([g.color for g in self.gaussians])
        opacities = np.array([g.opacity for g in self.gaussians])
        
        return {
            'positions': positions,
            'covariances': covariances,
            'colors': colors,
            'opacities': opacities,
            'num_gaussians': len(self.gaussians)
        }


class Camera:
    """
    Pinhole camera model for projecting 3D Gaussians to 2D.
    """
    
    def __init__(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov_x: float = 60.0,
        width: int = 800,
        height: int = 600,
        near: float = 0.1,
        far: float = 100.0
    ):
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov_x = fov_x
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        
        # Compute intrinsic and extrinsic matrices
        self.extrinsic = self._compute_extrinsic()
        self.intrinsic = self._compute_intrinsic()
        
    def _compute_extrinsic(self) -> np.ndarray:
        """Compute camera extrinsic matrix (world to camera)."""
        # Camera coordinate system
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Rotation matrix (camera to world)
        R = np.stack([right, up, -forward], axis=0)
        
        # Translation
        t = -R @ self.position
        
        # 4x4 extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        
        return extrinsic
    
    def _compute_intrinsic(self) -> np.ndarray:
        """Compute camera intrinsic matrix."""
        fov_x_rad = np.deg2rad(self.fov_x)
        f_x = self.width / (2 * np.tan(fov_x_rad / 2))
        f_y = f_x  # Assume square pixels
        
        c_x = self.width / 2
        c_y = self.height / 2
        
        intrinsic = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsic


def project_covariance_to_2d(
    covariance_3d: np.ndarray,
    position: np.ndarray,
    camera: Camera
) -> np.ndarray:
    """
    Project 3D covariance to 2D image space.
    
    Implements the key formula from the paper:
    Σ' = JW Σ W^T J^T
    
    where:
    - J is the Jacobian of the projection at the Gaussian center
    - W is the view transformation matrix
    - Σ is the 3D covariance
    
    Args:
        covariance_3d: (3, 3) world space covariance
        position: (3,) world space position
        camera: Camera object
        
    Returns:
        covariance_2d: (2, 2) screen space covariance
    """
    # Transform to camera space
    pos_h = np.append(position, 1.0)
    pos_cam = camera.extrinsic @ pos_h
    pos_cam = pos_cam[:3]
    
    # Check if behind camera
    if pos_cam[2] < camera.near:
        return np.eye(2) * 1e-6  # Very small covariance
    
    # Compute Jacobian of perspective projection
    # For pinhole camera: x' = fx * X/Z + cx, y' = fy * Y/Z + cy
    # Jacobian J = [[fx/Z, 0, -fx*X/Z^2],
    #               [0, fy/Z, -fy*Y/Z^2]]
    
    fx, fy = camera.intrinsic[0, 0], camera.intrinsic[1, 1]
    X, Y, Z = pos_cam
    Z2 = Z * Z
    
    J = np.array([
        [fx / Z, 0, -fx * X / Z2],
        [0, fy / Z, -fy * Y / Z2]
    ])
    
    # Get rotation part of extrinsic
    W = camera.extrinsic[:3, :3]
    
    # Project covariance: Σ' = JW Σ W^T J^T
    T = J @ W
    cov_2d = T @ covariance_3d @ T.T
    
    # Ensure positive definite
    cov_2d = (cov_2d + cov_2d.T) / 2  # Symmetrize
    eigvals = np.linalg.eigvals(cov_2d)
    if np.any(eigvals <= 0):
        cov_2d += np.eye(2) * (abs(eigvals.min()) + 1e-6)
    
    return cov_2d


def gaussian_2d_value(
    xy: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray
) -> float:
    """
    Evaluate 2D Gaussian at point xy.
    
    G(x) = exp(-0.5 * (x-μ)^T Σ^(-1) (x-μ))
    """
    diff = xy - mean
    try:
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * diff.T @ inv_cov @ diff
        return np.exp(exponent)
    except np.linalg.LinAlgError:
        return 0.0


def render_gaussians(
    gaussians: dict,
    camera: Camera,
    tile_size: int = 16
) -> np.ndarray:
    """
    Render Gaussians using tile-based rasterization.
    
    Simplified implementation of Algorithm 1 from the paper.
    
    Args:
        gaussians: dict with 'positions', 'covariances', 'colors', 'opacities'
        camera: Camera configuration
        tile_size: Size of tiles for rasterization
        
    Returns:
        image: (H, W, 3) rendered image
    """
    H, W = camera.height, camera.width
    image = np.zeros((H, W, 3), dtype=np.float32)
    
    N = gaussians['num_gaussians']
    if N == 0:
        return image
    
    # Step 1: Project all Gaussians to 2D
    means_2d = []
    covs_2d = []
    depths = []
    
    for i in range(N):
        pos = gaussians['positions'][i]
        cov_3d = gaussians['covariances'][i]
        
        # Project position
        pos_h = np.append(pos, 1.0)
        pos_cam = camera.extrinsic @ pos_h
        depth = pos_cam[2]
        
        if depth < camera.near or depth > camera.far:
            continue
        
        # Project to screen
        pos_screen = camera.intrinsic @ (pos_cam[:3] / pos_cam[2])
        mean_2d = pos_screen[:2]
        
        # Project covariance
        cov_2d = project_covariance_to_2d(cov_3d, pos, camera)
        
        # Check if on screen (with margin)
        margin = 3 * np.sqrt(max(cov_2d[0, 0], cov_2d[1, 1]))
        if (mean_2d[0] < -margin or mean_2d[0] > W + margin or
            mean_2d[1] < -margin or mean_2d[1] > H + margin):
            continue
        
        means_2d.append(mean_2d)
        covs_2d.append(cov_2d)
        depths.append(depth)
    
    if not means_2d:
        return image
    
    means_2d = np.array(means_2d)
    covs_2d = np.array(covs_2d)
    depths = np.array(depths)
    
    # Step 2: Sort by depth (back to front for α-blending)
    sorted_indices = np.argsort(-depths)  # Descending (back to front)
    
    # Step 3: Rasterize (simplified - per-pixel evaluation)
    # Create pixel grid
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    pixels = np.stack([x_coords, y_coords], axis=-1).astype(np.float32)
    
    # Accumulate transmittance
    T = np.ones((H, W), dtype=np.float32)  # Transmittance
    
    for idx in sorted_indices:
        mean = means_2d[idx]
        cov = covs_2d[idx]
        color = gaussians['colors'][idx]
        alpha = gaussians['opacities'][idx]
        
        # Compute Gaussian values for all pixels
        # Optimization: compute bounding box first
        sigma_x = np.sqrt(cov[0, 0])
        sigma_y = np.sqrt(cov[1, 1])
        
        x_min = max(0, int(mean[0] - 3 * sigma_x))
        x_max = min(W, int(mean[0] + 3 * sigma_x + 1))
        y_min = max(0, int(mean[1] - 3 * sigma_y))
        y_max = min(H, int(mean[1] + 3 * sigma_y + 1))
        
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Evaluate Gaussian in bounding box
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                xy = np.array([x, y], dtype=np.float32)
                G = gaussian_2d_value(xy, mean, cov)
                
                # α-blending: C = Σ ci αi Gi Ti
                alpha_i = alpha * G
                weight = alpha_i * T[y, x]
                
                image[y, x] += weight * color
                T[y, x] *= (1 - alpha_i)
    
    return image


def visualize_scene_2d(
    gaussians: dict,
    camera: Camera,
    save_path: Optional[str] = None
):
    """Visualize projected Gaussians in 2D (debugging tool)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    N = gaussians['num_gaussians']
    
    for i in range(min(N, 50)):  # Limit to 50 for visibility
        pos = gaussians['positions'][i]
        cov_3d = gaussians['covariances'][i]
        color = gaussians['colors'][i]
        
        # Project to 2D
        pos_h = np.append(pos, 1.0)
        pos_cam = camera.extrinsic @ pos_h
        
        if pos_cam[2] < camera.near:
            continue
        
        pos_screen = camera.intrinsic @ (pos_cam[:3] / pos_cam[2])
        mean_2d = pos_screen[:2]
        
        cov_2d = project_covariance_to_2d(cov_3d, pos, camera)
        
        # Get ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * 2 * np.sqrt(eigvals[0])  # 2 sigma
        height = 2 * 2 * np.sqrt(eigvals[1])
        
        # Draw ellipse
        ellipse = Ellipse(
            xy=mean_2d,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor='black',
            alpha=gaussians['opacities'][i] * 0.5,
            linewidth=1
        )
        ax.add_patch(ellipse)
        ax.plot(mean_2d[0], mean_2d[1], 'k.', markersize=2)
    
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)  # Flip Y
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Projected Gaussians ({camera.width}x{camera.height})')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    """Test implementation."""
    print("=" * 60)
    print("3D Gaussian Splatting - NumPy CPU Implementation")
    print("Phase 1: Basic Projection and Rendering Test")
    print("=" * 60)
    
    # Create scene
    print("\n[1] Creating scene...")
    scene = GaussianScene()
    scene.add_random_gaussians(200)
    print(f"  Created {scene.num_gaussians} random Gaussians")
    
    # Get arrays
    gaussian_data = scene.get_arrays()
    print(f"  Positions shape: {gaussian_data['positions'].shape}")
    print(f"  Covariances shape: {gaussian_data['covariances'].shape}")
    
    # Create camera
    print("\n[2] Setting up camera...")
    camera = Camera(
        position=[2.0, 2.0, 2.0],
        look_at=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
        fov_x=60.0,
        width=640,
        height=480
    )
    print(f"  Camera: {camera.width}x{camera.height}")
    print(f"  Position: {camera.position}")
    
    # Test projection
    print("\n[3] Testing covariance projection...")
    test_cov = project_covariance_to_2d(
        gaussian_data['covariances'][0],
        gaussian_data['positions'][0],
        camera
    )
    print(f"  3D cov shape: {gaussian_data['covariances'][0].shape}")
    print(f"  2D cov shape: {test_cov.shape}")
    print(f"  2D cov:\n{test_cov}")
    
    # Visualize 2D projection
    print("\n[4] Creating 2D projection visualization...")
    fig, ax = visualize_scene_2d(gaussian_data, camera, 
                                    save_path="/tmp/gaussian_2d_projection.png")
    print("  Saved to /tmp/gaussian_2d_projection.png")
    
    # Render
    print("\n[5] Rendering image (this may take a while)...")
    import time
    start = time.time()
    image = render_gaussians(gaussian_data, camera)
    elapsed = time.time() - start
    print(f"  Rendered in {elapsed:.2f}s")
    print(f"  Image shape: {image.shape}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Save rendered image
    plt.imsave("/tmp/gaussian_render.png", np.clip(image, 0, 1))
    print("  Saved to /tmp/gaussian_render.png")
    
    print("\n" + "=" * 60)
    print("✓ Phase 1 tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Implement spherical harmonics for view-dependent color")
    print("  - Add adaptive density control (clone/split)")
    print("  - Optimize rendering with proper tile-based rasterization")
    print("  - Train on real dataset (Synthetic NeRF or Mip-NeRF 360)")
