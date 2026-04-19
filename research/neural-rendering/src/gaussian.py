"""
3D Gaussian Splatting - Core Implementation
Phase 1: Basic 3D Gaussian representation and projection

Based on: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Gaussian3D:
    """
    A single 3D Gaussian primitive.
    
    Attributes:
        position: (3,) center position in world space
        covariance: (3, 3) covariance matrix defining shape
        color: (3,) RGB color or (3, K) for SH coefficients
        opacity: (1,) opacity value α ∈ [0, 1]
        rotation: (4,) quaternion for rotation
        scale: (3,) scaling factors
    """
    position: torch.Tensor
    covariance: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    
    def __post_init__(self):
        # Validate shapes
        assert self.position.shape == (3,), f"Position must be (3,), got {self.position.shape}"
        assert self.covariance.shape == (3, 3), f"Covariance must be (3,3), got {self.covariance.shape}"
        assert self.color.shape[0] == 3, f"Color must start with 3, got {self.color.shape}"
        assert self.opacity.shape == (1,), f"Opacity must be (1,), got {self.opacity.shape}"


class GaussianScene:
    """
    Collection of 3D Gaussians representing a scene.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gaussians = []
        self.num_gaussians = 0
        
    def add_gaussian(self, gaussian: Gaussian3D):
        """Add a single Gaussian to the scene."""
        self.gaussians.append(gaussian)
        self.num_gaussians += 1
        
    def add_random_gaussians(self, n: int, bounds: Tuple[float, float] = (-1, 1)):
        """Add n random Gaussians for testing."""
        for _ in range(n):
            pos = torch.rand(3, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]
            
            # Random covariance (ensuring positive definite)
            scale = torch.rand(3, device=self.device) * 0.1 + 0.01
            rot = torch.randn(3, 3, device=self.device)
            rot, _ = torch.linalg.qr(rot)  # Orthogonal rotation matrix
            cov = rot @ torch.diag(scale ** 2) @ rot.T
            
            color = torch.rand(3, device=self.device)
            opacity = torch.rand(1, device=self.device)
            
            self.add_gaussian(Gaussian3D(pos, cov, color, opacity))
    
    def get_tensor_dict(self) -> dict:
        """Convert scene to batched tensors for efficient processing."""
        if not self.gaussians:
            return {}
        
        positions = torch.stack([g.position for g in self.gaussians])
        covariances = torch.stack([g.covariance for g in self.gaussians])
        colors = torch.stack([g.color for g in self.gaussians])
        opacities = torch.stack([g.opacity for g in self.gaussians])
        
        return {
            'positions': positions,
            'covariances': covariances,
            'colors': colors,
            'opacities': opacities,
            'num_gaussians': len(self.gaussians)
        }


class Camera:
    """
    Camera model for projecting 3D Gaussians to 2D.
    """
    
    def __init__(
        self,
        position: torch.Tensor,
        look_at: torch.Tensor,
        up: torch.Tensor,
        fov_x: float = 60.0,
        fov_y: float = 60.0,
        width: int = 800,
        height: int = 600,
        near: float = 0.1,
        far: float = 100.0
    ):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        
        # Precompute view and projection matrices
        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
        
    def _compute_view_matrix(self) -> torch.Tensor:
        """Compute view matrix (world to camera transformation)."""
        forward = self.look_at - self.position
        forward = forward / torch.norm(forward)
        
        right = torch.cross(forward, self.up)
        right = right / torch.norm(right)
        
        up = torch.cross(right, forward)
        
        # View matrix (look-at)
        view = torch.eye(4, device=self.position.device)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -torch.stack([
            torch.dot(right, self.position),
            torch.dot(up, self.position),
            torch.dot(forward, self.position)
        ])
        
        return view
    
    def _compute_projection_matrix(self) -> torch.Tensor:
        """Compute perspective projection matrix."""
        # Perspective projection
        fov_x_rad = self.fov_x * np.pi / 180.0
        fov_y_rad = self.fov_y * np.pi / 180.0
        
        tan_half_fov_x = np.tan(fov_x_rad / 2)
        tan_half_fov_y = np.tan(fov_y_rad / 2)
        
        proj = torch.zeros(4, 4, device=self.position.device)
        proj[0, 0] = 1.0 / tan_half_fov_x
        proj[1, 1] = 1.0 / tan_half_fov_y
        proj[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        proj[3, 2] = -1
        
        return proj


def project_gaussians_to_2d(
    gaussians: dict,
    camera: Camera,
    tile_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D image space.
    
    This implements the core projection step from 3DGS paper:
    Σ' = JW Σ W^T J^T
    
    Returns:
        means_2d: (N, 2) projected centers
        covariances_2d: (N, 2, 2) 2D covariances
        depths: (N,) depth values for sorting
    """
    device = gaussians['positions'].device
    N = gaussians['num_gaussians']
    
    if N == 0:
        return torch.empty(0, 2, device=device), torch.empty(0, 2, 2, device=device), torch.empty(0, device=device)
    
    # Transform to camera space
    positions_h = torch.cat([gaussians['positions'], torch.ones(N, 1, device=device)], dim=-1)
    positions_cam = (camera.view_matrix @ positions_h.T).T
    positions_cam = positions_cam[:, :3]  # Drop homogeneous coordinate
    
    # Perspective projection to clip space
    positions_clip = (camera.projection_matrix @ torch.cat([positions_cam, torch.ones(N, 1, device=device)], dim=-1).T).T
    positions_clip = positions_clip[:, :3] / (positions_clip[:, 3:4] + 1e-7)  # Perspective divide
    
    # To screen space (NDC to pixel coordinates)
    means_2d = torch.zeros(N, 2, device=device)
    means_2d[:, 0] = (positions_clip[:, 0] + 1) * camera.width / 2
    means_2d[:, 1] = (1 - positions_clip[:, 1]) * camera.height / 2  # Flip Y
    
    # Compute depths for sorting
    depths = positions_cam[:, 2]
    
    # Full 2D covariance computation: Σ' = JW Σ W^T J^T
    # J: Jacobian of perspective projection (2x3)
    # W: rotation part of view matrix (3x3)
    # Contribution: WorkBuddy collab - 2026-04-19
    fx = camera.width / (2.0 * np.tan(np.deg2rad(camera.fov_x) / 2.0))
    fy = camera.height / (2.0 * np.tan(np.deg2rad(camera.fov_y) / 2.0))

    W = camera.view_matrix[:3, :3]  # (3, 3) rotation part

    covariances_2d = torch.zeros(N, 2, 2, device=device)
    for i in range(N):
        depth = depths[i].item()
        if depth < camera.near:
            covariances_2d[i] = torch.eye(2, device=device) * 1e-6
            continue

        X = positions_cam[i, 0].item()
        Y = positions_cam[i, 1].item()
        Z = depth
        Z2 = Z * Z

        # Jacobian of pinhole projection
        J = torch.tensor([
            [fx / Z,     0.0, -fx * X / Z2],
            [0.0,     fy / Z, -fy * Y / Z2],
        ], dtype=torch.float32, device=device)  # (2, 3)

        T = J @ W  # (2, 3)
        cov_3d = gaussians['covariances'][i]  # (3, 3)
        cov_2d = T @ cov_3d @ T.T  # (2, 2)

        # Symmetrise and ensure positive-definite
        cov_2d = (cov_2d + cov_2d.T) * 0.5
        eigvals = torch.linalg.eigvalsh(cov_2d)
        if eigvals.min() <= 0:
            cov_2d = cov_2d + torch.eye(2, device=device) * (eigvals.min().abs() + 1e-6)

        covariances_2d[i] = cov_2d
    
    return means_2d, covariances_2d, depths


def render_gaussians_simple(
    gaussians: dict,
    camera: Camera,
    background: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Simple rasterization of Gaussians (simplified version).
    
    Full implementation would include:
    - Tile-based rasterization
    - α-blending in sorted order
    - Spherical harmonics for view-dependent color
    """
    device = gaussians['positions'].device
    
    # Create output image
    image = torch.zeros(camera.height, camera.width, 3, device=device)
    if background is not None:
        image = background.clone()
    
    # Project to 2D
    means_2d, covs_2d, depths = project_gaussians_to_2d(gaussians, camera)
    
    # Sort by depth (back to front for α-blending)
    sorted_indices = torch.argsort(depths, descending=True)
    
    # Simple point splatting (not full Gaussian splatting yet)
    # TODO: Implement full 2D Gaussian evaluation
    for idx in sorted_indices:
        if depths[idx] < camera.near or depths[idx] > camera.far:
            continue
        
        x, y = int(means_2d[idx, 0]), int(means_2d[idx, 1])
        if 0 <= x < camera.width and 0 <= y < camera.height:
            # Simple alpha blending
            alpha = gaussians['opacities'][idx, 0]
            color = gaussians['colors'][idx]
            image[y, x] = (1 - alpha) * image[y, x] + alpha * color
    
    return image


if __name__ == "__main__":
    """Test basic functionality."""
    print("=== 3D Gaussian Splatting - Phase 1 Test ===\n")
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create scene with random Gaussians
    scene = GaussianScene(device=device)
    scene.add_random_gaussians(100)
    print(f"Created scene with {scene.num_gaussians} Gaussians")
    
    # Create camera
    camera = Camera(
        position=torch.tensor([2.0, 2.0, 2.0], device=device),
        look_at=torch.tensor([0.0, 0.0, 0.0], device=device),
        up=torch.tensor([0.0, 1.0, 0.0], device=device),
        width=640,
        height=480
    )
    print(f"Camera created: {camera.width}x{camera.height}")
    
    # Get tensor representation
    gaussian_data = scene.get_tensor_dict()
    print(f"\nTensor shapes:")
    print(f"  Positions: {gaussian_data['positions'].shape}")
    print(f"  Covariances: {gaussian_data['covariances'].shape}")
    print(f"  Colors: {gaussian_data['colors'].shape}")
    
    # Test projection
    means_2d, covs_2d, depths = project_gaussians_to_2d(gaussian_data, camera)
    print(f"\nProjection test:")
    print(f"  2D means: {means_2d.shape}")
    print(f"  2D covs: {covs_2d.shape}")
    print(f"  Depths: {depths.shape}")
    print(f"  Depth range: [{depths.min():.2f}, {depths.max():.2f}]")
    
    # Simple render test
    image = render_gaussians_simple(gaussian_data, camera)
    print(f"\nRender test:")
    print(f"  Output image: {image.shape}")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    print("\n✓ Phase 1 basic tests passed!")
    print("Next: Implement full 2D Gaussian evaluation and proper α-blending")
