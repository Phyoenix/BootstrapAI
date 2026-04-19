"""
Training Framework for 3D Gaussian Splatting
Integrates WorkBuddy's Spherical Harmonics implementation

Author: Kraber (building on WorkBuddy's SH foundation)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from pathlib import Path

from gaussian_numpy import GaussianScene, Camera, project_gaussians_to_2d, render_gaussians
from spherical_harmonics import ViewDependentColor, sh_num_coeffs
from dataset import MockDataset, NeRFDataset


class GaussianModel:
    """
    Learnable 3D Gaussian representation.
    
    Combines:
    - 3D positions (learnable)
    - Covariances (learnable via rotation + scale)
    - Opacities (learnable)
    - View-dependent colors via SH (WorkBuddy's implementation)
    """
    
    def __init__(self, num_points: int, sh_degree: int = 3):
        self.num_points = num_points
        self.sh_degree = sh_degree
        
        # Initialize from random point cloud
        # In practice, this would come from COLMAP or random initialization
        
        # Positions: (N, 3)
        self._xyz = torch.randn(num_points, 3) * 0.5
        
        # Rotation (as quaternions): (N, 4)
        self._rotation = torch.zeros(num_points, 4)
        self._rotation[:, 0] = 1.0  # Identity rotation
        
        # Scale: (N, 3) - log space for stability
        self._scaling = torch.log(torch.ones(num_points, 3) * 0.01)
        
        # Opacity: (N, 1) - logit space
        self._opacity = torch.logit(torch.ones(num_points, 1) * 0.5)
        
        # View-dependent color via SH (using WorkBuddy's module)
        self.sh_color = ViewDependentColor(num_points, sh_degree)
        
        # Make learnable
        self._xyz = nn.Parameter(self._xyz)
        self._rotation = nn.Parameter(self._rotation)
        self._scaling = nn.Parameter(self._scaling)
        self._opacity = nn.Parameter(self._opacity)
    
    def get_covariance(self, idx: int = None) -> np.ndarray:
        """
        Compute covariance matrix from rotation and scale.
        
        Covariance = R @ S @ S^T @ R^T
        """
        if idx is None:
            # Return all
            covs = []
            for i in range(self.num_points):
                covs.append(self.get_covariance(i))
            return np.array(covs)
        
        # Single covariance
        # Convert quaternion to rotation matrix (simplified - proper implementation needed)
        q = self._rotation[idx].detach().numpy()
        # For now, use identity as placeholder
        R = np.eye(3)
        
        # Scale from log space
        s = np.exp(self._scaling[idx].detach().numpy())
        S = np.diag(s ** 2)
        
        cov = R @ S @ R.T
        return cov
    
    def get_opacity(self) -> np.ndarray:
        """Get opacity values in [0, 1]."""
        return torch.sigmoid(self._opacity).detach().numpy()
    
    def get_positions(self) -> np.ndarray:
        """Get positions."""
        return self._xyz.detach().numpy()
    
    def to_numpy_scene(self, camera_position: np.ndarray) -> Dict:
        """Convert to numpy format for rendering."""
        positions = self.get_positions()
        
        # Get view-dependent colors for this camera
        cam_pos_torch = torch.tensor(camera_position, dtype=torch.float32)
        with torch.no_grad():
            colors = model.sh_color(self._xyz, cam_pos_torch).numpy()
        
        opacities = self.get_opacity()
        covariances = self.get_covariance()
        
        return {
            'positions': positions,
            'covariances': covariances,
            'colors': colors,
            'opacities': opacities[:, 0],
            'num_gaussians': self.num_points
        }
    
    def parameters(self) -> List[nn.Parameter]:
        """Get all learnable parameters."""
        return [
            self._xyz,
            self._rotation,
            self._scaling,
            self._opacity,
            self.sh_color.sh_coeffs
        ]


class SimpleTrainer:
    """
    Simplified training loop for 3D Gaussian Splatting.
    
    Loss: L1 + λ * SSIM
    """
    
    def __init__(self, model: GaussianModel, lr: float = 0.01):
        self.model = model
        self.lr = lr
        
        # Simple optimizer (Adam would be better)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        self.loss_history = []
    
    def compute_l1_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """L1 loss between rendered and target image."""
        return np.abs(pred - target).mean()
    
    def train_step(self, camera: Camera, target_image: np.ndarray) -> float:
        """
        Single training step.
        
        Note: This is a simplified version. Full implementation would:
        1. Render using differentiable rasterization
        2. Compute loss with gradients
        3. Backpropagate
        4. Update parameters
        
        Currently using numpy renderer (non-differentiable) as placeholder.
        """
        # Render current scene
        scene = self.model.to_numpy_scene(camera.position.numpy() if hasattr(camera.position, 'numpy') else np.array(camera.position))
        
        rendered = render_gaussians(scene, camera)
        
        # Compute loss (numpy version)
        loss = self.compute_l1_loss(rendered, target_image)
        self.loss_history.append(loss)
        
        # TODO: Implement differentiable rendering for actual training
        # For now, just report loss
        
        return loss
    
    def train(self, dataset, num_iterations: int = 100):
        """Training loop."""
        print(f"Training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Sample random camera
            idx = np.random.randint(0, len(dataset))
            camera, target_image = dataset[idx]
            
            # Convert Camera to numpy format
            # (dataset.py CameraInfo to gaussian_numpy Camera)
            from gaussian_numpy import Camera as GCamera
            cam_np = GCamera(
                position=-camera.R.T @ camera.T,
                look_at=np.array([0, 0, 0]),
                up=np.array([0, 1, 0]),
                width=camera.width,
                height=camera.height
            )
            
            loss = self.train_step(cam_np, target_image)
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration}: Loss = {loss:.4f}")
        
        print(f"Training complete. Final loss: {loss:.4f}")


def test_training():
    """Test training framework."""
    print("=" * 60)
    print("Training Framework Test (with WorkBuddy's SH)")
    print("=" * 60)
    
    # Create mock dataset
    print("\n[1] Creating mock dataset...")
    dataset = MockDataset(num_cameras=8, radius=2.0)
    
    # Create model
    print("\n[2] Creating Gaussian model...")
    model = GaussianModel(num_points=100, sh_degree=1)
    print(f"  Model has {model.num_points} Gaussians")
    print(f"  SH degree: {model.sh_degree}")
    
    # Test view-dependent color (WorkBuddy's SH)
    print("\n[3] Testing view-dependent color (WorkBuddy's SH module)...")
    camera_pos = torch.tensor([2.0, 2.0, 2.0])
    colors = model.sh_color(model._xyz, camera_pos)
    print(f"  Colors shape: {colors.shape}")
    print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    
    # Create trainer
    print("\n[4] Creating trainer...")
    trainer = SimpleTrainer(model, lr=0.01)
    
    # Run a few training steps
    print("\n[5] Running test training (5 iterations)...")
    for i in range(5):
        idx = i % len(dataset)
        camera_info, target = dataset[idx]
        
        from gaussian_numpy import Camera as GCamera
        cam = GCamera(
            position=-camera_info.R.T @ camera_info.T,
            look_at=np.array([0, 0, 0]),
            up=np.array([0, 1, 0]),
            width=camera_info.width,
            height=camera_info.height
        )
        
        loss = trainer.train_step(cam, target)
        print(f"  Iter {i}: Loss = {loss:.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] Training framework test passed!")
    print("=" * 60)
    print("\nNotes:")
    print("  - Using WorkBuddy's SphericalHarmonics for view-dependent color")
    print("  - Differentiable rendering needed for real training")
    print("  - Next: Implement proper backprop through rasterization")


if __name__ == "__main__":
    test_training()
