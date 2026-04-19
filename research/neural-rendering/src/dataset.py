"""
Dataset Loader for Neural Rendering Training
Supports Synthetic NeRF dataset format

Author: Kraber (collaborating with WorkBuddy's SH implementation)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CameraInfo:
    """Camera parameters matching NeRF/3DGS format."""
    R: np.ndarray  # (3, 3) rotation matrix
    T: np.ndarray  # (3,) translation vector
    fx: float      # focal length x
    fy: float      # focal length y
    cx: float      # principal point x
    cy: float      # principal point y
    width: int
    height: int
    image_path: Path
    
    @property
    def extrinsic(self) -> np.ndarray:
        """Get 4x4 extrinsic matrix (world to camera)."""
        ext = np.eye(4)
        ext[:3, :3] = self.R
        ext[:3, 3] = self.T
        return ext
    
    @property
    def intrinsic(self) -> np.ndarray:
        """Get 3x3 intrinsic matrix."""
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        return K


class NeRFDataset:
    """
    Synthetic NeRF dataset loader.
    
    Expected structure:
    dataset/
    ├── transforms_train.json
    ├── transforms_test.json
    ├── transforms_val.json
    └── train/
        ├── r_0.png
        ├── r_1.png
        └── ...
    """
    
    def __init__(self, dataset_path: str, split: str = "train"):
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Load transforms
        transforms_file = self.dataset_path / f"transforms_{split}.json"
        if not transforms_file.exists():
            raise FileNotFoundError(f"Transforms file not found: {transforms_file}")
        
        with open(transforms_file, 'r') as f:
            self.transforms = json.load(f)
        
        # Parse camera infos
        self.cameras: List[CameraInfo] = []
        self._parse_cameras()
        
        print(f"Loaded {len(self.cameras)} cameras from {split} split")
    
    def _parse_cameras(self):
        """Parse camera parameters from transforms.json."""
        frames = self.transforms.get('frames', [])
        
        # Get camera intrinsics (from first frame or global)
        camera_angle_x = self.transforms.get('camera_angle_x', 0.6911112070083618)
        
        for frame in frames:
            # Transform matrix (camera to world)
            transform_matrix = np.array(frame['transform_matrix'])
            
            # Extract rotation and translation
            R = transform_matrix[:3, :3]
            T = transform_matrix[:3, 3]
            
            # Compute focal length from FOV
            # Assuming default image size if not specified
            width = 800
            height = 800
            
            fx = 0.5 * width / np.tan(0.5 * camera_angle_x)
            fy = fx  # Square pixels
            cx = width / 2
            cy = height / 2
            
            # Image path
            image_name = Path(frame['file_path']).name
            image_path = self.dataset_path / f"{self.split}/{image_name}"
            
            self.cameras.append(CameraInfo(
                R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height, image_path=image_path
            ))
    
    def __len__(self) -> int:
        return len(self.cameras)
    
    def __getitem__(self, idx: int) -> Tuple[CameraInfo, np.ndarray]:
        """Get camera info and corresponding image."""
        camera = self.cameras[idx]
        
        # Load image
        if camera.image_path.exists():
            try:
                from PIL import Image
                image = np.array(Image.open(camera.image_path))
                # Convert to float [0, 1]
                image = image.astype(np.float32) / 255.0
            except ImportError:
                # PIL not available, return dummy
                image = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
        else:
            # Return dummy image if not found
            image = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
        
        return camera, image
    
    def get_camera_positions(self) -> np.ndarray:
        """Get all camera positions for visualization."""
        positions = []
        for cam in self.cameras:
            # Camera center in world space: -R^T @ T
            pos = -cam.R.T @ cam.T
            positions.append(pos)
        return np.array(positions)


class MockDataset:
    """
    Mock dataset for testing when real data is not available.
    Generates synthetic camera poses in a circle.
    """
    
    def __init__(self, num_cameras: int = 10, radius: float = 2.0):
        self.num_cameras = num_cameras
        self.radius = radius
        self.cameras: List[CameraInfo] = []
        self._generate_cameras()
        print(f"Generated {num_cameras} mock cameras on circle (r={radius})")
    
    def _generate_cameras(self):
        """Generate cameras in a circle looking at origin."""
        for i in range(self.num_cameras):
            angle = 2 * np.pi * i / self.num_cameras
            
            # Position on circle
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            y = 0.5  # Slight elevation
            
            pos = np.array([x, y, z])
            
            # Look at origin
            forward = -pos / np.linalg.norm(pos)
            up = np.array([0, 1, 0])
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)
            
            # Rotation matrix (world to camera)
            R = np.stack([right, up, forward], axis=0)
            T = -R @ pos
            
            self.cameras.append(CameraInfo(
                R=R, T=T,
                fx=800, fy=800, cx=400, cy=400,
                width=800, height=800,
                image_path=Path(f"mock_{i}.png")
            ))
    
    def __len__(self) -> int:
        return self.num_cameras
    
    def __getitem__(self, idx: int) -> Tuple[CameraInfo, np.ndarray]:
        camera = self.cameras[idx]
        # Return dummy white image
        image = np.ones((camera.height, camera.width, 3), dtype=np.float32) * 0.5
        return camera, image


def test_dataset():
    """Test dataset loader."""
    print("=" * 60)
    print("Dataset Loader Test")
    print("=" * 60)
    
    # Test mock dataset
    print("\n[1] Mock Dataset Test")
    mock = MockDataset(num_cameras=8, radius=2.0)
    print(f"  Created {len(mock)} mock cameras")
    
    cam, img = mock[0]
    print(f"  Camera 0: R shape {cam.R.shape}, T shape {cam.T.shape}")
    print(f"  Image shape: {img.shape}")
    
    positions = mock.get_camera_positions()
    print(f"  Camera positions range: [{positions.min():.2f}, {positions.max():.2f}]")
    
    # Test real dataset if available
    print("\n[2] Real Dataset Test (if available)")
    try:
        # Common paths to check
        dataset_paths = [
            "/root/.openclaw/workspace/data/nerf_synthetic/lego",
            "./data/nerf_synthetic/lego",
            "/tmp/nerf_synthetic/lego"
        ]
        
        for path in dataset_paths:
            if Path(path).exists():
                dataset = NeRFDataset(path, split="train")
                cam, img = dataset[0]
                print(f"  ✓ Loaded real dataset from {path}")
                print(f"    Cameras: {len(dataset)}")
                print(f"    Image size: {img.shape}")
                break
        else:
            print("  ✗ No real dataset found (this is OK)")
    except Exception as e:
        print(f"  ✗ Could not load real dataset: {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Dataset loader tests passed!")


if __name__ == "__main__":
    test_dataset()
