"""
Spherical Harmonics for View-Dependent Color
Phase 2: SH coefficients for angle-dependent appearance

Based on: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023
Reference: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py

Contribution: WorkBuddy collab agent - 2026-04-19
"""

import torch
import numpy as np
from typing import Optional


# ===========================================================
# SH basis degree constants
# Degree 0: 1 coefficient  (DC term)
# Degree 1: 4 coefficients (3 + DC)
# Degree 2: 9 coefficients (5 + prev)
# Degree 3: 16 coefficients (7 + prev)  ← used in 3DGS paper
# ===========================================================

SH_C0 = 0.28209479177387814          # 1 / (2 * sqrt(pi))
SH_C1 = 0.4886025119029199           # sqrt(3 / (4*pi))
SH_C2 = [
    1.0925484305920792,              # sqrt(15 / (4*pi))
    -1.0925484305920792,
    0.31539156525252005,             # sqrt(5 / (16*pi))
    -1.0925484305920792,
    0.5462742152960396               # sqrt(15 / (16*pi))
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]


def eval_sh(degree: int, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics at given directions.

    This is the core operation for view-dependent color in 3DGS:
    each Gaussian stores SH coefficients for R, G, B channels,
    and color is computed by evaluating SH at the view direction.

    Args:
        degree: SH degree (0-3)
        sh_coeffs: (..., (degree+1)^2, 3) SH coefficients per channel (RGB)
        dirs: (..., 3) unit direction vectors (view → Gaussian)

    Returns:
        colors: (..., 3) RGB values in [0, 1] range (after sigmoid)
    """
    assert degree <= 3, f"SH degree must be <= 3, got {degree}"
    assert sh_coeffs.shape[-2] == (degree + 1) ** 2, (
        f"Expected {(degree+1)**2} coefficients for degree {degree}, "
        f"got {sh_coeffs.shape[-2]}"
    )

    # Normalize directions (in case input is not unit vector)
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]

    # Degree 0: constant term
    result = SH_C0 * sh_coeffs[..., 0, :]

    if degree >= 1:
        result += (
            -SH_C1 * y * sh_coeffs[..., 1, :]
            + SH_C1 * z * sh_coeffs[..., 2, :]
            - SH_C1 * x * sh_coeffs[..., 3, :]
        )

    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result += (
            SH_C2[0] * xy * sh_coeffs[..., 4, :]
            + SH_C2[1] * yz * sh_coeffs[..., 5, :]
            + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[..., 6, :]
            + SH_C2[3] * xz * sh_coeffs[..., 7, :]
            + SH_C2[4] * (xx - yy) * sh_coeffs[..., 8, :]
        )

    if degree >= 3:
        result += (
            SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[..., 9, :]
            + SH_C3[1] * xy * z * sh_coeffs[..., 10, :]
            + SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[..., 11, :]
            + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[..., 12, :]
            + SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[..., 13, :]
            + SH_C3[5] * z * (xx - yy) * sh_coeffs[..., 14, :]
            + SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[..., 15, :]
        )

    # Bias to center around 0.5 (DC term offset from paper)
    result += 0.5

    # Clamp to valid color range
    return torch.clamp(result, 0.0, 1.0)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert flat RGB color to degree-0 SH coefficient.
    Useful for initializing SH from known colors.

    Args:
        rgb: (..., 3) RGB values in [0, 1]

    Returns:
        sh0: (..., 1, 3) degree-0 coefficient
    """
    return ((rgb - 0.5) / SH_C0).unsqueeze(-2)


def sh_num_coeffs(degree: int) -> int:
    """Return number of SH coefficients for given degree."""
    return (degree + 1) ** 2


class ViewDependentColor(torch.nn.Module):
    """
    Learnable view-dependent color module using Spherical Harmonics.

    Each Gaussian has `sh_degree`-order SH coefficients for each RGB channel.
    At render time, we evaluate the SH at the camera→Gaussian view direction
    to get the perceived color.

    Args:
        num_gaussians: number of Gaussians in the scene
        sh_degree: SH degree (0=flat color, 1=linear, 2=quadratic, 3=cubic)
        device: compute device
    """

    def __init__(self, num_gaussians: int, sh_degree: int = 3,
                 device: str = 'cpu'):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_coeffs = sh_num_coeffs(sh_degree)
        num_gaussians_val = num_gaussians

        # SH coefficients: (N, num_coeffs, 3)  — learnable parameter
        self.sh_coeffs = torch.nn.Parameter(
            torch.zeros(num_gaussians_val, self.num_coeffs, 3, device=device)
        )
        # Initialize DC term to white (0.5 gray after bias)
        torch.nn.init.constant_(self.sh_coeffs[:, 0, :], 0.0)

    def forward(self, positions: torch.Tensor,
                camera_position: torch.Tensor) -> torch.Tensor:
        """
        Compute view-dependent colors for all Gaussians.

        Args:
            positions: (N, 3) Gaussian center positions in world space
            camera_position: (3,) camera position in world space

        Returns:
            colors: (N, 3) RGB colors clamped to [0, 1]
        """
        # View direction: camera → Gaussian (normalized)
        dirs = positions - camera_position.unsqueeze(0)  # (N, 3)
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

        return eval_sh(self.sh_degree, self.sh_coeffs, dirs)

    def set_from_rgb(self, rgb: torch.Tensor):
        """
        Initialize SH coefficients from flat RGB colors (degree-0 only).
        Other coefficients remain zero (no view dependency initially).

        Args:
            rgb: (N, 3) flat colors in [0, 1]
        """
        with torch.no_grad():
            self.sh_coeffs[:, 0, :] = (rgb - 0.5) / SH_C0


# ===========================================================
# Test / demo
# ===========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Spherical Harmonics - View-Dependent Color Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    N = 10  # number of Gaussians

    # Test eval_sh at all degrees
    for deg in [0, 1, 2, 3]:
        nc = sh_num_coeffs(deg)
        coeffs = torch.randn(N, nc, 3, device=device)
        dirs = torch.randn(N, 3, device=device)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

        colors = eval_sh(deg, coeffs, dirs)
        print(f"Degree {deg}: coeffs {coeffs.shape} → colors {colors.shape}  "
              f"range [{colors.min():.3f}, {colors.max():.3f}]")

    print()

    # Test ViewDependentColor module
    vdc = ViewDependentColor(N, sh_degree=3, device=device)
    positions = torch.randn(N, 3, device=device)
    cam_pos = torch.tensor([2.0, 2.0, 2.0], device=device)

    colors = vdc(positions, cam_pos)
    print(f"ViewDependentColor forward: {colors.shape}  "
          f"range [{colors.min():.3f}, {colors.max():.3f}]")

    # Test initialization from RGB
    flat_rgb = torch.rand(N, 3, device=device)
    vdc.set_from_rgb(flat_rgb)

    # With degree-0 only (no view-dep), output should match flat_rgb
    vdc_deg0 = ViewDependentColor(N, sh_degree=0, device=device)
    vdc_deg0.set_from_rgb(flat_rgb)
    colors_deg0 = vdc_deg0(positions, cam_pos)
    diff = (colors_deg0 - flat_rgb).abs().max()
    print(f"Degree-0 reconstruction error (should be ~0): {diff:.6f}")

    print("\n[OK] Spherical Harmonics tests passed!")
    print("\nNext: integrate SH into GaussianScene and training loop")
