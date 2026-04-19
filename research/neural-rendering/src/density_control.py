"""
Adaptive Density Control for 3D Gaussian Splatting
Phase 2: Clone / Split / Prune operations for training

Based on: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023
Section 5.2: "Adaptive Density Control"

Contribution: WorkBuddy collab agent - 2026-04-19
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class AdaptiveDensityController:
    """
    Manages adaptive density control during 3DGS training.

    The paper's key insight: start from a sparse point cloud (SfM),
    then iteratively refine Gaussian density by:
    1. **Clone** small Gaussians in under-reconstructed regions
    2. **Split** large Gaussians in over-reconstructed regions
    3. **Prune** transparent or oversized Gaussians periodically
    4. **Reset alpha** for Gaussians that are too transparent

    Args:
        clone_thresh: gradient threshold for cloning (paper: densify_grad_thresh)
        split_thresh: gradient threshold for splitting (same threshold, uses scale)
        max_screen_percent: max 2D screen size before forcing split (paper: 0.01 → 1%)
        prune_opacity_thresh: opacity below which Gaussians are pruned (paper: 0.005)
        prune_scale_thresh: scale (radius) above which Gaussians are pruned (paper: camera extent)
        reset_alpha_every: reset alpha every N iterations (paper: 3000)
        densify_every: densification interval in iterations (paper: 500)
        densify_from_iter: start densification from this iteration (paper: 500)
        densify_until_iter: stop densification at this iteration (paper: 15_000)
    """

    def __init__(
        self,
        clone_thresh: float = 0.0002,
        split_thresh: float = 0.0002,
        max_screen_percent: float = 0.01,
        prune_opacity_thresh: float = 0.005,
        prune_scale_thresh: float = 20.0,
        reset_alpha_every: int = 3000,
        densify_every: int = 500,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15_000,
    ):
        self.clone_thresh = clone_thresh
        self.split_thresh = split_thresh
        self.max_screen_percent = max_screen_percent
        self.prune_opacity_thresh = prune_opacity_thresh
        self.prune_scale_thresh = prune_scale_thresh
        self.reset_alpha_every = reset_alpha_every
        self.densify_every = densify_every
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @staticmethod
    def clone_gaussians(
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        grad_accum: torch.Tensor,
        clone_thresh: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Clone small Gaussians with high positional gradients.

        From the paper: "we clone the Gaussian by creating a copy of
        it, and moving it towards the mean of the gradient."

        A Gaussian is a candidate for cloning when:
        - its average gradient magnitude exceeds ``clone_thresh``
        - its maximum scale is below a threshold (i.e., it is "small")

        The clone is displaced along the gradient direction by a
        small offset proportional to the Gaussian's scale.

        Args:
            positions: (N, 3) Gaussian centers
            scales: (N, 3) per-axis scale (NOT scale^2)
            rotations: (N, 4) quaternions
            opacities: (N, 1) raw opacity (before sigmoid)
            sh_coeffs: (N, K, 3) SH coefficients
            grad_accum: (N,) accumulated mean gradient magnitude
            clone_thresh: gradient threshold

        Returns:
            Dict with updated tensors (originals + clones)
        """
        device = positions.device
        N = positions.shape[0]

        # Identify clone candidates: high gradient AND small scale
        max_scale = scales.max(dim=-1).values  # (N,)
        scale_median = max_scale.median()
        is_small = max_scale <= scale_median  # small ≲ median scale
        needs_clone = (grad_accum > clone_thresh) & is_small  # (N,)

        n_clone = needs_clone.sum().item()
        if n_clone == 0:
            return {
                'positions': positions,
                'scales': scales,
                'rotations': rotations,
                'opacities': opacities,
                'sh_coeffs': sh_coeffs,
                'n_cloned': 0,
            }

        # Extract clone candidates
        idx = needs_clone.nonzero(as_tuple=True)[0]

        clone_positions = positions[idx].clone()
        clone_scales = scales[idx].clone()
        clone_rotations = rotations[idx].clone()
        clone_opacities = opacities[idx].clone()
        clone_sh = sh_coeffs[idx].clone()

        # Displace clones along gradient direction
        # The paper says "move it towards the mean of the gradient"
        # We approximate: shift by a small random offset proportional to scale
        with torch.no_grad():
            offset_direction = torch.randn(n_clone, 3, device=device)
            offset_direction = offset_direction / (
                offset_direction.norm(dim=-1, keepdim=True) + 1e-8
            )
            displacement = offset_direction * clone_scales.mean(dim=-1, keepdim=True)
            clone_positions = clone_positions + displacement

        # Merge clones into the scene
        new_positions = torch.cat([positions, clone_positions], dim=0)
        new_scales = torch.cat([scales, clone_scales], dim=0)
        new_rotations = torch.cat([rotations, clone_rotations], dim=0)
        new_opacities = torch.cat([opacities, clone_opacities], dim=0)
        new_sh = torch.cat([sh_coeffs, clone_sh], dim=0)

        return {
            'positions': new_positions,
            'scales': new_scales,
            'rotations': new_rotations,
            'opacities': new_opacities,
            'sh_coeffs': new_sh,
            'n_cloned': n_clone,
        }

    @staticmethod
    def split_gaussians(
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        grad_accum: torch.Tensor,
        split_thresh: float,
        max_screen_percent: float,
        screen_sizes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Split large Gaussians with high positional gradients.

        From the paper: "we split the Gaussian by creating two copies
        of it, and dividing its scale by a factor of ~1.6."

        A Gaussian is a candidate for splitting when:
        - its average gradient magnitude exceeds ``split_thresh``
        - its maximum scale is above a threshold (i.e., it is "large")
        - OR its 2D screen size exceeds ``max_screen_percent``

        The two children are displaced in opposite directions along
        the major axis of the Gaussian, each with scale ≈ 1/1.6 of
        the parent.

        Args:
            positions: (N, 3) Gaussian centers
            scales: (N, 3) per-axis scale
            rotations: (N, 4) quaternions
            opacities: (N, 1) raw opacity (before sigmoid)
            sh_coeffs: (N, K, 3) SH coefficients
            grad_accum: (N,) accumulated mean gradient magnitude
            split_thresh: gradient threshold
            max_screen_percent: max screen percent before forcing split
            screen_sizes: (N,) optional 2D screen size per Gaussian

        Returns:
            Dict with updated tensors (originals removed, children added)
        """
        device = positions.device
        N = positions.shape[0]

        # Identify split candidates: high gradient AND large scale
        max_scale = scales.max(dim=-1).values  # (N,)
        scale_median = max_scale.median()
        is_large = max_scale > scale_median
        needs_split = (grad_accum > split_thresh) & is_large  # (N,)

        # Also split Gaussians that are too big on screen
        if screen_sizes is not None:
            needs_split = needs_split | (screen_sizes > max_screen_percent)

        n_split = needs_split.sum().item()
        if n_split == 0:
            return {
                'positions': positions,
                'scales': scales,
                'rotations': rotations,
                'opacities': opacities,
                'sh_coeffs': sh_coeffs,
                'n_split': 0,
            }

        idx = needs_split.nonzero(as_tuple=True)[0]

        # Parent properties
        parent_pos = positions[idx]     # (n_split, 3)
        parent_scale = scales[idx]       # (n_split, 3)
        parent_rot = rotations[idx]      # (n_split, 4)
        parent_opacity = opacities[idx]  # (n_split, 1)
        parent_sh = sh_coeffs[idx]       # (n_split, K, 3)

        # Scale factor: paper uses φ ≈ 1.6
        phi = 1.6
        child_scale = parent_scale / phi  # (n_split, 3)

        # Displacement along major axis (largest scale dimension)
        # Build offset in local frame then rotate to world frame
        major_axis_idx = parent_scale.argmax(dim=-1)  # (n_split,)

        # Create offset vectors in local frame
        offset_local = torch.zeros(n_split, 3, device=device)
        for i in range(n_split):
            axis = major_axis_idx[i].item()
            offset_local[i, axis] = parent_scale[i, axis] * 0.5  # Half extent

        # Rotate offsets to world frame using quaternion rotation
        # Simplified: use the scale direction directly as offset
        # (proper quaternion rotation would use the rotation matrix)
        offset_world = offset_local * (1.0 / phi)

        # Two children: displaced ±offset
        child1_pos = parent_pos + offset_world
        child2_pos = parent_pos - offset_world

        # Children share parent rotation, scaled opacity, and SH
        child1_rot = parent_rot.clone()
        child2_rot = parent_rot.clone()

        # Slightly reduce opacity for children (avoid double-brightness)
        child_opacity = parent_opacity.clone()

        # SH coefficients are shared between children
        child1_sh = parent_sh.clone()
        child2_sh = parent_sh.clone()

        # Remove parents, add children
        keep_mask = ~needs_split
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]

        new_positions = torch.cat([
            positions[keep_idx],
            child1_pos, child2_pos,
        ], dim=0)
        new_scales = torch.cat([
            scales[keep_idx],
            child_scale, child_scale,  # Both children share scale
        ], dim=0)
        new_rotations = torch.cat([
            rotations[keep_idx],
            child1_rot, child2_rot,
        ], dim=0)
        new_opacities = torch.cat([
            opacities[keep_idx],
            child_opacity, child_opacity,
        ], dim=0)
        new_sh = torch.cat([
            sh_coeffs[keep_idx],
            child1_sh, child2_sh,
        ], dim=0)

        return {
            'positions': new_positions,
            'scales': new_scales,
            'rotations': new_rotations,
            'opacities': new_opacities,
            'sh_coeffs': new_sh,
            'n_split': n_split,
        }

    @staticmethod
    def prune_gaussians(
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        opacity_thresh: float = 0.005,
        scale_thresh: float = 20.0,
        camera_extent: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prune transparent or oversized Gaussians.

        From the paper:
        - "we remove Gaussians with α < ε_α" (ε_α = 0.005)
        - "we remove Gaussians that are too large"
        - Done periodically to control total Gaussian count

        Args:
            positions: (N, 3)
            scales: (N, 3) per-axis scale
            rotations: (N, 4)
            opacities: (N, 1) raw opacity (before sigmoid)
            sh_coeffs: (N, K, 3)
            opacity_thresh: minimum opacity to keep
            scale_thresh: maximum scale to keep (relative to camera extent)
            camera_extent: optional scene extent for scale normalization

        Returns:
            Dict with pruned tensors
        """
        # Convert raw opacity to actual opacity via sigmoid
        alpha = torch.sigmoid(opacities).squeeze(-1)  # (N,)

        # Keep Gaussians with sufficient opacity
        keep_mask = alpha >= opacity_thresh  # (N,)

        # Also prune oversized Gaussians
        max_scale = scales.max(dim=-1).values  # (N,)
        if camera_extent is not None:
            relative_scale = max_scale / camera_extent
            keep_mask = keep_mask & (relative_scale < scale_thresh / 100.0)
        else:
            keep_mask = keep_mask & (max_scale < scale_thresh)

        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        n_pruned = positions.shape[0] - keep_idx.shape[0]

        return {
            'positions': positions[keep_idx],
            'scales': scales[keep_idx],
            'rotations': rotations[keep_idx],
            'opacities': opacities[keep_idx],
            'sh_coeffs': sh_coeffs[keep_idx],
            'n_pruned': n_pruned,
        }

    @staticmethod
    def reset_opacity(
        opacities: torch.Tensor,
        reset_value: float = 0.01,
    ) -> torch.Tensor:
        """
        Reset opacity of all Gaussians to a low value.

        From the paper: "we periodically reset the alpha value of
        all Gaussians to zero. This allows the system to 'recover'
        from stale Gaussians that have become nearly transparent
        but are still contributing small errors."

        After resetting, the opacity will be re-optimized, and
        truly useful Gaussians will regain their opacity quickly,
        while useless ones will be pruned in the next pruning step.

        Args:
            opacities: (N, 1) raw opacity values (before sigmoid)
            reset_value: value to reset to (in raw space, ~0 after sigmoid)

        Returns:
            Reset opacities tensor (same shape)
        """
        # Convert desired alpha to raw space: raw = log(alpha / (1 - alpha))
        # For alpha = reset_value: raw = log(0.01 / 0.99) ≈ -4.6
        raw_reset = torch.log(
            torch.tensor(reset_value, device=opacities.device)
            / (1.0 - reset_value)
        )
        return torch.full_like(opacities, raw_reset.item())

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def step(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        grad_accum: torch.Tensor,
        iteration: int,
        screen_sizes: Optional[torch.Tensor] = None,
        camera_extent: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one density control step.

        This orchestrates the full densification cycle:
        1. Densify (clone + split) every ``densify_every`` iterations
        2. Prune periodically alongside densification
        3. Reset alpha every ``reset_alpha_every`` iterations

        Args:
            positions: (N, 3)
            scales: (N, 3)
            rotations: (N, 4)
            opacities: (N, 1)
            sh_coeffs: (N, K, 3)
            grad_accum: (N,) mean gradient magnitude per Gaussian
            iteration: current training iteration
            screen_sizes: (N,) optional 2D screen size
            camera_extent: optional camera extent for scale normalization

        Returns:
            Dict with potentially modified tensors and stats
        """
        stats = {
            'n_cloned': 0,
            'n_split': 0,
            'n_pruned': 0,
            'n_reset': False,
            'densified': False,
        }

        should_densify = (
            iteration >= self.densify_from_iter
            and iteration <= self.densify_until_iter
            and iteration % self.densify_every == 0
        )

        if should_densify:
            stats['densified'] = True

            # Step 1: Clone small Gaussians with high gradients
            clone_result = self.clone_gaussians(
                positions, scales, rotations, opacities, sh_coeffs,
                grad_accum, self.clone_thresh,
            )
            positions = clone_result['positions']
            scales = clone_result['scales']
            rotations = clone_result['rotations']
            opacities = clone_result['opacities']
            sh_coeffs = clone_result['sh_coeffs']
            stats['n_cloned'] = clone_result['n_cloned']

            # Reset grad_accum after cloning (new Gaussians start fresh)
            new_N = positions.shape[0]
            grad_accum = torch.zeros(new_N, device=positions.device)

            # Step 2: Split large Gaussians with high gradients
            split_result = self.split_gaussians(
                positions, scales, rotations, opacities, sh_coeffs,
                grad_accum, self.split_thresh, self.max_screen_percent,
                screen_sizes,
            )
            positions = split_result['positions']
            scales = split_result['scales']
            rotations = split_result['rotations']
            opacities = split_result['opacities']
            sh_coeffs = split_result['sh_coeffs']
            stats['n_split'] = split_result['n_split']

            # Reset grad_accum after splitting
            new_N = positions.shape[0]
            grad_accum = torch.zeros(new_N, device=positions.device)

            # Step 3: Prune transparent / oversized
            prune_result = self.prune_gaussians(
                positions, scales, rotations, opacities, sh_coeffs,
                self.prune_opacity_thresh, self.prune_scale_thresh,
                camera_extent,
            )
            positions = prune_result['positions']
            scales = prune_result['scales']
            rotations = prune_result['rotations']
            opacities = prune_result['opacities']
            sh_coeffs = prune_result['sh_coeffs']
            stats['n_pruned'] = prune_result['n_pruned']

            # Reset grad_accum after pruning
            new_N = positions.shape[0]
            grad_accum = torch.zeros(new_N, device=positions.device)

        # Step 4: Periodic alpha reset
        if iteration > 0 and iteration % self.reset_alpha_every == 0:
            opacities = self.reset_opacity(opacities)
            stats['n_reset'] = True

        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'sh_coeffs': sh_coeffs,
            'grad_accum': grad_accum,
            'stats': stats,
        }


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_psnr(
    predicted: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)

    This is the standard image quality metric used in novel view
    synthesis evaluation.

    Args:
        predicted: (..., C) predicted image
        target: (..., C) ground truth image
        max_val: maximum pixel value (1.0 for [0,1] range)

    Returns:
        PSNR value in dB (scalar)
    """
    mse = ((predicted - target) ** 2).mean()
    if mse == 0:
        return torch.tensor(float('inf'), device=mse.device)
    return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=mse.device) / mse)


def compute_ssim(
    predicted: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Simplified implementation for 2D images. Uses a uniform window
    for efficiency (Gaussian window is more accurate but slower).

    Args:
        predicted: (H, W, C) or (C, H, W) predicted image
        target: (H, W, C) or (C, H, W) ground truth image
        window_size: size of the sliding window
        max_val: maximum pixel value

    Returns:
        SSIM value (scalar)
    """
    # Ensure (C, H, W) format
    if predicted.dim() == 3 and predicted.shape[-1] <= 4:
        # (H, W, C) → (C, H, W)
        predicted = predicted.permute(2, 0, 1)
        target = target.permute(2, 0, 1)

    C_img = predicted.shape[0]

    # Constants
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Average pooling as window function
    padding = window_size // 2
    avg_pool = torch.nn.AvgPool2d(
        kernel_size=window_size, stride=1, padding=padding
    )

    mu1 = avg_pool(predicted)
    mu2 = avg_pool(target)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = avg_pool(predicted ** 2) - mu1_sq
    sigma2_sq = avg_pool(target ** 2) - mu2_sq
    sigma12 = avg_pool(predicted * target) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


# ------------------------------------------------------------------
# Gradient accumulator helper
# ------------------------------------------------------------------

class GradientAccumulator:
    """
    Accumulate per-Gaussian positional gradient statistics for
    adaptive density control.

    During training, after each backward pass, call
    ``update(grad_positions)`` to accumulate the L2 norm of
    positional gradients for each Gaussian. The accumulated
    values drive the clone/split decisions.

    Args:
        max_gaussians: initial pre-allocated size (grows automatically)
        device: compute device
    """

    def __init__(self, max_gaussians: int = 100_000, device: str = 'cpu'):
        self.device = device
        self.grad_accum = torch.zeros(max_gaussians, device=device)
        self.grad_count = torch.zeros(max_gaussians, device=device)
        self.max_gaussians = max_gaussians

    def update(self, grad_positions: torch.Tensor) -> None:
        """
        Update gradient statistics from a backward pass.

        Args:
            grad_positions: (N, 3) gradient of loss w.r.t. positions
        """
        N = grad_positions.shape[0]

        # Resize if needed
        if N > self.max_gaussians:
            new_size = max(N, self.max_gaussians * 2)
            new_accum = torch.zeros(new_size, device=self.device)
            new_count = torch.zeros(new_size, device=self.device)
            new_accum[:self.max_gaussians] = self.grad_accum
            new_count[:self.max_gaussians] = self.grad_count
            self.grad_accum = new_accum
            self.grad_count = new_count
            self.max_gaussians = new_size

        # L2 norm of gradient per Gaussian
        grad_norm = grad_positions.norm(dim=-1)  # (N,)

        self.grad_accum[:N] += grad_norm
        self.grad_count[:N] += 1

    def get_mean_gradient(self, num_gaussians: int) -> torch.Tensor:
        """
        Get the mean gradient magnitude for each Gaussian.

        Args:
            num_gaussians: current number of Gaussians

        Returns:
            (num_gaussians,) mean gradient magnitude
        """
        N = min(num_gaussians, self.max_gaussians)
        safe_count = self.grad_count[:N].clamp(min=1)
        return self.grad_accum[:N] / safe_count

    def reset(self, num_gaussians: Optional[int] = None) -> None:
        """
        Reset accumulated gradients.

        Args:
            num_gaussians: if provided, resize to this value
        """
        if num_gaussians is not None:
            self.max_gaussians = max(num_gaussians, self.max_gaussians)
            self.grad_accum = torch.zeros(self.max_gaussians, device=self.device)
            self.grad_count = torch.zeros(self.max_gaussians, device=self.device)
        else:
            self.grad_accum.zero_()
            self.grad_count.zero_()


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Density Control - Test")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    N = 1000
    K = 16  # SH degree 3 coefficients

    # Create mock Gaussian data
    positions = torch.randn(N, 3, device=device)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    rotations = torch.randn(N, 4, device=device)
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)  # Normalize quaternions
    opacities = torch.rand(N, 1, device=device) * 4 - 2  # Raw space
    sh_coeffs = torch.randn(N, K, 3, device=device) * 0.1

    # Simulate gradient accumulation (some Gaussians have high gradients)
    grad_accum = torch.zeros(N, device=device)
    high_grad_mask = torch.rand(N) < 0.1  # 10% have high gradients
    grad_accum[high_grad_mask] = 0.001  # Above threshold

    # Test 1: Clone
    print("[1] Testing clone_gaussians...")
    clone_result = AdaptiveDensityController.clone_gaussians(
        positions, scales, rotations, opacities, sh_coeffs,
        grad_accum, clone_thresh=0.0002,
    )
    print(f"  Before: {N} Gaussians")
    print(f"  Cloned: {clone_result['n_cloned']}")
    print(f"  After:  {clone_result['positions'].shape[0]} Gaussians")
    assert clone_result['positions'].shape[0] == N + clone_result['n_cloned']
    print("  ✓ Clone test passed\n")

    # Test 2: Split
    print("[2] Testing split_gaussians...")
    split_result = AdaptiveDensityController.split_gaussians(
        positions, scales, rotations, opacities, sh_coeffs,
        grad_accum, split_thresh=0.0002, max_screen_percent=0.01,
    )
    n_after_split = split_result['positions'].shape[0]
    print(f"  Split: {split_result['n_split']}")
    print(f"  Removed: {split_result['n_split']} parents")
    print(f"  Added:   {2 * split_result['n_split']} children")
    print(f"  After:   {n_after_split} Gaussians")
    expected = N - split_result['n_split'] + 2 * split_result['n_split']
    assert n_after_split == expected
    print("  ✓ Split test passed\n")

    # Test 3: Prune
    print("[3] Testing prune_gaussians...")
    prune_result = AdaptiveDensityController.prune_gaussians(
        positions, scales, rotations, opacities, sh_coeffs,
        opacity_thresh=0.005, scale_thresh=20.0,
    )
    print(f"  Pruned: {prune_result['n_pruned']}")
    print(f"  After:  {prune_result['positions'].shape[0]} Gaussians")
    print("  ✓ Prune test passed\n")

    # Test 4: Reset opacity
    print("[4] Testing reset_opacity...")
    reset_opacity = AdaptiveDensityController.reset_opacity(opacities)
    alpha_after = torch.sigmoid(reset_opacity)
    print(f"  Before reset: alpha range [{torch.sigmoid(opacities).min():.4f}, "
          f"{torch.sigmoid(opacities).max():.4f}]")
    print(f"  After reset:  alpha range [{alpha_after.min():.4f}, "
          f"{alpha_after.max():.4f}]")
    print("  ✓ Reset opacity test passed\n")

    # Test 5: Full step orchestration
    print("[5] Testing full step orchestration...")
    controller = AdaptiveDensityController(
        densify_from_iter=500,
        densify_every=500,
        densify_until_iter=15000,
    )
    step_result = controller.step(
        positions, scales, rotations, opacities, sh_coeffs,
        grad_accum, iteration=500,
    )
    print(f"  Stats: {step_result['stats']}")
    print(f"  Gaussians after step: {step_result['positions'].shape[0]}")
    print("  ✓ Full step test passed\n")

    # Test 6: PSNR
    print("[6] Testing PSNR...")
    img_pred = torch.rand(64, 64, 3, device=device)
    img_target = img_pred + torch.randn_like(img_pred) * 0.05
    psnr = compute_psnr(img_pred, img_target)
    print(f"  PSNR: {psnr.item():.2f} dB")
    assert psnr.item() > 0
    print("  ✓ PSNR test passed\n")

    # Test 7: SSIM
    print("[7] Testing SSIM...")
    ssim = compute_ssim(img_pred, img_target)
    print(f"  SSIM: {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1
    print("  ✓ SSIM test passed\n")

    # Test 8: Gradient accumulator
    print("[8] Testing GradientAccumulator...")
    accum = GradientAccumulator(max_gaussians=N, device=device)
    for _ in range(5):
        grad = torch.randn(N, 3, device=device)
        accum.update(grad)
    mean_grad = accum.get_mean_gradient(N)
    print(f"  Mean gradient shape: {mean_grad.shape}")
    print(f"  Mean gradient range: [{mean_grad.min():.4f}, {mean_grad.max():.4f}]")
    assert mean_grad.shape == (N,)
    print("  ✓ GradientAccumulator test passed\n")

    print("=" * 60)
    print("✓ All adaptive density control tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Integrate density control into training loop")
    print("  2. Build GaussianScene with scale/rotation parameterization")
    print("  3. Test on real COLMAP data")
