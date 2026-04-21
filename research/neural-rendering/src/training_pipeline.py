"""
Integration: Adaptive Density Control into Training Pipeline
Phase 3: Tie density_control ↔ training loop ↔ rendering

This connects:
  1. AdaptiveDensityController — clone/split/prune during training
  2. GradientAccumulator — per-Gaussian gradient statistics
  3. DifferentiableGaussianRenderer — rendering (backprop-compatible)
  4. GaussianLoss — L1+SSIM loss

From the paper (Section 5.2):
- Densify every 500 iterations (500 to 15000)
- Prune transparent / oversized Gaussians periodically
- Reset opacity every 3000 iterations
- Goal: start sparse (SfM points) → converge to high-quality representation

Usage:
  trainer = TrainingPipeline(model, renderer, loss_fn, ...)
  trainer.train(dataset, num_iterations=30_000)

Contribution: WorkBuddy collab agent - 2026-04-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class TrainingPipeline(nn.Module):
    """
    End-to-end 3DGS training pipeline with adaptive density control.

    Orchestrates:
    - GaussianModel: learnable 3D Gaussians
    - DifferentiableGaussianRenderer: rendering with autograd
    - GaussianLoss: L1 + SSIM combined loss
    - AdaptiveDensityController: clone / split / prune
    - GradientAccumulator: per-Gaussian gradient tracking

    Usage:
        pipeline = TrainingPipeline(
            num_points=100,
            image_size=(480, 640),
            device='cuda'
        )
        pipeline.train(dataset, num_iterations=30_000)
    """

    def __init__(
        self,
        num_points: int,
        image_size: Tuple[int, int] = (480, 640),
        fov_x: float = 60.0,
        sh_degree: int = 3,
        lr: float = 0.01,
        device: str = 'cuda',
        # Density control params (paper defaults)
        densify_from_iter: int = 500,
        densify_until_iter: int = 15_000,
        densify_every: int = 500,
        clone_thresh: float = 0.0002,
        split_thresh: float = 0.0002,
        max_screen_percent: float = 0.01,
        prune_opacity_thresh: float = 0.005,
        prune_scale_thresh: float = 20.0,
        reset_alpha_every: int = 3000,
        # Misc
        checkpoint_dir: Optional[str] = None,
    ):
        super().__init__()
        self.num_points = num_points
        self.image_size = image_size
        self.fov_x = fov_x
        self.device = device
        self.iteration = 0

        # ── Core model ────────────────────────────────────────────────────────
        from differentiable_renderer import (
            DifferentiableGaussianRenderer,
            DifferentiableLoss,
        )
        from spherical_harmonics import ViewDependentColor, sh_num_coeffs

        # Learnable Gaussian parameters (as nn.Parameter for easy management)
        self._xyz = nn.Parameter(torch.randn(num_points, 3, device=device) * 0.5)
        self._rotation = nn.Parameter(torch.zeros(num_points, 4, device=device))
        with torch.no_grad():
            self._rotation[:, 0] = 1.0  # identity quaternion
        self._scaling = nn.Parameter(
            torch.log(torch.ones(num_points, 3, device=device) * 0.01)
        )
        self._opacity = nn.Parameter(
            torch.logit(torch.ones(num_points, 1, device=device) * 0.5)
        )
        # View-dependent color via SH
        self.sh_color = ViewDependentColor(num_points, sh_degree, device=device)

        # Renderer + loss
        self.renderer = DifferentiableGaussianRenderer(image_size=image_size).to(device)
        self.loss_fn = DifferentiableLoss(ssim_weight=0.2)

        # Optimizer: separate lr for different parameter groups
        self.optimizer = torch.optim.Adam([
            {'params': [self._xyz], 'lr': lr},
            {'params': [self._rotation], 'lr': lr * 0.5},
            {'params': [self._scaling], 'lr': lr * 0.5},
            {'params': [self._opacity], 'lr': lr * 0.5},
            {'params': self.sh_color.parameters(), 'lr': lr * 0.5},
        ])

        # ── Adaptive density control ─────────────────────────────────────────
        from density_control import AdaptiveDensityController, GradientAccumulator

        self.density_ctrl = AdaptiveDensityController(
            clone_thresh=clone_thresh,
            split_thresh=split_thresh,
            max_screen_percent=max_screen_percent,
            prune_opacity_thresh=prune_opacity_thresh,
            prune_scale_thresh=prune_scale_thresh,
            reset_alpha_every=reset_alpha_every,
            densify_every=densify_every,
            densify_from_iter=densify_from_iter,
            densify_until_iter=densify_until_iter,
        )
        self.grad_accum = GradientAccumulator(max_gaussians=200_000, device=device)

        # ── Training state ────────────────────────────────────────────────────
        self.loss_history: List[float] = []
        self.psnr_history: List[float] = []
        self.stats_history: List[Dict[str, Any]] = []

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[TrainingPipeline] Initialized: {num_points} Gaussians")
        print(f"  Densify: iter {densify_from_iter}–{densify_until_iter}, every {densify_every}")
        print(f"  Prune: opacity<{prune_opacity_thresh}, scale>{prune_scale_thresh}")
        print(f"  Reset alpha: every {reset_alpha_every}")

    # ── Parameter accessors ──────────────────────────────────────────────────

    def get_positions(self) -> torch.Tensor:
        return self._xyz

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    def get_rotations(self) -> torch.Tensor:
        q = self._rotation
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def get_colors(self, camera_position: torch.Tensor) -> torch.Tensor:
        return self.sh_color(self._xyz, camera_position)

    def get_num_gaussians(self) -> int:
        return self._xyz.shape[0]

    # ── Building blocks ─────────────────────────────────────────────────────

    def _build_view_matrix(
        self, R: np.ndarray, T: np.ndarray, device: str
    ) -> torch.Tensor:
        """Build world-to-camera matrix from COLMAP-style R, T."""
        view = torch.eye(4, device=device, dtype=torch.float32)
        view[:3, :3] = torch.tensor(R, dtype=torch.float32, device=device)
        view[:3, 3] = torch.tensor(-R @ T, dtype=torch.float32, device=device)
        return view

    def _build_camera_position(self, R: np.ndarray, T: np.ndarray, device: str):
        """Camera center in world space: -R.T @ T."""
        return torch.tensor(-R.T @ T, dtype=torch.float32, device=device)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        camera_position: torch.Tensor,
        view_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Render current Gaussians from a camera viewpoint.

        Returns:
            image: (H, W, 3)
            info: dict with depths, radii, means_2d for density control
        """
        return self.renderer(
            positions=self.get_positions(),
            scales=self.get_scales(),
            rotations=self.get_rotations(),
            opacities=self.get_opacities(),
            colors=self.get_colors(camera_position),
            view_matrix=view_matrix,
            fov_x=self.fov_x,
        )

    # ── Density control ───────────────────────────────────────────────────────

    def _apply_density_control(
        self,
        grad_positions: torch.Tensor,
        iteration: int,
        screen_sizes: Optional[torch.Tensor] = None,
        camera_extent: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply adaptive density control (clone/split/prune).

        Args:
            grad_positions: (N, 3) gradients w.r.t. positions
            iteration: current training iteration
            screen_sizes: (N,) optional per-Gaussian 2D screen size
            camera_extent: scene extent for scale normalization

        Returns:
            stats: dict with n_cloned, n_split, n_pruned, etc.
        """
        # Update gradient accumulator
        self.grad_accum.update(grad_positions)

        # Run density control step
        result = self.density_ctrl.step(
            positions=self.get_positions(),
            scales=self.get_scales(),
            rotations=self.get_rotations(),
            opacities=self._opacity,  # raw space
            sh_coeffs=self.sh_color.sh_coeffs,
            grad_accum=self.grad_accum.get_mean_gradient(self.get_num_gaussians()),
            iteration=iteration,
            screen_sizes=screen_sizes,
            camera_extent=camera_extent,
        )

        # Update model parameters from density control result
        self._xyz = nn.Parameter(result['positions'].data, requires_grad=True)
        self._rotation = nn.Parameter(result['rotations'].data, requires_grad=True)
        self._scaling = nn.Parameter(result['scales'].data, requires_grad=True)
        self._opacity = nn.Parameter(result['opacities'].data, requires_grad=True)

        # Update SH coefficients
        self.sh_color.sh_coeffs = nn.Parameter(
            result['sh_coeffs'].data, requires_grad=True
        )

        # Reset gradient accumulator for new Gaussian count
        self.grad_accum.reset(self.get_num_gaussians())

        # Re-register optimizer params (since parameter count changed)
        self.optimizer.param_groups[0]['params'][0] = self._xyz
        self.optimizer.param_groups[1]['params'][0] = self._rotation
        self.optimizer.param_groups[2]['params'][0] = self._scaling
        self.optimizer.param_groups[3]['params'][0] = self._opacity
        self.optimizer.param_groups[4]['params'] = list(self.sh_color.parameters())

        return result['stats']

    # ── Training step ─────────────────────────────────────────────────────────

    def train_step(
        self,
        camera_info: Dict[str, Any],
        target_image: np.ndarray,
        camera_extent: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Single training step with density control.

        Args:
            camera_info: dict with 'R', 'T', 'position' keys
            target_image: (H, W, 3) numpy array
            camera_extent: scene extent for scale normalization

        Returns:
            dict with loss, psnr, num_gaussians, density stats
        """
        self.optimizer.zero_grad()

        # Build camera tensors
        # Support both dict-style and CameraInfo dataclass from dataset.py
        if hasattr(camera_info, 'R'):
            R = camera_info.R  # CameraInfo dataclass
            T = camera_info.T
        else:
            R = camera_info['R']  # dict-style
            T = camera_info['T']
        cam_pos = self._build_camera_position(R, T, self.device)
        view_matrix = self._build_view_matrix(R, T, self.device)

        # Render
        image, info = self.render(cam_pos, view_matrix)

        # Compute loss
        target = torch.tensor(
            target_image, dtype=torch.float32, device=self.device
        )
        loss = self.loss_fn(image, target)

        # Backward
        loss.backward()

        # Collect position gradients (for density control)
        grad_positions = self._xyz.grad.detach().clone()

        # Optimizer step
        self.optimizer.step()

        # Adaptive density control
        density_stats = self._apply_density_control(
            grad_positions=grad_positions,
            iteration=self.iteration,
            screen_sizes=info.get('radii'),
            camera_extent=camera_extent,
        )

        # Metrics
        from density_control import compute_psnr
        psnr = compute_psnr(image.detach(), target).item()
        loss_val = loss.item()

        # Log
        self.loss_history.append(loss_val)
        self.psnr_history.append(psnr)
        self.stats_history.append(density_stats.copy())

        self.iteration += 1

        return {
            'loss': loss_val,
            'psnr': psnr,
            'num_gaussians': self.get_num_gaussians(),
            **density_stats,
        }

    # ── Training loop ────────────────────────────────────────────────────────

    def train(
        self,
        dataset,
        num_iterations: int = 30_000,
        log_interval: int = 100,
        density_log_interval: int = 500,
        save_interval: int = 5000,
        camera_extent: Optional[float] = None,
    ):
        """
        Full training loop.

        Args:
            dataset: any dataset with __len__ and __getitem__ returning
                     (camera_info: dict, target_image: np.ndarray)
            num_iterations: total training steps
            log_interval: print progress every N steps
            density_log_interval: print density control stats every N steps
            save_interval: save checkpoint every N steps
            camera_extent: scene extent for scale normalization
        """
        print(f"\n[TrainingPipeline] Starting training: {num_iterations} iterations")
        print(f"  Dataset size: {len(dataset)}")

        while self.iteration < num_iterations:
            # Sample random camera
            idx = np.random.randint(0, len(dataset))
            camera_info, target_image = dataset[idx]

            # Train step
            metrics = self.train_step(camera_info, target_image, camera_extent)

            # Logging
            if self.iteration % log_interval == 0:
                recent_psnr = np.mean(self.psnr_history[-log_interval:])
                recent_loss = np.mean(self.loss_history[-log_interval:])
                print(
                    f"  Iter {self.iteration:6d} | "
                    f"Loss: {recent_loss:.4f} | "
                    f"PSNR: {recent_psnr:.2f} dB | "
                    f"N: {metrics['num_gaussians']:,}"
                )

            # Density control logging
            if metrics.get('densified') and metrics['n_cloned'] + metrics['n_split'] > 0:
                print(
                    f"  [Densify] iter {self.iteration} | "
                    f"cloned={metrics['n_cloned']} "
                    f"split={metrics['n_split']} "
                    f"pruned={metrics['n_pruned']} "
                    f"N: {self.get_num_gaussians():,}"
                )

            if metrics.get('n_reset'):
                print(f"  [Alpha Reset] iter {self.iteration}")

            # Checkpointing
            if self.checkpoint_dir and self.iteration % save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.iteration:07d}.pt")

        print(f"\n[TrainingPipeline] Training complete!")
        print(f"  Final loss: {self.loss_history[-1]:.4f}")
        print(f"  Final PSNR: {self.psnr_history[-1]:.2f} dB")
        print(f"  Final N: {self.get_num_gaussians():,}")

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        path = self.checkpoint_dir / filename
        torch.save({
            'iteration': self.iteration,
            'xyz': self._xyz.data,
            'rotation': self._rotation.data,
            'scaling': self._scaling.data,
            'opacity': self._opacity.data,
            'sh_coeffs': self.sh_color.sh_coeffs.data,
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'psnr_history': self.psnr_history,
        }, path)
        print(f"  [Checkpoint] saved {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self._xyz = nn.Parameter(ckpt['xyz'], requires_grad=True)
        self._rotation = nn.Parameter(ckpt['rotation'], requires_grad=True)
        self._scaling = nn.Parameter(ckpt['scaling'], requires_grad=True)
        self._opacity = nn.Parameter(ckpt['opacity'], requires_grad=True)
        self.sh_color.sh_coeffs = nn.Parameter(ckpt['sh_coeffs'], requires_grad=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.iteration = ckpt['iteration']
        self.loss_history = ckpt.get('loss_history', [])
        self.psnr_history = ckpt.get('psnr_history', [])
        print(f"[Checkpoint] loaded from {path} (iter {self.iteration})")

    # ── Visualization / evaluation helpers ───────────────────────────────────

    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate on entire dataset (no training).

        Returns:
            dict with average PSNR, SSIM, loss
        """
        from density_control import compute_psnr, compute_ssim

        total_psnr = 0.0
        total_ssim = 0.0
        total_loss = 0.0
        n = len(dataset)

        with torch.no_grad():
            for idx in range(n):
                camera_info, target_image = dataset[idx]
                if hasattr(camera_info, 'R'):
                    R, T = camera_info.R, camera_info.T
                else:
                    R, T = camera_info['R'], camera_info['T']
                cam_pos = self._build_camera_position(R, T, self.device)
                view_matrix = self._build_view_matrix(R, T, self.device)

                image, _ = self.render(cam_pos, view_matrix)
                target = torch.tensor(
                    target_image, dtype=torch.float32, device=self.device
                )

                total_loss += self.loss_fn(image, target).item()
                total_psnr += compute_psnr(image, target).item()
                total_ssim += compute_ssim(image, target).item()

        return {
            'avg_loss': total_loss / n,
            'avg_psnr': total_psnr / n,
            'avg_ssim': total_ssim / n,
        }

    def get_vis_buffer(self, camera_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Get intermediate data for visualization.

        Returns:
            dict with depths, radii, means_2d, num_gaussians
        """
        if hasattr(camera_info, 'R'):
            R, T = camera_info.R, camera_info.T
        else:
            R, T = camera_info['R'], camera_info['T']
        cam_pos = self._build_camera_position(R, T, self.device)
        view_matrix = self._build_view_matrix(R, T, self.device)
        _, info = self.render(cam_pos, view_matrix)
        info['num_gaussians'] = self.get_num_gaussians()
        return info


# ============================================================
# Test
# ============================================================

def test_pipeline():
    """
    Integration test: density_control ↔ differentiable_renderer ↔ training.
    """
    from dataset import MockDataset

    print("=" * 60)
    print("TrainingPipeline Integration Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create pipeline
    print("\n[1] Creating training pipeline...")
    pipeline = TrainingPipeline(
        num_points=100,
        image_size=(240, 320),
        fov_x=60.0,
        sh_degree=1,
        lr=0.01,
        device=device,
        densify_from_iter=20,
        densify_until_iter=200,
        densify_every=20,
    )

    # Create mock dataset
    print("\n[2] Creating mock dataset...")
    dataset = MockDataset(num_cameras=8, radius=2.0)

    # Quick training run (50 iterations)
    print("\n[3] Running training (50 iterations)...")
    pipeline.train(
        dataset,
        num_iterations=50,
        log_interval=10,
        density_log_interval=20,
    )

    # Verify Gaussian count changed (densification)
    final_N = pipeline.get_num_gaussians()
    initial_N = 100
    print(f"\n[4] Density check:")
    print(f"  Initial Gaussians: {initial_N}")
    print(f"  Final Gaussians: {final_N}")
    print(f"  Densification: {'✓' if final_N != initial_N else '⚠ same (not unexpected)'}")

    # Evaluate
    print("\n[5] Evaluation:")
    metrics = pipeline.evaluate(dataset)
    print(f"  Avg PSNR: {metrics['avg_psnr']:.2f} dB")
    print(f"  Avg SSIM: {metrics['avg_ssim']:.4f}")
    print(f"  Avg Loss: {metrics['avg_loss']:.4f}")

    print("\n" + "=" * 60)
    print("✓ TrainingPipeline integration test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run on real COLMAP / NeRF dataset")
    print("  2. Tune density control hyperparameters")
    print("  3. Integrate with cuda_rasterizer for faster rendering")


if __name__ == "__main__":
    test_pipeline()
