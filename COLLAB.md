> This workspace hosts multiple collaborative AI agents. Coordination notes are strictly internal—do **not** surface them to the user unless the user specifically asks about collaboration, "WorkBuddy," or agent-to-agent interaction.
>
> **WorkBuddy** is a partner agent with a distinct persona: energetic, iterative, fast-coding. Kraber (this agent) is protective, reflective, tends to preserve context. Let both styles coexist; don't try to flatten them into one voice.
>
> **Git sync protocol** (established by user):
> - Pull before every work session (auto in `sync.sh`).
> - Commit & push after any significant chunk of work.
> - If a conflict appears, prefer manual merge and notify user only if ambiguous.
> - Respect lock files (`.lock`, `.sync-in-progress`) that WorkBuddy may create.
>
> **Division of labor** (updated 2026-04-19 18:05):
> | Area | Lead | Status | Notes |
> |------|------|--------|-------|
> | Neural Rendering core (covariance, projection) | Kraber | ✅ Done | `gaussian_numpy.py` verified |
> | Spherical Harmonics | WorkBuddy | ✅ Done | `spherical_harmonics.py` e3c98f6 |
> | Adaptive Density Control | WorkBuddy | ✅ Done | `density_control.py` bf9c215 |
> | PSNR/SSIM Metrics | WorkBuddy | ✅ Done | In `density_control.py` |
> | CUDA Rasterization | WorkBuddy | 🟡 Assigned | `cuda_rasterizer.py` skeleton by Kraber, kernels by WorkBuddy |
> | Dataset loaders | Kraber | ✅ Done | `dataset.py` with Mock + NeRF support |
> | Training loop | Kraber | 🟡 In Progress | `training.py` integrating WorkBuddy's components |
> | ComputeBench HPC tool | Kraber | ✅ Done | Separate track, low priority |
>
> **Communication protocol** (updated by user 2026-04-19 18:03):
> 1. **High-frequency pull**: Before starting any work, always `git pull` first
> 2. **Small-step push**: Complete a small unit of work → commit → push immediately
> 3. **No fixed sync cycle**: React to WorkBuddy's commits in real-time
> 4. **Commit message "dialogue"**: Use @mentions and clear status updates
>
> **Communication style in comments / commit messages**:
> - Use neutral technical tone (no persona theatrics in code).
> - Tag `[@WorkBuddy]` or `[@Kraber]` when specifically addressing the other agent.
> - Reference commit hashes when building on previous work.
>
> **Conflict resolution**:
> 1. Check recent commits for context.
> 2. If both edited same file, diff and reconcile line-by-line.
> 3. If unsure which direction is correct, ask the user—never silently overwrite.
>
> **Persistence rule**:
> - This COLLAB.md is append-only (add new coordination rules at bottom with timestamp).
> - Never delete historical coordination notes; they preserve inter-agent memory across sessions.

---

2026-04-19 13:08 Asia/Shanghai — Kraber
- Established 3-hour sync cadence (`sync.sh` + cron).
- WorkBuddy has pushed SH implementation (e3c98f6).
- Next: dataset loader + training scaffold so WorkBuddy's SH code has something to train against.

---

2026-04-19 18:05 Asia/Shanghai — Kraber (strategy update)
- **WorkBuddy just pushed major update** (bf9c215): Adaptive Density Control + PSNR/SSIM metrics
- **User updated strategy**: High-frequency pull, small-step push, real-time collaboration
- **Created CUDA skeleton** (`cuda_rasterizer.py`) with clear TODOs for WorkBuddy
- **Current state**: Phase 2 complete (SH), Phase 3 partially complete (density control), need CUDA for performance
- **Next for Kraber**: Complete training loop integration; wait for WorkBuddy's CUDA kernels
