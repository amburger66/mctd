#!/usr/bin/env python3
"""
Inference script for pushblock_offline diffusion planning.

Current pushblock_offline observation layout (6D):
  obs = [tcp_x, tcp_y, block_x, block_y, yaw_a, yaw_b]
where (yaw_a, yaw_b) is a continuous 2D encoding of block yaw (often cos,sin).

Usage (run from submodules/mctd/):
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode unguided
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --num_samples 4

  # 4-value shorthand: tcp_x,tcp_y,block_x,block_y (yaw dims filled with obs mean)
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --start='-0.2,0.0,-0.15,0.0' --goal='0.1,0.1,0.05,0.05'

  # Full obs spec: provide all obs_dim values directly (e.g. 6 values for pushblock_offline)
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --start='...' --goal='...'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_MCTD_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MCTD_ROOT))


def _mkdir(path: Path) -> None:
    """mkdir -p with 0o777 permissions, bypassing the process umask.

    When the script runs inside Docker as root, directories created with the
    default umask (022) end up as root:root 755 on the bind-mounted host
    filesystem — the host user can read but not write them.  Forcing 0o777
    makes every new directory component world-writable so the host user can
    delete, rename, or overwrite outputs without sudo.
    """
    prev = os.umask(0)
    try:
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
    finally:
        os.umask(prev)


# Default indices for the 6D pushblock_offline observation vector:
#   [tcp_x, tcp_y, block_x, block_y, yaw_a, yaw_b]
_TCP_XY_IDX = (0, 1)
_BLOCK_XY_IDX = (2, 3)
_BLOCK_YAW2_IDX = (4, 5)  # visualization expects (cos, sin) ordering

# 4-value shorthand: (tcp_x, tcp_y, block_x, block_y) → obs dims 0,1,2,3.
_SHORTHAND_4D_MAP = [0, 1, 2, 3]

# Set from args in main() before load_config() is called.
_FRAME_STACK: int = 10


def parse_args():
    parser = argparse.ArgumentParser(description="PushBoundary2D inference")
    parser.add_argument(
        "--checkpoint",
        default="/workspace/checkpoints/epoch=1063-step=50000.ckpt",
        help="Path to .ckpt or wandb run id (8 chars)",
    )
    parser.add_argument(
        "--mode",
        choices=["unguided", "guided", "mctd"],
        default="unguided",
        help="Inference mode",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of trajectories"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference_results/cylinder"),
        help="Output directory for visualizations (GIF/MP4) and trajectories",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="0.0,0.0,-0.13,0.0",
        help=(
            "Start observation for guided mode. Either 4 values (tcp_x,tcp_y,block_x,block_y) "
            "as a shorthand — remaining dims filled with obs mean — or the full obs_dim "
            "observation vector (e.g. 6 values for pushblock_offline). "
            "Use --start='-0.2,0,-0.15,0' for negative values."
        ),
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help=(
            "Goal observation for guided mode. Same format as --start: 4-value shorthand "
            "(tcp_x,tcp_y,block_x,block_y) or full obs_dim vector."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help=(
            "Guidance scale for guided mode when not using --guidance_scales (default: from config). "
            "Ignored when --guidance_scales is set."
        ),
    )
    parser.add_argument(
        "--guidance_scales",
        type=str,
        default=None,
        help=(
            "Guided mode only: comma-separated guidance scales (e.g. 0.1,0.5,1,2). "
            "Each run writes to output_dir/guided/scale_<tag>/. "
            "Omit for a single run under output_dir/guided/ using --guidance_scale or checkpoint default."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Planning horizon in env steps (default: episode_len)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=10,
        help="frame_stack used during training (default: 4)",
    )
    parser.add_argument(
        "--yaw_encoding",
        choices=["cos_sin", "sin_cos"],
        default="cos_sin",
        help=(
            "How the 2 yaw dims encode yaw; used only for visualization. "
            "If the rotation direction looks flipped, try --yaw_encoding sin_cos."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format for trajectory visualizations (default: gif)",
    )
    parser.add_argument(
        "--no_reset_seed_between_guidance_scales",
        action="store_true",
        help=(
            "Guided guidance sweep only: do not reset torch/numpy RNG before each guidance scale "
            "(default resets so start/goal and noise match across scales)."
        ),
    )
    parser.add_argument(
        "--block_shape",
        choices=["square", "circle"],
        default="square",
        help="BEV block geometry in GIF/MP4 (circle matches envs/push_boundary.py CIRCLE_RADIUS)",
    )
    # MCTD-specific overrides (used only with --mode mctd)
    parser.add_argument(
        "--max_search_num",
        type=int,
        default=None,
        help="MCTD max search iterations (default: from config)",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=None,
        help="MCTD denoising steps per tree level (default: from config)",
    )
    parser.add_argument(
        "--mctd_guidance_scales",
        type=str,
        default=None,
        help="MCTD-only: comma-separated guidance scales for tree actions (e.g. 0,1,2,5,10)",
    )
    parser.add_argument(
        "--warp_threshold",
        type=float,
        default=0.04,
        help=(
            "MCTD-only: warp detection threshold in unnormalized obs-space units "
            "(default: 0.04)."
        ),
    )
    parser.add_argument(
        "--goal_threshold",
        type=float,
        default=0.05,
        help=(
            "MCTD-only: goal-achievement threshold in unnormalized obs-space units "
            "(default: 0.05)."
        ),
    )
    return parser.parse_args()


def _parse_guidance_scales(s: str | None) -> list[float] | None:
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    out: list[float] = []
    seen: set[float] = set()
    for p in parts:
        v = float(p)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _guidance_scale_dirname(g: float) -> str:
    """Filesystem-safe tag for a guidance scale (used in scale_<tag> directories)."""
    s = f"{g:.6g}"
    return s.replace(".", "p").replace("-", "m")


def _build_hydra_overrides(args) -> list[str]:
    overrides = [
        "experiment=exp_planning",
        "dataset=pushblock_offline",
        "algorithm=df_planning",
        "+name=pushblock_inference",
        "wandb.mode=disabled",
        "algorithm.no_sim_env=true",
        f"algorithm.viz_block_shape={args.block_shape}",
        f"algorithm.frame_stack={_FRAME_STACK}",
    ]

    if args.mode == "mctd":
        overrides.append("algorithm.mctd=true")
        if args.max_search_num is not None:
            overrides.append(f"algorithm.mctd_max_search_num={args.max_search_num}")
        if args.num_denoising_steps is not None:
            overrides.append(
                f"algorithm.mctd_num_denoising_steps={args.num_denoising_steps}"
            )
        if (
            args.mctd_guidance_scales is not None
            and str(args.mctd_guidance_scales).strip()
        ):
            overrides.append(
                f"algorithm.mctd_guidance_scales=[{args.mctd_guidance_scales}]"
            )
        overrides.append(f"algorithm.warp_threshold={args.warp_threshold}")
        overrides.append(f"algorithm.goal_threshold={args.goal_threshold}")

    return overrides


def load_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_path = str(_MCTD_ROOT / "configurations")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config", overrides=_build_hydra_overrides(args))
    with open_dict(cfg):
        cfg.experiment._name = "exp_planning"
        cfg.dataset._name = "pushblock_offline"
        cfg.algorithm._name = "df_planning"
    return cfg


def resolve_checkpoint(checkpoint_arg: str) -> Path:
    from utils.ckpt_utils import download_latest_checkpoint, is_run_id

    path = Path(checkpoint_arg)
    print(path)
    if path.exists():
        return path
    if is_run_id(checkpoint_arg):
        # Minimal config for wandb path resolution
        class _Args:
            mode = "guided"
            block_shape = "square"
            max_search_num = None
            num_denoising_steps = None
            mctd_guidance_scales = None
            warp_threshold = 0.04
            goal_threshold = 0.05

        cfg = load_config(_Args())
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{checkpoint_arg}"
        return download_latest_checkpoint(run_path, Path("outputs/downloaded"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def build_algo_and_dataset(args):
    from experiments import build_experiment

    cfg = load_config(args)
    # print(cfg)
    experiment = build_experiment(cfg, logger=None, ckpt_path=None)
    algo = experiment._build_algo()
    # dataset = experiment._build_dataset("validation")
    dataset = None  # FIXME
    return algo, dataset, cfg


def load_checkpoint(algo, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        algo.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        algo.load_state_dict(ckpt, strict=True)
    algo.eval()
    algo.to(device)
    return algo


def run_unguided(
    algo,
    dataset,
    num_samples: int,
    output_dir: Path,
    device: torch.device,
    output_format: str = "gif",
    mode_dir: Path | None = None,
    block_shape: str = "square",
    horizon: int | None = None,
    yaw_encoding: str = "cos_sin",
):
    """
    Unguided rollout via ``algo.plan(..., guidance_scale=0)``.

    For each batch, the first observation of each trajectory (from the shuffled
    validation loader) is used as the start; ``goal`` is a dummy (cloned start)
    and is ignored because guidance is disabled. Length is ``H + 1`` frames
    (start plus ``H`` planned steps), where ``H`` is ``--horizon`` if set, else
    ``algo.episode_len``.

    If ``mode_dir`` is None, writes under ``output_dir / "unguided"``.
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=min(num_samples, 4), shuffle=True, num_workers=0
    )
    if mode_dir is None:
        mode_dir = output_dir / "unguided"
    _mkdir(mode_dir)

    algo.eval()
    H = horizon if horizon is not None else algo.episode_len
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)
    obs_mean_t = torch.from_numpy(obs_mean).to(device).float()
    obs_std_t = torch.from_numpy(obs_std).to(device).float()
    model_device = next(algo.parameters()).device

    trajectories = []
    sample_idx = 0
    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)
    if block_yaw_indices is not None and yaw_encoding == "sin_cos":
        block_yaw_indices = (block_yaw_indices[1], block_yaw_indices[0])

    for batch_idx, batch in enumerate(dataloader):
        if sample_idx >= num_samples:
            break
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        observations = batch[0][..., : algo.observation_dim].float()
        batch_size = observations.shape[0]
        start_norm = (observations[:, 0] - obs_mean_t) / obs_std_t
        goal_norm = start_norm.clone()
        start_norm = start_norm.to(model_device).contiguous()
        goal_norm = goal_norm.to(model_device).contiguous()

        with torch.no_grad():
            plan_hist = algo.plan(start_norm, goal_norm, H)
        plan_final = plan_hist[-1]

        for i in range(batch_size):
            if sample_idx >= num_samples:
                break
            pf = plan_final[:, i : i + 1, :]
            plan_unnorm = algo._unnormalize_x(pf)
            obs_unnorm, _, _ = algo.split_bundle(plan_unnorm)
            start_unnorm = (start_norm[i] * obs_std_t + obs_mean_t).unsqueeze(0)
            start_obs = start_unnorm[:, : algo.observation_dim].unsqueeze(1)
            full_traj = torch.cat(
                [start_obs, obs_unnorm[:, : algo.observation_dim]],
                0,
            )
            states = full_traj.detach().cpu().numpy().astype(np.float32)
            trajectories.append(states)
            states_2d = states.squeeze(1)
            np.savez(
                mode_dir / f"trajectory_{sample_idx}.npz",
                states=states_2d.astype(np.float32),
                source="diffuser_inference",
                version=np.int32(1),
                mode="unguided",
                horizon=np.int32(H),
            )
            viz_dir = (
                mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"
            )
            algo._log_or_save_pushboundary_2d_gif(
                namespace="unguided",
                states=states_2d,
                sample_idx=sample_idx,
                tcp_xy_indices=tcp_xy_indices,
                block_xy_indices=block_xy_indices,
                block_yaw_indices=block_yaw_indices,
                gif_out_dir=viz_dir,
                output_format=output_format,
                block_shape=block_shape,
            )
            sample_idx += 1

    return trajectories


def _expand_obs_shorthand(values: list[float], obs_mean: np.ndarray) -> np.ndarray:
    """
    Expand a user-provided observation specification to the full obs_dim.

    - If len(values) == obs_dim: used directly.
    - If len(values) == 4: treated as (tcp_x, tcp_y, block_x, block_y), mapped to
      dims 0,1,2,3; all other dims (including yaw encoding) filled from obs_mean.
    - Otherwise: raises ValueError.
    """
    obs_dim = len(obs_mean)
    if len(values) == obs_dim:
        return np.array(values, dtype=np.float32)
    if len(values) == 4:
        obs = obs_mean.copy()
        for dst, src in zip(_SHORTHAND_4D_MAP, values):
            obs[dst] = src
        return obs.astype(np.float32)
    raise ValueError(
        f"--start/--goal must have 4 values (tcp_x,tcp_y,block_x,block_y shorthand) "
        f"or {obs_dim} values (full observation); got {len(values)}."
    )


def _normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((obs - mean) / std).float()


def _infer_viz_indices(
    algo,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    """
    Determine indices for TCP XY, block XY, and (optionally) a 2D yaw encoding in the
    current observation representation.

    For the circle 2D offline dataset, observations are 4D:
        [tcp_x, tcp_y, block_x, block_y]
    because the dataset slices the original 18D pushblock state via
    state_indices [0, 1, 9, 10].
    """
    obs_dim = int(getattr(algo, "observation_dim", 0))
    if obs_dim == 4:
        # No yaw channels exist in the 4D representation.
        return (0, 1), (2, 3), None
    if obs_dim == 6:
        return _TCP_XY_IDX, _BLOCK_XY_IDX, _BLOCK_YAW2_IDX
    # Fallback: treat unknown layouts as no-yaw for visualization.
    return (0, 1), (2, 3), None


def _block_xy_from_obs_unnorm(obs_unnorm_1d: torch.Tensor, algo) -> np.ndarray:
    """Extract unnormalised block XY from a 1D observation tensor."""
    obs_dim = int(getattr(algo, "observation_dim", 0))
    if obs_dim == 4:
        xy = obs_unnorm_1d[[2, 3]]
    elif obs_dim == 6:
        xy = obs_unnorm_1d[[2, 3]]
    else:
        xy = obs_unnorm_1d[[2, 3]]
    return xy.detach().cpu().numpy().astype(np.float32)


def _get_start_goal_from_dataset(dataset, num_samples, algo, device):
    from torch.utils.data import DataLoader

    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs, *_ = batch
    obs = obs[:, :, : algo.observation_dim]
    start = obs[:, 0].float().to(device)
    goal = obs[:, -1].float().to(device)
    mean_t = torch.from_numpy(obs_mean).to(device)
    std_t = torch.from_numpy(obs_std).to(device)
    start_norm = ((start - mean_t) / std_t).float()
    goal_norm = ((goal - mean_t) / std_t).float()
    return start_norm, goal_norm


def _get_start_goal_from_dataset_with_unnorm(
    dataset, num_samples: int, algo, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Sample start/goal and return both normalized tensors and unnormalized numpy arrays."""
    from torch.utils.data import DataLoader

    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs, *_ = batch
    obs = obs[:, :, : algo.observation_dim]

    start_unnorm = obs[:, 0].float().cpu().numpy().astype(np.float32)
    goal_unnorm = obs[:, -1].float().cpu().numpy().astype(np.float32)

    start_norm = _normalize_obs(start_unnorm, obs_mean, obs_std).to(device)
    goal_norm = _normalize_obs(goal_unnorm, obs_mean, obs_std).to(device)

    return start_norm, goal_norm, start_unnorm, goal_unnorm


def run_mctd(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
):
    """Run MCTD planning (algo.p_mctd_plan) for each start/goal pair."""
    assert dataset is not None, "MCTD mode requires a dataset for sampling start/goal."
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_vals = [float(x) for x in args.start.split(",")]
        goal_vals = [float(x) for x in args.goal.split(",")]
        start_obs = _expand_obs_shorthand(start_vals, obs_mean)[None]
        goal_obs = _expand_obs_shorthand(goal_vals, obs_mean)[None]
        start_unnorm = np.repeat(start_obs.astype(np.float32), args.num_samples, axis=0)
        goal_unnorm = np.repeat(goal_obs.astype(np.float32), args.num_samples, axis=0)
        start_norm = _normalize_obs(start_unnorm, obs_mean, obs_std).to(device)
        goal_norm = _normalize_obs(goal_unnorm, obs_mean, obs_std).to(device)
    else:
        start_norm, goal_norm, start_unnorm, goal_unnorm = (
            _get_start_goal_from_dataset_with_unnorm(
                dataset, args.num_samples, algo, device
            )
        )

    horizon = args.horizon if args.horizon is not None else algo.episode_len

    _mkdir(output_dir)
    viz_dir = output_dir / ("gifs" if args.format == "gif" else "videos")

    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)
    if block_yaw_indices is not None and args.yaw_encoding == "sin_cos":
        block_yaw_indices = (block_yaw_indices[1], block_yaw_indices[0])

    model_device = next(algo.parameters()).device

    for i in range(args.num_samples):
        print(f"[{i+1}/{args.num_samples}] Running MCTD search...")
        start_i = start_norm[i : i + 1].to(model_device).float().contiguous()
        goal_i = goal_norm[i : i + 1].to(model_device).float().contiguous()
        start_unnorm_i = start_unnorm[i : i + 1]
        goal_unnorm_i = goal_unnorm[i : i + 1]

        with torch.no_grad():
            plan_hist = algo.p_mctd_plan(
                start_i,
                goal_i,
                horizon,
                None,  # conditions
                start_unnorm_i,
                goal_unnorm_i,
            )

        # Unnormalize and take final plan
        plan_hist = algo._unnormalize_x(plan_hist)
        plan_final = plan_hist[-1]  # (t, 1, bundle_dim)
        obs_traj, _, _ = algo.split_bundle(plan_final)  # (t, 1, obs_dim)
        states = obs_traj[:, 0, :].detach().cpu().numpy().astype(np.float32)

        start_marker = start_unnorm_i[0, list(block_xy_indices)].astype(np.float32)
        goal_marker = goal_unnorm_i[0, list(block_xy_indices)].astype(np.float32)

        out_path = output_dir / f"plan_{i}.npz"
        np.savez(
            out_path,
            states=states.astype(np.float32),
            target_xy=goal_marker.astype(np.float32),
            source="mctd_inference",
            version=np.int32(1),
            mode="mctd",
            horizon=np.int32(horizon),
            seed=np.int32(args.seed),
        )
        print(f"  Saved plan to {out_path}")

        algo._log_or_save_pushboundary_2d_gif(
            namespace="mctd",
            states=states,
            sample_idx=i,
            tcp_xy_indices=tcp_xy_indices,
            block_xy_indices=block_xy_indices,
            block_yaw_indices=block_yaw_indices,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=args.format,
            block_shape=args.block_shape,
        )

    print(f"\nAll outputs saved under {output_dir}")


def run_guided(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
    output_format: str = "gif",
    *,
    guidance_scale: float | None = None,
    mode_dir: Path | None = None,
):
    """
    Goal-conditioned planning. If ``mode_dir`` is None, writes under ``output_dir / "guided"``.
    If ``guidance_scale`` is None, uses ``args.guidance_scale`` or the checkpoint default.
    """
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_vals = [float(x) for x in args.start.split(",")]
        goal_vals = [float(x) for x in args.goal.split(",")]
        start_obs = _expand_obs_shorthand(start_vals, obs_mean)
        goal_obs = _expand_obs_shorthand(goal_vals, obs_mean)
        start_norm = _normalize_obs(start_obs[None], obs_mean, obs_std).to(device)
        goal_norm = _normalize_obs(goal_obs[None], obs_mean, obs_std).to(device)
        start_norm = start_norm.repeat(args.num_samples, 1)
        goal_norm = goal_norm.repeat(args.num_samples, 1)
    else:
        assert dataset is not None
        start_norm, goal_norm = _get_start_goal_from_dataset(
            dataset, args.num_samples, algo, device
        )

    horizon = args.horizon if args.horizon is not None else algo.episode_len
    if guidance_scale is None:
        guidance_scale = (
            args.guidance_scale
            if args.guidance_scale is not None
            else algo.guidance_scale
        )

    if mode_dir is None:
        mode_dir = output_dir / "guided"
    _mkdir(mode_dir)
    viz_dir = mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"

    model_device = next(algo.parameters()).device
    trajectories = []
    batch_size = start_norm.shape[0]
    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)
    if block_yaw_indices is not None and args.yaw_encoding == "sin_cos":
        block_yaw_indices = (block_yaw_indices[1], block_yaw_indices[0])
    for i in range(batch_size):
        start_i = start_norm[i : i + 1].to(model_device).float().contiguous()
        goal_i = goal_norm[i : i + 1].to(model_device).float().contiguous()
        with torch.no_grad():
            plan_hist = algo.plan(
                start_i, goal_i, horizon, guidance_scale=guidance_scale
            )
        plan_final = plan_hist[-1]
        plan_unnorm = algo._unnormalize_x(plan_final)
        obs_unnorm, _, _ = algo.split_bundle(plan_unnorm)
        start_unnorm = (
            start_norm[i] * torch.from_numpy(obs_std).to(device)
            + torch.from_numpy(obs_mean).to(device)
        ).unsqueeze(0)
        start_obs = start_unnorm[:, : algo.observation_dim].unsqueeze(1)
        full_traj = torch.cat(
            [start_obs, obs_unnorm[:, : algo.observation_dim]],
            0,
        )
        states = full_traj.detach().cpu().numpy().astype(np.float32)
        trajectories.append(states)
        states_2d = states.squeeze(1)
        goal_unnorm = goal_norm[i] * torch.from_numpy(obs_std).to(
            device
        ) + torch.from_numpy(obs_mean).to(device)
        start_marker = _block_xy_from_obs_unnorm(start_unnorm[0], algo)
        goal_marker = _block_xy_from_obs_unnorm(goal_unnorm, algo)
        np.savez(
            mode_dir / f"trajectory_{i}.npz",
            states=states_2d.astype(np.float32),
            target_xy=goal_marker.astype(np.float32),
            source="diffuser_inference",
            version=np.int32(1),
            mode="guided",
            guidance_scale=np.float32(guidance_scale),
            horizon=np.int32(horizon),
        )
        algo._log_or_save_pushboundary_2d_gif(
            namespace="guided",
            states=states_2d,
            sample_idx=i,
            tcp_xy_indices=tcp_xy_indices,
            block_xy_indices=block_xy_indices,
            block_yaw_indices=block_yaw_indices,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=output_format,
            block_shape=args.block_shape,
        )
    return trajectories


def main():
    global _FRAME_STACK
    args = parse_args()
    _FRAME_STACK = args.frame_stack
    if (
        args.guidance_scales is not None
        and str(args.guidance_scales).strip()
        and args.mode != "guided"
    ):
        sys.stderr.write("--guidance_scales is only supported with --mode guided.\n")
        sys.exit(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")

    algo, dataset, _ = build_algo_and_dataset(args)
    algo = load_checkpoint(algo, ckpt_path, device)

    output_dir = Path(args.output_dir)
    _mkdir(output_dir)

    if args.mode == "unguided":
        run_unguided(
            algo,
            dataset,
            args.num_samples,
            output_dir,
            device,
            args.format,
            block_shape=args.block_shape,
            horizon=args.horizon,
            yaw_encoding=args.yaw_encoding,
        )
    elif args.mode == "guided":
        scale_list = _parse_guidance_scales(args.guidance_scales)
        if scale_list is None:
            run_guided(algo, dataset, args, output_dir, device, args.format)
        else:
            guided_root = output_dir / "guided"
            _mkdir(guided_root)
            sweep_runs: list[dict] = []
            for g in scale_list:
                if not args.no_reset_seed_between_guidance_scales:
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                tag = _guidance_scale_dirname(g)
                subdir = guided_root / f"scale_{tag}"
                trajs = run_guided(
                    algo,
                    dataset,
                    args,
                    output_dir,
                    device,
                    args.format,
                    guidance_scale=g,
                    mode_dir=subdir,
                )
                sweep_runs.append(
                    {
                        "guidance_scale": g,
                        "relative_dir": str(subdir.relative_to(output_dir)),
                        "num_trajectories": len(trajs),
                        "trajectory_npy_glob": "trajectory_*.npy",
                    }
                )
            manifest = {
                "mode": "guided_guidance_sweep",
                "seed": args.seed,
                "reset_seed_between_guidance_scales": not args.no_reset_seed_between_guidance_scales,
                "num_samples_requested": args.num_samples,
                "runs": sweep_runs,
            }
            manifest_path = guided_root / "guidance_sweep_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"Guidance sweep: wrote {len(scale_list)} runs under {guided_root}/scale_<tag>/ "
                f"and {manifest_path.name}"
            )
    elif args.mode == "mctd":
        run_mctd(algo, dataset, args, output_dir / "mctd", device)

    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
