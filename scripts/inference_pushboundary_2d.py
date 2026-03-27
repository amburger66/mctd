#!/usr/bin/env python3
"""
Inference script for PushBoundary2D diffusion model.
Loads a trained checkpoint and runs unguided (conditional completion),
guided (goal-conditioned), or guided_inpaint (last-frame inpainting) inference.

Usage (run from submodules/mctd/):
    python scripts/inference_pushboundary_2d.py --checkpoint path/to/model.ckpt --mode unguided
    python scripts/inference_pushboundary_2d.py --checkpoint path/to/model.ckpt --mode guided --num_samples 4
    python scripts/inference_pushboundary_2d.py --checkpoint path/to/model.ckpt --mode guided --start -0.2,0.0,-0.2,0.0 --goal 0.1,0.1,0.05,0.05
    python scripts/inference_pushboundary_2d.py --checkpoint path/to/model.ckpt --mode guided --format mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_MCTD_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MCTD_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="PushBoundary2D inference")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .ckpt or wandb run id (8 chars)"
    )
    parser.add_argument(
        "--mode",
        choices=["unguided", "guided", "guided_inpaint"],
        default="unguided",
        help="Inference mode",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of trajectories"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference/pushboundary_2d"),
        help="Output directory for visualizations (GIF/MP4) and trajectories",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start state as tcp_x,tcp_y,block_x,block_y (guided mode). Use --start='-0.2,0,0,0' for negative values.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal state as tcp_x,tcp_y,block_x,block_y (guided mode). Use --goal='x,y,z,w' format.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for guided mode (default: from config)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Planning horizon in env steps (default: episode_len)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="mp4",
        help="Output format for trajectory visualizations (default: gif)",
    )
    parser.add_argument(
        "--block-geom",
        choices=["square", "circle"],
        default="square",
        help="Block shape in 2D viz (circle matches envs/push_boundary.py circle block)",
    )
    return parser.parse_args()


def load_config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    config_path = str(_MCTD_ROOT / "configurations")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=exp_planning",
                "dataset=pushboundary_2d_offline",
                "algorithm=df_planning",
                "+name=PushBoundary2D",
                "wandb.mode=disabled",
            ],
        )
    with open_dict(cfg):
        cfg.experiment._name = "exp_planning"
        cfg.dataset._name = "pushboundary_2d_offline"
        cfg.algorithm._name = "df_planning"
    return cfg


def resolve_checkpoint(checkpoint_arg: str) -> Path:
    from utils.ckpt_utils import download_latest_checkpoint, is_run_id

    path = Path(checkpoint_arg)
    if path.exists():
        return path
    if is_run_id(checkpoint_arg):
        cfg = load_config()
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{checkpoint_arg}"
        return download_latest_checkpoint(run_path, Path("outputs/downloaded"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def build_algo_and_dataset():
    from experiments import build_experiment

    cfg = load_config()
    experiment = build_experiment(cfg, logger=None, ckpt_path=None)
    algo = experiment._build_algo()
    dataset = experiment._build_dataset("validation")
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
    output_format: str = "mp4",
    block_geom: str = "square",
):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=min(num_samples, 4), shuffle=True, num_workers=0
    )
    mode_dir = output_dir / "unguided"
    mode_dir.mkdir(parents=True, exist_ok=True)

    algo.eval()
    n_context_frames = algo.context_frames // algo.frame_stack
    trajectories = []
    sample_idx = 0

    for batch_idx, batch in enumerate(dataloader):
        if sample_idx >= num_samples:
            break
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        with torch.no_grad():
            xs, conditions, masks = algo._preprocess_batch(batch)
        n_frames, batch_size, *_ = xs.shape
        xs_pred = xs[:n_context_frames].clone()
        curr_frame = n_context_frames

        while curr_frame < n_frames:
            horizon = (
                min(n_frames - curr_frame, algo.chunk_size)
                if algo.chunk_size > 0
                else n_frames - curr_frame
            )
            scheduling_matrix = algo._generate_scheduling_matrix(horizon)

            chunk = torch.randn(
                (horizon, batch_size, *algo.x_stacked_shape), device=device
            )
            chunk = torch.clamp(chunk, -algo.clip_noise, algo.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            start_frame = max(0, curr_frame + horizon - algo.n_tokens)

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                to_noise = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
                )[:, None].repeat(batch_size, axis=1)
                from_noise = torch.from_numpy(from_noise).to(device)
                to_noise = torch.from_numpy(to_noise).to(device)

                cond = (
                    None
                    if conditions is None
                    else conditions[start_frame : curr_frame + horizon]
                )
                xs_pred[start_frame:] = algo.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    cond,
                    from_noise[start_frame:],
                    to_noise[start_frame:],
                )

            curr_frame += horizon

        xs_pred = algo._unstack_and_unnormalize(xs_pred)
        obs, _, _ = algo.split_bundle(xs_pred)
        obs_np = obs.detach().cpu().numpy()

        for b in range(obs_np.shape[1]):
            if sample_idx >= num_samples:
                break
            states = obs_np[:, b, :].astype(np.float32)
            trajectories.append(states)
            np.save(mode_dir / f"trajectory_{sample_idx}.npy", states)
            viz_dir = (
                mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"
            )
            algo._log_or_save_pushboundary_2d_gif(
                namespace="unguided",
                states=states,
                sample_idx=sample_idx,
                gif_out_dir=viz_dir,
                output_format=output_format,
                block_geom=block_geom,
            )
            sample_idx += 1

    return trajectories


def _normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((obs - mean) / std).float()


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


def run_guided(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
    output_format: str = "mp4",
    block_geom: str = "square",
):
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_norm = _normalize_obs(
            np.array([[float(x) for x in args.start.split(",")]], dtype=np.float32),
            obs_mean,
            obs_std,
        ).to(device)
        goal_norm = _normalize_obs(
            np.array([[float(x) for x in args.goal.split(",")]], dtype=np.float32),
            obs_mean,
            obs_std,
        ).to(device)
        start_norm = start_norm.repeat(args.num_samples, 1)
        goal_norm = goal_norm.repeat(args.num_samples, 1)
    else:
        start_norm, goal_norm = _get_start_goal_from_dataset(
            dataset, args.num_samples, algo, device
        )

    horizon = args.horizon if args.horizon is not None else algo.episode_len
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else algo.guidance_scale
    )

    mode_dir = output_dir / "guided"
    mode_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"

    model_device = next(algo.parameters()).device
    trajectories = []
    batch_size = start_norm.shape[0]
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
        np.save(mode_dir / f"trajectory_{i}.npy", states)
        goal_unnorm = goal_norm[i] * torch.from_numpy(obs_std).to(
            device
        ) + torch.from_numpy(obs_mean).to(device)
        start_marker = start_unnorm[0, 2:4].detach().cpu().numpy().astype(np.float32)
        goal_marker = goal_unnorm[2:4].detach().cpu().numpy().astype(np.float32)
        states_2d = states.squeeze(1)
        algo._log_or_save_pushboundary_2d_gif(
            namespace="guided",
            states=states_2d,
            sample_idx=i,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=output_format,
            block_geom=block_geom,
        )
    return trajectories


def run_guided_inpaint(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
    output_format: str = "mp4",
    block_geom: str = "square",
):
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_norm = _normalize_obs(
            np.array([[float(x) for x in args.start.split(",")]], dtype=np.float32),
            obs_mean,
            obs_std,
        ).to(device)
        goal_norm = _normalize_obs(
            np.array([[float(x) for x in args.goal.split(",")]], dtype=np.float32),
            obs_mean,
            obs_std,
        ).to(device)
        start_norm = start_norm.repeat(args.num_samples, 1)
        goal_norm = goal_norm.repeat(args.num_samples, 1)
    else:
        start_norm, goal_norm = _get_start_goal_from_dataset(
            dataset, args.num_samples, algo, device
        )

    horizon = args.horizon if args.horizon is not None else algo.episode_len

    mode_dir = output_dir / "guided_inpaint"
    mode_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"

    model_device = next(algo.parameters()).device
    trajectories = []
    batch_size = start_norm.shape[0]
    for i in range(batch_size):
        start_i = start_norm[i : i + 1].to(model_device).float().contiguous()
        goal_i = goal_norm[i : i + 1].to(model_device).float().contiguous()
        with torch.no_grad():
            plan_hist = algo.plan_inpaint(start_i, goal_i, horizon)
        plan_final = plan_hist[0]
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
        np.save(mode_dir / f"trajectory_{i}.npy", states)
        goal_unnorm = goal_norm[i] * torch.from_numpy(obs_std).to(
            device
        ) + torch.from_numpy(obs_mean).to(device)
        start_marker = start_unnorm[0, 2:4].detach().cpu().numpy().astype(np.float32)
        goal_marker = goal_unnorm[2:4].detach().cpu().numpy().astype(np.float32)
        states_2d = states.squeeze(1)
        algo._log_or_save_pushboundary_2d_gif(
            namespace="guided_inpaint",
            states=states_2d,
            sample_idx=i,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=output_format,
            block_geom=block_geom,
        )
    return trajectories


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")

    algo, dataset, _ = build_algo_and_dataset()
    algo = load_checkpoint(algo, ckpt_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "unguided":
        run_unguided(
            algo,
            dataset,
            args.num_samples,
            output_dir,
            device,
            args.format,
            block_geom=args.block_geom,
        )
    elif args.mode == "guided":
        run_guided(
            algo,
            dataset,
            args,
            output_dir,
            device,
            args.format,
            block_geom=args.block_geom,
        )
    elif args.mode == "guided_inpaint":
        run_guided_inpaint(
            algo,
            dataset,
            args,
            output_dir,
            device,
            args.format,
            block_geom=args.block_geom,
        )

    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
