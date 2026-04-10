#!/usr/bin/env python3
"""
MCTD inference script for the circle_2d model.

Loads a trained checkpoint, runs p_mctd_plan() for each start/goal pair,
saves the resulting plan as a .npz file and renders it as a GIF or MP4.
No environment execution is performed (placeholder for ManiSkill hookup).

Usage (run from submodules/mctd/):
    python scripts/inference_circle_2d_mctd.py --checkpoint path/to/model.ckpt
    python scripts/inference_circle_2d_mctd.py --checkpoint path/to/model.ckpt \\
        --start -0.1,0.0,-0.1,0.0 --goal 0.05,0.1,0.05,0.1 --num_samples 1
    python scripts/inference_circle_2d_mctd.py --checkpoint path/to/model.ckpt \\
        --num_samples 4 --max_search_num 200 --format mp4 --block_shape circle
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
    parser = argparse.ArgumentParser(description="Circle2D MCTD inference")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .ckpt file or wandb run id (8 chars)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of start/goal pairs to plan for (default: 1)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start state as x,y,bx,by (all 4 obs dims, unnormalized). "
            "E.g. --start='-0.1,0.0,-0.1,0.0'. Sampled from dataset if omitted."
        ),
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help=(
            "Goal state as x,y,bx,by (all 4 obs dims, unnormalized). "
            "Sampled from dataset if omitted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference_results/circle_2d_mctd"),
        help="Output directory for raw .npz plans and GIF/MP4 visualizations",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Planning horizon in env steps (default: episode_len from config)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format for trajectory visualizations (default: gif)",
    )
    parser.add_argument(
        "--block_shape",
        choices=["square", "circle"],
        default="circle",
        help="Block geometry in BEV visualization (default: circle)",
    )
    # MCTD-specific overrides
    parser.add_argument(
        "--max_search_num",
        type=int,
        default=None,
        help="MCTD max search iterations (default: from config, typically 500)",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=None,
        help="MCTD denoising steps per tree level (default: from config, typically 20)",
    )
    parser.add_argument(
        "--guidance_scales",
        type=str,
        default="0,1,2,5,10",
        help=("Comma-separated guidance scales for MCTD tree actions"),
    )
    parser.add_argument(
        "--warp_threshold",
        type=float,
        default=0.04,
        help=(
            "Warp detection threshold in unnormalized obs-space units. "
            "A plan frame is flagged as a teleportation artifact if its 4-D Euclidean "
            "distance from the previous frame exceeds this value. (default: 0.04)"
        ),
    )
    parser.add_argument(
        "--goal_threshold",
        type=float,
        default=0.05,
        help=(
            "Goal-achievement distance threshold in unnormalized obs-space units. "
            "(default: 0.05)"
        ),
    )
    return parser.parse_args()


def _build_hydra_overrides(args) -> list[str]:
    overrides = [
        "experiment=exp_planning",
        "dataset=circle_2d_offline",
        "algorithm=df_planning",
        "+name=Circle2D_MCTD",
        "wandb.mode=disabled",
        "algorithm.mctd=true",
        f"algorithm.viz_block_shape={args.block_shape}",
        "algorithm.no_sim_env=true",
        "algorithm.frame_stack=10",
    ]
    if args.max_search_num is not None:
        overrides.append(f"algorithm.mctd_max_search_num={args.max_search_num}")
    if args.num_denoising_steps is not None:
        overrides.append(
            f"algorithm.mctd_num_denoising_steps={args.num_denoising_steps}"
        )
    if args.guidance_scales is not None:
        scales = "[" + args.guidance_scales + "]"
        overrides.append(f"algorithm.mctd_guidance_scales={scales}")
    overrides.append(f"algorithm.warp_threshold={args.warp_threshold}")
    overrides.append(f"algorithm.goal_threshold={args.goal_threshold}")
    return overrides


def load_config(overrides: list[str]):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_path = str(_MCTD_ROOT / "configurations")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config", overrides=overrides)
    with open_dict(cfg):
        cfg.experiment._name = "exp_planning"
        cfg.dataset._name = "circle_2d_offline"
        cfg.algorithm._name = "df_planning"
    return cfg


def build_algo_and_dataset(overrides: list[str]):
    from experiments import build_experiment

    cfg = load_config(overrides)
    experiment = build_experiment(cfg, logger=None, ckpt_path=None)
    algo = experiment._build_algo()
    dataset = experiment._build_dataset("validation")
    return algo, dataset, cfg


def load_checkpoint(algo, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    algo.load_state_dict(state_dict, strict=True)
    algo.eval()
    algo.to(device)
    return algo


def resolve_checkpoint(checkpoint_arg: str) -> Path:
    from utils.ckpt_utils import download_latest_checkpoint, is_run_id

    path = Path(checkpoint_arg)
    if path.exists():
        return path
    if is_run_id(checkpoint_arg):
        from hydra import compose, initialize_config_dir

        config_path = str(_MCTD_ROOT / "configurations")
        with initialize_config_dir(version_base=None, config_dir=config_path):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=exp_planning",
                    "dataset=circle_2d_offline",
                    "algorithm=df_planning",
                    "wandb.mode=disabled",
                ],
            )
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{checkpoint_arg}"
        return download_latest_checkpoint(run_path, Path("outputs/downloaded"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def _normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((obs - mean) / std).float()


def _get_start_goal_from_dataset(
    dataset, num_samples: int, algo, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Sample start/goal pairs from the dataset.

    Returns normalized tensors (for planning) and unnormalized numpy arrays
    (for calculate_values and visualization markers).
    """
    from torch.utils.data import DataLoader

    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs = batch[0]  # (batch, time, obs_dim) or similar layout from dataset
    obs = obs[:, :, : algo.observation_dim]

    start_unnorm = obs[:, 0].float().numpy().astype(np.float32)  # (N, obs_dim)
    goal_unnorm = obs[:, -1].float().numpy().astype(np.float32)  # (N, obs_dim)

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
    """Run p_mctd_plan for each sample, save raw .npz and render GIF/MP4."""
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    # Resolve start/goal from CLI args or dataset
    if args.start is not None and args.goal is not None:
        start_unnorm = np.array(
            [[float(x) for x in args.start.split(",")]], dtype=np.float32
        )  # (1, obs_dim)
        goal_unnorm = np.array(
            [[float(x) for x in args.goal.split(",")]], dtype=np.float32
        )
        start_unnorm = np.repeat(start_unnorm, args.num_samples, axis=0)
        goal_unnorm = np.repeat(goal_unnorm, args.num_samples, axis=0)
        start_norm = _normalize_obs(start_unnorm, obs_mean, obs_std).to(device)
        goal_norm = _normalize_obs(goal_unnorm, obs_mean, obs_std).to(device)
    else:
        start_norm, goal_norm, start_unnorm, goal_unnorm = _get_start_goal_from_dataset(
            dataset, args.num_samples, algo, device
        )

    horizon = args.horizon if args.horizon is not None else algo.episode_len

    viz_dir = output_dir / ("gifs" if args.format == "gif" else "videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_samples):
        print(f"[{i+1}/{args.num_samples}] Running MCTD search...")
        start_i = start_norm[i : i + 1].float().contiguous()  # (1, obs_dim)
        goal_i = goal_norm[i : i + 1].float().contiguous()
        start_unnorm_i = start_unnorm[i : i + 1]  # (1, obs_dim)
        goal_unnorm_i = goal_unnorm[i : i + 1]

        with torch.no_grad():
            plan_hist = algo.p_mctd_plan(
                start_i,
                goal_i,
                horizon,
                None,  # conditions
                start_unnorm_i,  # unnormalized, for calculate_values
                goal_unnorm_i,
            )

        # plan_hist is already the final plan tensor (t, 1, bundle_dim), unnormalized
        plan_hist = algo._unnormalize_x(plan_hist)
        plan_final = plan_hist[-1]  # (t, 1, bundle_dim)

        obs_traj, _, _ = algo.split_bundle(plan_final)  # (t, 1, obs_dim)
        states = obs_traj[:, 0, :].detach().cpu().numpy()  # (t, obs_dim)

        # Raw 2D NPZ format compatible with:
        #   python scripts/playback_floating.py --raw_npz <file> --conversion_mode 2d
        #
        # states: (T,4) ordered [tcp_x, tcp_y, block_x, block_y]
        target_xy = goal_unnorm_i[0, 2:4].astype(np.float32)  # block XY
        out_path = output_dir / f"plan_{i}.npz"
        np.savez(
            out_path,
            states=states.astype(np.float32),
            target_xy=target_xy,
            source="mctd_inference",
            version=np.int32(1),
            mode="guided",
            horizon=np.int32(horizon),
            seed=np.int32(args.seed),
        )
        print(f"  Saved plan to {out_path}")

        # Markers: block XY for start/goal visualization
        start_marker = start_unnorm_i[0, 2:4]
        goal_marker = goal_unnorm_i[0, 2:4]

        algo._log_or_save_pushboundary_2d_gif(
            namespace="mctd",
            states=states,
            sample_idx=i,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=args.format,
            block_shape=args.block_shape,
        )

    print(f"\nAll outputs saved under {output_dir}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overrides = _build_hydra_overrides(args)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")

    algo, dataset, _ = build_algo_and_dataset(overrides)
    algo = load_checkpoint(algo, ckpt_path, device)

    output_dir = Path(args.output_dir)
    run_mctd(algo, dataset, args, output_dir, device)


if __name__ == "__main__":
    main()
