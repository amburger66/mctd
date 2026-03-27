#!/usr/bin/env python3
"""
Compute observation/action mean and std from circle_demos_full.npz using only valid timesteps
(not zero padding). Prints YAML-friendly lists for pushboundary_circle_2d_offline.yaml.

Usage (from submodules/mctd):
  python scripts/compute_circle_demo_stats.py --npz ../../demos/circle_demos_full.npz
  python scripts/compute_circle_demo_stats.py --npz ../../demos/circle_demos_full.npz --state-indices 0 1 9 10 --action-indices 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_MCTD_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _MCTD_ROOT.parent.parent
sys.path.insert(0, str(_MCTD_ROOT))


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default="demos/circle_demos_full.npz",
        help="Path to NPZ (repo-relative or absolute)",
    )
    parser.add_argument("--states-key", type=str, default="states")
    parser.add_argument("--actions-key", type=str, default="actions")
    parser.add_argument("--valid-lengths-key", type=str, default="valid_lengths")
    parser.add_argument(
        "--state-indices",
        type=int,
        nargs="*",
        default=None,
        help="If set, slice state to these indices (e.g. 0 1 9 10 for 2D)",
    )
    parser.add_argument(
        "--action-indices",
        type=int,
        nargs="*",
        default=None,
        help="If set, slice actions to these indices",
    )
    args = parser.parse_args()

    path = _resolve_path(args.npz)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with np.load(path, allow_pickle=True) as z:
        states = np.asarray(z[args.states_key], dtype=np.float64)
        actions = np.asarray(z[args.actions_key], dtype=np.float64)
        valid_lengths = np.asarray(z[args.valid_lengths_key], dtype=np.int64)

    if states.ndim != 3 or actions.ndim != 3:
        raise ValueError(
            f"Expected states/actions 3D, got {states.shape}, {actions.shape}"
        )
    n, t_max, sd = states.shape
    if valid_lengths.shape != (n,):
        raise ValueError(f"valid_lengths shape {valid_lengths.shape}, expected ({n},)")

    obs_chunks = []
    act_chunks = []
    for i in range(n):
        L = int(valid_lengths[i])
        if L <= 0 or L > t_max:
            raise ValueError(f"traj {i}: invalid valid_lengths={L}")
        s = states[i, :L].copy()
        a = actions[i, :L].copy()
        if args.state_indices is not None:
            s = s[:, args.state_indices]
        if args.action_indices is not None:
            a = a[:, args.action_indices]
        obs_chunks.append(s.reshape(-1, s.shape[-1]))
        act_chunks.append(a.reshape(-1, a.shape[-1]))

    obs_all = np.concatenate(obs_chunks, axis=0)
    act_all = np.concatenate(act_chunks, axis=0)

    obs_mean = obs_all.mean(axis=0)
    obs_std = obs_all.std(axis=0)
    act_mean = act_all.mean(axis=0)
    act_std = act_all.std(axis=0)

    def fmt(arr: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(x):.8g}" for x in arr.tolist()) + "]"

    print(f"T_max = {t_max}  ->  episode_len = {t_max - 1}  (set in dataset YAML)")
    print(f"n_trajectories = {n}")
    print(f"observation_mean: {fmt(obs_mean)}")
    print(f"observation_std:  {fmt(obs_std)}")
    print(f"action_mean:      {fmt(act_mean)}")
    print(f"action_std:       {fmt(act_std)}")


if __name__ == "__main__":
    main()
