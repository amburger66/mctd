"""
Offline RL dataset wrapper for the custom PushBoundary trajectories.

This dataset exposes the tuple format expected by:
  `submodules/mctd/algorithms/diffusion_forcing/df_planning.py`

Return values (for each item):
  observation:  [n_frames, observation_dim]  float32
  action:       [n_frames, action_dim]       float32
  reward:       [n_frames]                  float32
  nonterminal:  [n_frames]                   bool
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def _resolve_repo_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p

    # File: <repo_root>/submodules/mctd/datasets/offline_rl/<this_file>.py
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / p


def _nonterminal_from_valid_length(L_eff: int, n_frames: int) -> torch.Tensor:
    """
    L_eff: number of valid timesteps in this trajectory (<= n_frames).
    Padding uses zeros for t >= L_eff; mask those out.

    Matches full-length convention: nonterminal True until last valid step, False at terminal.
    """
    nonterminal = torch.zeros(n_frames, dtype=torch.bool)
    if L_eff <= 0:
        return nonterminal
    L_eff = min(L_eff, n_frames)
    if L_eff > 1:
        nonterminal[: L_eff - 1] = True
    nonterminal[L_eff - 1] = False
    return nonterminal


class PushBoundaryOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        _ = split  # PushBoundary dataset is already offline; we ignore split.
        self.cfg = cfg

        self.episode_len = int(cfg.episode_len)
        self.n_frames = self.episode_len + 1

        # Shapes / dims.
        # Expected yaml fields:
        # - observation_shape: [obs_dim]
        # - action_dim: action_dim
        # Optional (for 2D slicing): state_indices, action_indices
        self.observation_dim = int(cfg.observation_shape[0])
        self.action_dim = int(cfg.action_dim)

        self._state_indices = (
            list(cfg.state_indices) if "state_indices" in cfg else None
        )
        self._action_indices = (
            list(cfg.action_indices) if "action_indices" in cfg else None
        )

        if self._state_indices is not None:
            if len(self._state_indices) != self.observation_dim:
                raise ValueError(
                    f"state_indices length {len(self._state_indices)} != observation_shape[0] {self.observation_dim}"
                )
        if self._action_indices is not None:
            if len(self._action_indices) != self.action_dim:
                raise ValueError(
                    f"action_indices length {len(self._action_indices)} != action_dim {self.action_dim}"
                )

        npz_path = cfg.get("npz_path", None)
        if npz_path:
            self._load_npz_bundle(cfg, npz_path)
        else:
            self._load_npy_pair(cfg)

        # Reward is not used by `df_planning.yaml` when `use_reward=False`, but the
        # dataset contract requires it.
        self._reward_template = torch.zeros(self.n_frames, dtype=torch.float32)

        if not self._use_valid_lengths:
            nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
            nonterminal[-1] = False
            self._nonterminal_template = nonterminal
        else:
            self._nonterminal_template = None

        # Basic sanity prints (only on rank zero in lightning; dataset isn't aware of rank).
        # Keep them lightweight.
        slice_info = ""
        if self._state_indices is not None or self._action_indices is not None:
            slice_info = f" (sliced to obs={self.observation_dim}D, act={self.action_dim}D)"
        vl_info = " valid_lengths" if self._use_valid_lengths else ""
        print(
            f"[PushBoundaryOfflineRLDataset] loaded states={self.states.shape} actions={self.actions.shape} "
            f"n_frames={self.n_frames}{slice_info}{vl_info}"
        )

    def _load_npz_bundle(self, cfg: DictConfig, npz_path: str) -> None:
        path = _resolve_repo_path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"npz not found: {path}")

        sk = cfg.get("npz_states_key", "states")
        ak = cfg.get("npz_actions_key", "actions")
        vk = cfg.get("npz_valid_lengths_key", "valid_lengths")

        with np.load(path, allow_pickle=False) as data:
            if sk not in data.files:
                raise KeyError(f"npz missing key {sk!r}, have {data.files}")
            if ak not in data.files:
                raise KeyError(f"npz missing key {ak!r}, have {data.files}")
            if vk not in data.files:
                raise KeyError(f"npz missing key {vk!r}, have {data.files}")
            states = np.asarray(data[sk])
            actions = np.asarray(data[ak])
            valid_lengths = np.asarray(data[vk]).astype(np.int64, copy=False)

        if states.ndim != 3:
            raise ValueError(f"states must be 3D [N, T, SD], got shape={states.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be 3D [N, T, AD], got shape={actions.shape}")
        if valid_lengths.ndim != 1:
            raise ValueError(f"valid_lengths must be 1D [N], got shape={valid_lengths.shape}")

        n_traj, t_total, sd = states.shape
        n_traj_a, t_total_a, ad = actions.shape
        if n_traj != n_traj_a or t_total != t_total_a:
            raise ValueError(
                f"states/actions shape mismatch: states={states.shape} actions={actions.shape}"
            )
        if valid_lengths.shape[0] != n_traj:
            raise ValueError(
                f"valid_lengths length {valid_lengths.shape[0]} != n_traj {n_traj}"
            )

        self._validate_dims(sd, ad)
        if t_total < self.n_frames:
            raise ValueError(
                f"Trajectory length too short: states/actions have T_total={t_total} "
                f"but require n_frames={self.n_frames}."
            )
        if np.any(valid_lengths < 0) or np.any(valid_lengths > t_total):
            raise ValueError(
                f"valid_lengths must be in [0, T_total], T_total={t_total}, "
                f"min={valid_lengths.min()} max={valid_lengths.max()}"
            )

        self.states = states
        self.actions = actions
        self._valid_lengths = valid_lengths
        self.n_traj = n_traj
        self._use_valid_lengths = True

    def _load_npy_pair(self, cfg: DictConfig) -> None:
        states_path = _resolve_repo_path(cfg.states_path)
        actions_path = _resolve_repo_path(cfg.actions_path)

        # Use mmap so we don't eagerly load the full arrays into each worker.
        states = np.load(str(states_path), mmap_mode="r")
        actions = np.load(str(actions_path), mmap_mode="r")

        if states.ndim != 3:
            raise ValueError(f"states must be 3D [N, T, SD], got shape={states.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be 3D [N, T, AD], got shape={actions.shape}")

        n_traj, t_total, sd = states.shape
        n_traj_a, t_total_a, ad = actions.shape
        if n_traj != n_traj_a or t_total != t_total_a:
            raise ValueError(
                f"states/actions shape mismatch: states={states.shape} actions={actions.shape}"
            )

        self._validate_dims(sd, ad)

        if t_total < self.n_frames:
            raise ValueError(
                f"Trajectory length too short: states/actions have T_total={t_total} "
                f"but require n_frames={self.n_frames}."
            )

        self.states = states
        self.actions = actions
        self.n_traj = n_traj
        self._valid_lengths = None
        self._use_valid_lengths = False

    def _validate_dims(self, sd: int, ad: int) -> None:
        if self._state_indices is not None:
            if sd <= max(self._state_indices):
                raise ValueError(
                    f"states last dim {sd} too small for state_indices {self._state_indices}"
                )
        else:
            if sd != self.observation_dim:
                raise ValueError(
                    f"states last dim mismatch: got {sd}, expected {self.observation_dim}"
                )

        if self._action_indices is not None:
            if ad <= max(self._action_indices):
                raise ValueError(
                    f"actions last dim {ad} too small for action_indices {self._action_indices}"
                )
        else:
            if ad != self.action_dim:
                raise ValueError(
                    f"actions last dim mismatch: got {ad}, expected {self.action_dim}"
                )

    def __len__(self) -> int:
        return self.n_traj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_np = np.asarray(self.states[idx, : self.n_frames], dtype=np.float32)  # [n_frames, SD]
        act_np = np.asarray(self.actions[idx, : self.n_frames], dtype=np.float32)  # [n_frames, AD]

        if self._state_indices is not None:
            obs_np = obs_np[:, self._state_indices]
        if self._action_indices is not None:
            act_np = act_np[:, self._action_indices]

        # Convert to torch tensors. reward/nonterminal are templates.
        observation = torch.from_numpy(obs_np.copy())  # float32
        action = torch.from_numpy(act_np.copy())  # float32
        reward = self._reward_template

        if self._use_valid_lengths:
            assert self._valid_lengths is not None
            L_eff = int(self._valid_lengths[idx])
            nonterminal = _nonterminal_from_valid_length(L_eff, self.n_frames)
        else:
            nonterminal = self._nonterminal_template  # type: ignore[assignment]

        return observation, action, reward, nonterminal
