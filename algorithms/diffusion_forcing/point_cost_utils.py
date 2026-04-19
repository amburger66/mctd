#!/usr/bin/env python3
"""
scripts/collect_demos_single.py

Scripted face-approach push demo collection for PushBoundary (floating gripper).

Each episode:
  1. Spawns the block at centre with a random yaw.
  2. Spawns the gripper at a collision-free position.
  3. Runs a two-phase policy: APPROACH standoff → PUSH.
  4. Resamples a new face every ~20-40 push steps.

Usage:
    python scripts/collect_demos_single.py --num_episodes 50 --record_dir demos/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm
from scipy.spatial.distance import cdist

import trimesh

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_mesh_points(camera_params, mesh, count=5000):
    """
    Samples points on mesh and classifies them as visible, occluded, or nonvisible.
    """
    points, face_indices = trimesh.sample.sample_surface(mesh, count=count)

    # unordered points
    visible_idx, occluded_idx, nonvisible_idx = get_point_visiblility(points, camera_params, mesh)
    visible_points, occluded_points, nonvisible_points = points[visible_idx], points[occluded_idx], points[nonvisible_idx]
    return visible_points, occluded_points, nonvisible_points


# This was written using Gemini assistance
def is_point_in_frustum(points, camera_params):
    """
    Checks if the point is in the camera frustum and can be seen.
    Returns True for points that can be seen
    """
    # cam_pose_inv = camera.get_global_pose().inv()
    # view_mat = cam_pose_inv.to_transformation_matrix()[0].numpy()
    view_mat = camera_params['view_mat']
    p_world_h = np.column_stack([points, np.ones(len(points))])
    p_cam = (p_world_h @ view_mat.T)[:, :3]

    # 2. Map SAPIEN axes to Image axes
    # SAPIEN: Forward=X, Right=-Y, Up=Z
    # Image projection usually expects: Forward=Z_img, Right=X_img, Up=Y_img
    x_img = -p_cam[:, 1]  # Right is -Y
    y_img = p_cam[:, 2]  # Up is Z
    z_img = p_cam[:, 0]  # Forward is X

    # 3. Extract Intrinsics
    # K = camera.get_intrinsic_matrix()[0].numpy()
    K = camera_params['K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w_img, h_img = camera_params['width'], camera_params['height']

    # 4. Depth Check (Forward is z_img)
    valid_depth = (z_img >= camera_params['near']) & (z_img <= camera_params['far'])

    # 5. Project to Pixels
    u = np.zeros_like(z_img)
    v = np.zeros_like(z_img)
    # Division by z_img (the forward distance)
    u[valid_depth] = (x_img[valid_depth] * fx / z_img[valid_depth]) + cx
    v[valid_depth] = (y_img[valid_depth] * fy / z_img[valid_depth]) + cy
    # 6. Final Visibility
    in_view = valid_depth & (u >= 0) & (u <= w_img) & (v >= 0) & (v <= h_img)
    return in_view


# Written using Gemini assistance
def get_point_visiblility(points, camera_params, meshes):
    """
    Returns unordered points that are visible and occluded

    returns 3 sets of indicies:
        1. visible to camera (in viewing frustum) and not occluded by any meshes)
        2. occluded by meshes
        3. all point not visible (occluded outside viewing frustum)
    """
    in_frustum = is_point_in_frustum(points, camera_params).nonzero()[0]
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(meshes)

    camera_pos = camera_params['cam_pos']
    ray_origins = np.tile(camera_pos, (points.shape[0], 1))
    ray_directions = points - camera_pos

    norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    ray_directions_normed = ray_directions / norms

    locations, index_ray, index_tri = intersector.intersects_location(ray_origins, ray_directions_normed, multiple_hits=False)

    dist_to_hit = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
    dist_to_point = norms[index_ray].flatten()

    occul_idxs = index_ray[(dist_to_hit < dist_to_point - 1e-3)]

    all_idxs = np.arange(points.shape[0])
    nonoccluded_set = np.setdiff1d(all_idxs, occul_idxs)
    visible_set = np.intersect1d(nonoccluded_set, in_frustum)
    nonvisible_set = np.setdiff1d(all_idxs, visible_set)

    return visible_set, occul_idxs, nonvisible_set


def is_close_mesh_points(points, obs_mesh_points, dist_threshold=0.05):
    """Compute Pairwise distance between points and obs_mesh_points 
       and returns boolean if min dist is under threshold
    """
    dists = cdist(points, obs_mesh_points)
    closest_vis_mesh_dist = np.min(dists, axis=1)
    return closest_vis_mesh_dist <= dist_threshold


def get_point_costs(
        points,
        # camera_params,
        # meshes,
        # static_obs_mesh_points,
        occluded_cost=0.1,
        nonvisible_cost=0.2,
        seen_cost=1,
        reconstructed_cost=0.7,
        vis_dist_thresh=0.05,
        obstruct_dist_thresh=0.05,
        visualize=False,
):
    """
    points: points to Check
    camera_params: RenderCamera from sapien
    meshes: Trimesh for object
    static_obs_mesh_points: static mesh points

    Example Usage
    costs = get_point_costs(points, side_camera, obstacle_merged_mesh, static_mesh_points)
    """
    # Test points
    camera_params, meshes, static_obs_mesh_points = setup_side_camera('/data/user_data/mbronars/packages/mctd_updated/algorithms/diffusion_forcing/')
    visible_idx, occluded_idx, nonvisible_idx = get_point_visiblility(points, camera_params, meshes)

    # TODO: (ray) use block mesh as a part of visibility but not obstacle closeness cost
    is_seen = is_close_mesh_points(points, static_obs_mesh_points['visible'], dist_threshold=vis_dist_thresh)
    is_reconstructed = is_close_mesh_points(points, static_obs_mesh_points['nonvisible'], dist_threshold=obstruct_dist_thresh)
    cost = np.zeros(points.shape[0])
    cost[nonvisible_idx] += nonvisible_cost
    cost[occluded_idx] += occluded_cost
    cost += seen_cost * is_seen
    cost += reconstructed_cost * is_reconstructed

    if visualize:
        import matplotlib.pyplot as plt
        cam_pos = camera_params['cam_pos']
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        is_seen_points = points[is_seen]
        is_recon_points = points[is_reconstructed]
        ax.scatter(is_seen_points[:, 0], is_seen_points[:, 1], is_seen_points[:, 2], marker='o')
        ax.scatter(is_recon_points[:, 0], is_recon_points[:, 1], is_recon_points[:, 2], marker='^')
        ax.scatter(points[occluded_idx, 0], points[occluded_idx, 1], points[occluded_idx, 2], marker='v')
        ax.scatter(points[visible_idx, 0], points[visible_idx, 1], points[visible_idx, 2], marker='4')
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], marker='8')
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_aspect("equal")
        plt.show()
    return cost


def setup_side_camera(config_path, test=False):
    """
    Setup side camera and static obstacle meshes

    Returns:
        camera_params: Dict of K and view matrix
        obstacle_merged_mesh: merged mesh of all obstacles
        static_mesh_points: dict of classified obstacle points
    """
    import pickle
    with open(config_path + 'obs_mesh.pickle', 'rb') as handle:
        obstacle_merged_mesh = pickle.load(handle)

    with open(config_path + 'static_mesh_points.pickle', 'rb') as handle:
        static_mesh_points = pickle.load(handle)


    camera_params = {'K': np.load(config_path + 'K_intrinsics.npy'),
                        'view_mat': np.load(config_path + 'view_mat.npy'),
                        'cam_pos': np.array([0.4, -0.5, 0.3]),
                        'width': 512,
                        'height': 512,
                        'near': 0.01,
                        'far': 100,
                        }
    return camera_params, obstacle_merged_mesh, static_mesh_points



#!/usr/bin/env python3
"""
lowdim_to_pointcloud_torch.py — Differentiable low-dim → point-cloud conversion.

Torch counterpart to lowdim_to_pointcloud.py, designed for diffusion-model
guidance losses. Gradients flow through `states` → point clouds → loss.

Templates are treated as constants (no grad). Sample them ONCE via
`sample_templates_batched(…)`, cache the tensors, and reuse them across
guidance calls — this keeps the loss meaningful (same shape being rendered)
and avoids per-call RNG work.

Inputs
------
states    : (B, T, 6) torch.Tensor, requires_grad supported.
            [tcp_x, tcp_y, block_x, block_y, block_cos, block_sin]
block_tpl : (B, n_block, 3) or (n_block, 3) torch.Tensor
hand_tpl  : (B, n_hand,  3) or (n_hand,  3) torch.Tensor

Output
------
pointclouds : (B, T, N, 3) torch.Tensor, N = n_hand + n_block.
              Gripper first [:n_hand], block second [n_hand:].

Centering
---------
At t=0 the reference object (default: block) is placed at xy = (0, 0) and
its lowest point at z = 0. The same offset is applied uniformly across
all frames, preserving relative motion.

- offset_mode="pose"  (default, SMOOTH): uses the state's pose position at
  t=0 for xy. Identical to bbox centering for a cube template; differs by a
  constant for a T template. Fully smooth gradient.
- offset_mode="bbox"  (matches numpy version EXACTLY): uses min/max of the
  generated points at t=0. Subgradient (only two particles contribute per
  spatial axis). Fine for optimisation but less smooth.
"""

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
 

from vis_push_pcd_floating import sample_templates  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Template sampling — numpy, non-differentiable, do this once and cache.
# ─────────────────────────────────────────────────────────────────────────────

def sample_templates_batched(
    batch_size: int,
    n_block   : int,
    n_hand    : int,
    shape     : str             = "cube",
    seed      : int | None      = None,
    device                       = None,
    dtype     : torch.dtype     = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `batch_size` independent (block_tpl, hand_tpl) pairs and return
    them as torch tensors of shape (B, n_block, 3) / (B, n_hand, 3).

    The per-trajectory fresh draw mirrors h5_to_wds_full_traj.py. Pass
    `seed` for reproducibility.
    """
    if shape not in ("cube", "T"):
        raise ValueError(f"shape must be 'cube' or 'T'; got {shape!r}")
    use_T  = (shape == "T")
    rng    = np.random.default_rng(seed)
    block  = np.zeros((batch_size, n_block, 3), dtype=np.float32)
    hand   = np.zeros((batch_size, n_hand,  3), dtype=np.float32)
    for b in range(batch_size):
        tpls     = sample_templates(n_block, n_hand, rng)
        block[b] = tpls["T"] if use_T else tpls["cube"]
        hand[b]  = tpls["hand"]
    return (
        torch.as_tensor(block, dtype=dtype, device=device),
        torch.as_tensor(hand,  dtype=dtype, device=device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Differentiable conversion
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_cos_sin_to_rotmat_z(
    cos_yaw: torch.Tensor,
    sin_yaw: torch.Tensor,
    eps    : float = 1e-8,
) -> torch.Tensor:
    """
    (cos_yaw, sin_yaw) with shape (...) → R of shape (..., 3, 3) about +z.
    Robust to non-unit inputs (diffusion outputs may not lie exactly on the
    unit circle) — normalised with an epsilon for numerical stability.
    """
    norm = torch.sqrt(cos_yaw * cos_yaw + sin_yaw * sin_yaw + eps)
    c    = cos_yaw / norm
    s    = sin_yaw / norm
    zero = torch.zeros_like(c)
    one  = torch.ones_like(c)
    R = torch.stack([
        torch.stack([c,   -s, zero], dim=-1),
        torch.stack([s,    c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)
    return R


def states_to_pointclouds(
    states     : torch.Tensor,
    center_on  : str = "block",
    offset_mode: str = "pose",
) -> torch.Tensor:
    """
    Fully vectorised, differentiable in `states`. Templates are constants.

    Args
    ----
    states      : (B, T, 6)
    block_tpl   : (B, n_block, 3) or (n_block, 3)
    hand_tpl    : (B, n_hand,  3) or (n_hand,  3)
    center_on   : "block" or "gripper"
    offset_mode : "pose" (smooth, default) or "bbox" (matches numpy)

    Returns
    -------
    (B, T, n_hand + n_block, 3) — gripper first, block second.
    """
    if states.dim() != 3 or states.shape[-1] != 6:
        raise ValueError(f"Expected states of shape (B, T, 6); got {tuple(states.shape)}")
    if center_on not in ("block", "gripper"):
        raise ValueError(f"center_on must be 'block' or 'gripper'; got {center_on!r}")
    if offset_mode not in ("pose", "bbox"):
        raise ValueError(f"offset_mode must be 'pose' or 'bbox'; got {offset_mode!r}")

    B, T, _ = states.shape
    device, dtype = states.device, states.dtype
    
    tpls = np.load(_HERE / "point_cost_templates.npz")
    block_tpl = torch.as_tensor(tpls["block_tpl"], dtype=dtype, device=device)  # (n_block, 3)
    hand_tpl  = torch.as_tensor(tpls["hand_tpl"],  dtype=dtype, device=device)  # (n_hand, 3)
    

    # Broadcast templates to (B, ·, 3) without materialising copies.
    if block_tpl.dim() == 2:
        block_tpl = block_tpl.unsqueeze(0).expand(B, -1, -1)
    if hand_tpl.dim() == 2:
        hand_tpl = hand_tpl.unsqueeze(0).expand(B, -1, -1)
    block_tpl = block_tpl.to(device=device, dtype=dtype)
    hand_tpl  = hand_tpl.to(device=device, dtype=dtype)
    n_hand    = hand_tpl.shape[-2]

    # Unpack state.
    tcp_xy    = states[..., 0:2]   # (B, T, 2)
    block_xy  = states[..., 2:4]
    block_cos = states[..., 4]     # (B, T)
    block_sin = states[..., 5]

    # Build 3D positions with z = 0 (low-dim state carries no z).
    zeros     = torch.zeros_like(tcp_xy[..., :1])          # (B, T, 1)
    tcp_pos   = torch.cat([tcp_xy,   zeros], dim=-1)       # (B, T, 3)
    block_pos = torch.cat([block_xy, zeros], dim=-1)       # (B, T, 3)

    # Block rotation matrix about +z.
    R = _yaw_cos_sin_to_rotmat_z(block_cos, block_sin)     # (B, T, 3, 3)

    # Gripper: identity rotation → broadcast + translate.
    #   hand_tpl : (B, n_hand, 3) → (B, 1, n_hand, 3)
    #   tcp_pos  : (B, T,     3) → (B, T, 1,      3)
    hand_pts = hand_tpl.unsqueeze(1) + tcp_pos.unsqueeze(-2)   # (B, T, n_hand, 3)

    # Block: rotate then translate.
    block_rot = torch.einsum("btij,bnj->btni", R, block_tpl)   # (B, T, n_block, 3)
    block_pts = block_rot + block_pos.unsqueeze(-2)            # (B, T, n_block, 3)

    out = torch.cat([hand_pts, block_pts], dim=-2)             # (B, T, N, 3)

    # ── Centering ─────────────────────────────────────────────────────────
    if offset_mode == "pose":
        if center_on == "block":
            offset_xy = block_pos[:, 0, :2]                    # (B, 2)
            offset_z  = block_tpl[..., 2].amin(dim=-1)         # (B,) — yaw-invariant
        else:
            offset_xy = tcp_pos[:, 0, :2]
            offset_z  = hand_tpl[..., 2].amin(dim=-1)
    else:  # "bbox"
        ref = out[:, 0, n_hand:, :] if center_on == "block" else out[:, 0, :n_hand, :]
        # ref: (B, n_ref, 3)
        offset_xy = 0.5 * (ref[..., :2].amin(dim=-2) + ref[..., :2].amax(dim=-2))  # (B, 2)
        offset_z  = ref[..., 2].amin(dim=-1)                                       # (B,)

    offset = torch.cat([offset_xy, offset_z.unsqueeze(-1)], dim=-1)  # (B, 3)
    out    = out - offset.view(B, 1, 1, 3)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: cache templates once, reuse for many guidance calls.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # sample hand and block templates once, save to disk for reuse across guidance calls
    block_tpl, hand_tpl = sample_templates_batched(
        batch_size=1,
        n_block=128,
        n_hand=128,
        shape="cube",
        seed=0,
    )
    block_tpl_np = block_tpl.squeeze(0).cpu().numpy()  # (n_block, 3)
    hand_tpl_np  = hand_tpl.squeeze(0).cpu().numpy()   # (n_hand, 3)
    np.savez("point_cost_templates.npz", block_tpl=block_tpl_np, hand_tpl=hand_tpl_np)
    
if __name__ == "__main__":
    main()

