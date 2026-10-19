"""Stage-2g.5 movement diagnosis data and coarse spatial belief.

Reuses the Stage-2g.3 real M0 closed-loop NPZ states. The saved 2,858-d
feature vector contains a native 3x8x8 Goal-Current heatmap and 8x8 pooled
baseline/coherent maps; it does **not** contain the original 24x24 maps.
The 24x24 Stage-2a cache is aligned only at step 0 and diverges afterwards,
so it is not used for real closed-loop states. B3 therefore uses an explicit
8x8 coarse spatial belief and never claims 24x24 alignment.

No distance, heading, visibility, oracle action, or true goal position is an
input. Those fields are kept only for offline labels/evaluation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV  # noqa: E402
from env import FORWARD, LEFT, RIGHT  # noqa: E402

from v83_spatial_stage2.stage2g_features import feature_names  # noqa: E402
from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    build_history,
    history_input,
    load_event_split,
    view_mask,
)

BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
MOVEMENT_NAMES = {0: "FORWARD", 1: "LEFT", 2: "RIGHT"}
FEATURE_NAMES = feature_names()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
HEAT_INDICES = np.asarray([
    [FEATURE_INDEX[f"heat_{channel}_{index}"] for index in range(64)]
    for channel in range(3)
], dtype=np.int64)
BASE_INDICES = np.asarray(
    [FEATURE_INDEX[f"base_pool_{index}"] for index in range(64)],
    dtype=np.int64,
)
COH_INDICES = np.asarray(
    [FEATURE_INDEX[f"coh_pool_{index}"] for index in range(64)],
    dtype=np.int64,
)
SPATIAL_CHANNEL_NAMES = [
    "cur_heat_0", "cur_heat_1", "cur_heat_2", "cur_base", "cur_coh",
    "warp_heat_0", "warp_heat_1", "warp_heat_2", "warp_base", "warp_coh",
    "blend_heat_0", "blend_heat_1", "blend_heat_2", "blend_base", "blend_coh",
    "temporal_valid",
]
SPATIAL_CHANNELS = len(SPATIAL_CHANNEL_NAMES)
TEMPORAL_CARRY = 0.30


def reconstruct_current_maps(features):
    """Reconstruct 5x8x8 observable spatial maps from saved features."""
    features = np.asarray(features, dtype=np.float32)
    n = len(features)
    heat = features[:, HEAT_INDICES.reshape(-1)].reshape(n, 3, 8, 8)
    base = features[:, BASE_INDICES].reshape(n, 1, 8, 8)
    coh = features[:, COH_INDICES].reshape(n, 1, 8, 8)
    return np.concatenate([heat, base, coh], axis=1)


def yaw_warp_maps(maps, actions, turn_step_deg=None,
                  image_size=None, hfov_deg=None):
    """Apply the verified pinhole yaw reprojection to 8x8 maps.

    This is the same geometry as visual_grounding_v83.yaw_warp_previous, but
    applied to the saved coarse maps. FORWARD is deliberately invalid.
    """
    maps = torch.as_tensor(maps, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long)
    if maps.ndim != 4 or maps.shape[1:] != (5, 8, 8):
        raise ValueError("Expected maps [B,5,8,8]")
    if turn_step_deg is None:
        turn_step_deg = ENV.turn_step_deg
    if image_size is None:
        image_size = ENCODER.image_size
    if hfov_deg is None:
        hfov_deg = ENV.camera_hfov_deg
    valid = (actions == LEFT) | (actions == RIGHT)
    if not bool(valid.any()):
        return torch.zeros_like(maps), valid
    size = maps.shape[-1]
    yaw = torch.where(
        actions == LEFT,
        torch.ones_like(actions, dtype=torch.float32),
        -torch.ones_like(actions, dtype=torch.float32),
    ) * math.radians(float(turn_step_deg))
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    px_current = (x + 0.5) * (image_size / size)
    fx = (image_size * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    theta_cur = torch.atan((image_size * 0.5 - px_current) / fx)
    old_px = image_size * 0.5 - fx * torch.tan(
        theta_cur.unsqueeze(0) + yaw[:, None, None]
    )
    grid_x = 2 * old_px / image_size - 1
    grid_y = (2 * (y + 0.5) / size - 1)[None].expand_as(grid_x)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(
        maps, sampling_grid, mode="bilinear", padding_mode="zeros",
        align_corners=False,
    )
    warped = torch.where(
        valid[:, None, None, None], warped, torch.zeros_like(warped)
    )
    return warped, valid


def build_temporal_spatial(data):
    """Current 8x8 maps + previous-turn yaw-aligned map + validity plane."""
    current = reconstruct_current_maps(data["current_features"])
    n = len(current)
    warped = np.zeros_like(current)
    valid = np.zeros(n, dtype=bool)
    episode = np.asarray(data["episode_id"])
    step = np.asarray(data["step"])
    action = np.asarray(data["action"])
    previous = np.full(n, -1, dtype=np.int64)
    previous_action = np.full(n, -1, dtype=np.int64)
    if n > 1:
        same = (
            (episode[1:] == episode[:-1])
            & (step[1:] == step[:-1] + 1)
        )
        previous[1:][same] = np.arange(n - 1)[same]
        previous_action[1:][same] = action[:-1][same]
    candidate = (previous >= 0) & np.isin(previous_action, [LEFT, RIGHT])
    if candidate.any():
        selected = np.flatnonzero(candidate)
        warped_selected, valid_selected = yaw_warp_maps(
            current[previous[selected]], previous_action[selected]
        )
        warped[selected] = warped_selected.numpy()
        valid[selected] = valid_selected.numpy()
    blend = (
        (1.0 - TEMPORAL_CARRY) * current
        + TEMPORAL_CARRY * warped
    )
    blend[~valid] = current[~valid]
    valid_plane = np.broadcast_to(
        valid.astype(np.float32)[:, None, None, None],
        (n, 1, 8, 8),
    )
    spatial = np.concatenate([
        current,
        warped,
        blend,
        valid_plane,
    ], axis=1)
    return {
        "current": current,
        "warped": warped,
        "spatial": spatial.astype(np.float32),
        "valid": valid,
        "previous_index": previous,
        "previous_action": previous_action,
    }


def build_movement_inputs(raw, common_mask, variant, history_cache=None):
    """Return tabular and (for B3) spatial arrays for a variant."""
    if variant == "B1":
        tabular = np.asarray(raw["current_features"], dtype=np.float32)
        names = list(FEATURE_NAMES)
        spatial = None
        spatial_names = None
    elif variant == "B2":
        if history_cache is not None and "history16" in history_cache:
            history, actions, dt = history_cache["history16"]
        else:
            history, actions, dt = build_history(raw, 16)
            if history_cache is not None:
                history_cache["history16"] = (history, actions, dt)
        tabular = history_input(raw, history, actions, dt)
        names = []
        names.extend(FEATURE_NAMES)
        from v83_spatial_stage2.stage2g1_common import COMPACT_FEATURE_NAMES
        for offset in range(1, 17):
            names.extend(
                f"h{offset}_compact_{x}" for x in COMPACT_FEATURE_NAMES
            )
            names.extend(
                f"h{offset}_action_{x}" for x in (
                    "FORWARD", "LEFT", "RIGHT", "STOP"
                )
            )
            names.append(f"h{offset}_dt")
        spatial = None
        spatial_names = None
    elif variant == "B3":
        if history_cache is not None and "history16" in history_cache:
            history, actions, dt = history_cache["history16"]
        else:
            history, actions, dt = build_history(raw, 16)
            if history_cache is not None:
                history_cache["history16"] = (history, actions, dt)
        tabular = history_input(raw, history, actions, dt)
        names = []
        names.extend(FEATURE_NAMES)
        from v83_spatial_stage2.stage2g1_common import COMPACT_FEATURE_NAMES
        for offset in range(1, 17):
            names.extend(
                f"h{offset}_compact_{x}" for x in COMPACT_FEATURE_NAMES
            )
            names.extend(
                f"h{offset}_action_{x}" for x in (
                    "FORWARD", "LEFT", "RIGHT", "STOP"
                )
            )
            names.append(f"h{offset}_dt")
        if history_cache is not None and "temporal_spatial" in history_cache:
            temporal = history_cache["temporal_spatial"]
        else:
            temporal = build_temporal_spatial(raw)
            if history_cache is not None:
                history_cache["temporal_spatial"] = temporal
        spatial = temporal["spatial"]
        spatial_names = list(SPATIAL_CHANNEL_NAMES)
    else:
        raise ValueError(f"Unknown movement variant {variant}")
    return {
        "tabular": np.asarray(tabular[common_mask], dtype=np.float32),
        "tabular_names": names,
        "spatial": (
            None if spatial is None
            else np.asarray(spatial[common_mask], dtype=np.float32)
        ),
        "spatial_names": spatial_names,
    }


def movement_view(raw):
    return view_mask(raw, raw["common_mask"])


def class_weights(labels, num_classes=3):
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(float)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (
        present.sum() * counts[present]
    )
    return weights, counts


def subset_masks(data):
    visible = np.asarray(data["target_visible"]) == 1
    distance = np.asarray(data["distance_m"])
    since = np.asarray(data["since_localizable"])
    return {
        "visible": visible,
        "invisible": ~visible,
        "near_visible": (distance < 100.0) & visible,
        "near_invisible": (distance < 100.0) & ~visible,
        "far_visible": (distance >= 100.0) & visible,
        "far_invisible": (distance >= 100.0) & ~visible,
        "since_1_4": (since >= 1) & (since <= 4),
        "since_5_16": (since >= 5) & (since <= 16),
        "since_never_or_gt16": since < 0,
    }
