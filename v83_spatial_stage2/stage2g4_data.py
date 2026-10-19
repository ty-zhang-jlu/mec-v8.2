"""Stage-2g.4 data: long-horizon target-loss event labels and inputs.

Reuses the Stage-2g.3 real closed-loop NPZ files. Long histories are
reconstructed from each state's saved `current_features` (the 74 compact
columns), so no new collection is needed. Human-readable labels (distance,
visibility, localizability) are used only to create the offline event label;
they are never part of the model input.

Event label:
    recent_near_loss_K = distance < 100 m
                         AND target_visible == 0
                         AND target_localizable == 1 at some t-j, 1<=j<=K

The last-seen memory is a *label-free* deterministic summary of observable
visual-match confidence, the heatmap statistics and history actions. It never
reads target_localizable / target_visible at runtime.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP  # noqa: E402

from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    compact_indices,
    one_hot_actions,
)
from v83_spatial_stage2.stage2g_features import feature_names  # noqa: E402

BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
EVENT_KS = (4, 8, 16)
MIN_STEP = 16
# Observable confidence components. These are all model-observable columns;
# no distance / visibility / localizability column is used.
CONF_FEATURES = (
    "sim_max_all_top5",
    "heat_0_top1",
    "heat_1_top1",
    "heat_2_top1",
    "base_max",
    "coh_max",
)
CONF_PERCENTILE = 70.0
MEMORY_HORIZON = 16
MEMORY_MAX_AGE = 64
FEATURE_NAMES = feature_names()
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
COMPACT_IDX = compact_indices(FEATURE_NAMES)


def _step_masks(steps, actions):
    return (
        (np.asarray(steps) >= MIN_STEP)
        & (np.asarray(actions) != STOP)
    )


def event_labels(near, since, k):
    """Pure causal event label: near-invisible and localizable within K."""
    near = np.asarray(near, dtype=int)
    since = np.asarray(since, dtype=int)
    return (
        (near == 1)
        & (since >= 1)
        & (since <= int(k))
    )


def _since_until(localizable):
    localizable = np.asarray(localizable, dtype=int)
    n = len(localizable)
    since = np.full(n, -1, dtype=np.int64)
    until = np.full(n, -1, dtype=np.int64)
    last = -1
    for t in range(n):
        if localizable[t]:
            since[t] = 0
            last = t
        elif last >= 0:
            since[t] = t - last
    next_seen = -1
    for t in range(n - 1, -1, -1):
        if localizable[t]:
            until[t] = 0
            next_seen = t
        elif next_seen >= 0:
            until[t] = next_seen - t
    return since, until


def load_event_split(directory):
    """Load one split and attach event labels / long-history source arrays."""
    directory = Path(directory)
    files = sorted(directory.glob("episode_*.npz"))
    if not files:
        raise RuntimeError(f"No episode files in {directory}")
    parts = {
        key: []
        for key in (
            "episode_id", "band", "step", "action", "stop", "distance_m",
            "target_visible", "target_localizable", "near_target_invisible",
            "current_features", "since_localizable", "until_next_localizable",
            "oracle_move_action", "heading_error_deg", "goal_half_size_px",
            "goal_visible_geom",
        )
    }
    for path in files:
        data = np.load(path)
        n = len(data["step"])
        episode = int(data["episode_id"])
        parts["episode_id"].append(
            np.full(n, episode, dtype=np.int64)
        )
        parts["band"].append(
            np.full(n, int(data["band"]), dtype=np.int64)
        )
        for key in (
            "step", "action", "stop", "distance_m", "target_visible",
            "target_localizable", "near_target_invisible",
            "oracle_move_action", "heading_error_deg", "goal_half_size_px",
            "goal_visible_geom",
        ):
            parts[key].append(np.asarray(data[key]))
        parts["current_features"].append(
            np.asarray(data["current_features"], dtype=np.float32)
        )
        since, until = _since_until(data["target_localizable"])
        parts["since_localizable"].append(since)
        parts["until_next_localizable"].append(until)
    output = {
        key: np.concatenate(value, axis=0)
        for key, value in parts.items()
    }
    output["compact_features"] = output["current_features"][:, COMPACT_IDX]
    output["common_mask"] = _step_masks(
        output["step"], output["action"]
    )
    output["close_mask"] = (
        (output["step"] >= 2) & (output["action"] != STOP)
    )
    output["events"] = {}
    for k in EVENT_KS:
        output["events"][k] = event_labels(
            output["near_target_invisible"],
            output["since_localizable"],
            k,
        )
    return output


def event_stats(data):
    """Counts / band / episode coverage for each K."""
    close = data["close_mask"]
    common = data["common_mask"]
    stats = {}
    for k in EVENT_KS:
        event = data["events"][k]
        def block(mask):
            ev = event & mask
            episodes = np.asarray(data["episode_id"])[ev]
            bands = np.asarray(data["band"])[ev]
            return {
                "states": int(mask.sum()),
                "positives": int(ev.sum()),
                "positive_rate": (
                    float(ev.sum() / max(int(mask.sum()), 1))
                ),
                "episodes": int(np.unique(episodes).size),
                "band_positives": {
                    BAND_NAMES[int(b)]: int((bands == b).sum())
                    for b in sorted(set(bands.tolist()))
                },
                "band_episodes": {
                    BAND_NAMES[int(b)]: int(np.unique(
                        np.asarray(data["episode_id"])[ev & (
                            np.asarray(data["band"]) == b
                        )]
                    ).size)
                    for b in sorted(set(np.asarray(data["band"])[ev].tolist()))
                },
            }
        entry = {
            "close": block(close),
            "common": block(common),
            "since_localizable": {},
            "until_next_localizable": {},
            "never_relocalized": int(
                np.sum(
                    event & common
                    & (data["until_next_localizable"] < 0)
                )
            ),
            "relocalized_within_10": int(
                np.sum(
                    event & common
                    & (data["until_next_localizable"] >= 0)
                    & (data["until_next_localizable"] <= 10)
                )
            ),
        }
        since = data["since_localizable"][event & common]
        until = data["until_next_localizable"][event & common]
        if len(since):
            entry["since_localizable"] = {
                "mean": float(since.mean()),
                "quantiles": {
                    str(q): float(np.quantile(since, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
                },
            }
        observed = until[until >= 0]
        if len(observed):
            entry["until_next_localizable"] = {
                "n_observed": int(len(observed)),
                "mean": float(observed.mean()),
                "quantiles": {
                    str(q): float(np.quantile(observed, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
                },
            }
        stats[int(k)] = entry
    return stats


def event_episode_concentration(data, k):
    event = data["events"][k] & data["common_mask"]
    episodes = np.asarray(data["episode_id"])[event]
    counts = Counter(int(e) for e in episodes)
    values = np.asarray(sorted(counts.values()), dtype=float)
    if not len(values):
        return {"episodes": 0}
    ordered = np.sort(values)[::-1]
    shares = ordered.cumsum() / ordered.sum()
    return {
        "episodes": int(len(values)),
        "total": int(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "top1_share": float(shares[0]),
        "top3_share": float(shares[min(2, len(shares) - 1)]),
        "top5_share": float(shares[min(4, len(shares) - 1)]),
        "top10_share": float(shares[min(9, len(shares) - 1)]),
        "top_episodes": [
            {"episode_id": int(ep), "positives": int(c)}
            for ep, c in counts.most_common(10)
        ],
    }


def confidence_stats(train_data, percentile=CONF_PERCENTILE):
    """Fit label-free confidence normalization on Train common states."""
    mask = train_data["common_mask"]
    features = train_data["current_features"][mask]
    mean = {}
    std = {}
    for name in CONF_FEATURES:
        values = features[:, FEATURE_INDEX[name]].astype(np.float64)
        mean[name] = float(values.mean())
        std[name] = float(values.std() + 1e-9)
    confidence = confidence_vector(features, mean, std)
    return {
        "mean": mean,
        "std": std,
        "percentile": float(percentile),
        "threshold": float(np.percentile(confidence, percentile)),
        "components": list(CONF_FEATURES),
    }


def confidence_vector(features, mean, std):
    terms = []
    for name in CONF_FEATURES:
        values = features[:, FEATURE_INDEX[name]].astype(np.float64)
        terms.append((values - mean[name]) / std[name])
    return np.mean(terms, axis=0)


MEMORY_NAMES = [
    "mem_conf",
    "mem_high_at_t",
    "mem_age_high",
    "mem_has_high",
    "mem_last_conf",
    "mem_n_high_last_16",
    "mem_max_conf_last_16",
    "mem_mean_conf_last_16",
    "mem_argmax_age_last_16",
    *[f"mem_last_action_{name}" for name in ("FORWARD", "LEFT", "RIGHT", "STOP")],
    *[f"mem_last_compact_{index}" for index in range(len(COMPACT_FEATURE_NAMES))],
]


def build_memory(data, stats, horizon=MEMORY_HORIZON):
    """Causal observable last-seen memory; labels are not read here."""
    n = len(data["step"])
    features = np.zeros((n, len(MEMORY_NAMES)), dtype=np.float32)
    confidence = confidence_vector(
        data["current_features"], stats["mean"], stats["std"]
    )
    threshold = float(stats["threshold"])
    compact = data["compact_features"]
    actions = np.asarray(data["action"], dtype=np.int64)
    episode_ids = np.asarray(data["episode_id"])
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        last_high = -1
        for position, t in enumerate(indices.tolist()):
            high = bool(confidence[t] >= threshold)
            if high:
                last_high = position
            start = max(0, position - horizon + 1)
            window = confidence[indices[start:position + 1]]
            argmax = int(np.argmax(window))
            row = []
            row.append(confidence[t])
            row.append(1.0 if high else 0.0)
            age = (
                (position - last_high) if last_high >= 0 else MEMORY_MAX_AGE
            )
            row.append(min(age, MEMORY_MAX_AGE))
            row.append(1.0 if last_high >= 0 else 0.0)
            if last_high >= 0:
                last_t = int(indices[last_high])
                row.append(float(confidence[last_t]))
            else:
                last_t = -1
                row.append(0.0)
            row.append(float(np.sum(window >= threshold)))
            row.append(float(np.max(window)))
            row.append(float(np.mean(window)))
            row.append(float(position - start - argmax))
            if last_t >= 0:
                action = int(actions[last_t])
                action_onehot = np.zeros(4, dtype=np.float32)
                if 0 <= action < 4:
                    action_onehot[action] = 1.0
                last_compact = compact[last_t]
            else:
                action_onehot = np.zeros(4, dtype=np.float32)
                last_compact = np.zeros(len(COMPACT_FEATURE_NAMES), dtype=np.float32)
            row.extend(action_onehot.tolist())
            row.extend(last_compact.tolist())
            features[t] = np.asarray(row, dtype=np.float32)
    return features


def build_history(data, horizon):
    """Build t-1 ... t-horizon compact features, actions and dt."""
    n = len(data["step"])
    compact = data["compact_features"]
    actions = np.asarray(data["action"], dtype=np.int64)
    episodes = np.asarray(data["episode_id"])
    history = np.zeros(
        (n, horizon, len(COMPACT_FEATURE_NAMES)), dtype=np.float32
    )
    history_actions = np.full((n, horizon), -1, dtype=np.int8)
    history_dt = np.zeros((n, horizon), dtype=np.float32)
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        for offset in range(1, horizon + 1):
            if len(indices) <= offset:
                continue
            source = indices[:-offset]
            target = indices[offset:]
            history[target, offset - 1] = compact[source]
            history_actions[target, offset - 1] = actions[source]
            history_dt[target, offset - 1] = 1.0
    return history, history_actions, history_dt


def history_input(data, history, history_actions, history_dt):
    n = len(data["step"])
    horizon = history.shape[1]
    action_flat = np.asarray(history_actions, dtype=np.int64).reshape(-1)
    action_onehot = one_hot_actions(action_flat).reshape(n, horizon * 4)
    return np.concatenate([
        data["current_features"],
        history.reshape(n, -1),
        action_onehot,
        history_dt,
    ], axis=1)


def variant_spec(name):
    if name == "P0":
        return {"horizon": 0, "memory": False}
    if name == "P1-2":
        return {"horizon": 2, "memory": False}
    if name == "P1-8":
        return {"horizon": 8, "memory": False}
    if name == "P1-16":
        return {"horizon": 16, "memory": False}
    if name == "P1-16+mem":
        return {"horizon": 16, "memory": True}
    raise ValueError(f"Unknown variant {name}")


def build_variant(data, name, history_cache=None, memory_stats=None):
    spec = variant_spec(name)
    horizon = spec["horizon"]
    if horizon == 0:
        matrix = data["current_features"]
        names = list(FEATURE_NAMES)
    else:
        if history_cache is not None and horizon in history_cache:
            history, actions, dt = history_cache[horizon]
        else:
            history, actions, dt = build_history(data, horizon)
            if history_cache is not None:
                history_cache[horizon] = (history, actions, dt)
        matrix = history_input(data, history, actions, dt)
        names = []
        names.extend(FEATURE_NAMES)
        for offset in range(1, horizon + 1):
            names.extend(f"h{offset}_compact_{x}" for x in COMPACT_FEATURE_NAMES)
            names.extend(f"h{offset}_action_{x}" for x in (
                "FORWARD", "LEFT", "RIGHT", "STOP"
            ))
            names.append(f"h{offset}_dt")
    if spec["memory"]:
        if memory_stats is None:
            raise ValueError("P1-16+mem requires memory_stats")
        memory = build_memory(data, memory_stats)
        matrix = np.concatenate([matrix, memory], axis=1)
        names = names + list(MEMORY_NAMES)
    return np.asarray(matrix, dtype=np.float32), names


def view_mask(data, mask):
    """Create a shallow common-state view for metrics/training."""
    mask = np.asarray(mask, dtype=bool)
    output = {}
    for key, value in data.items():
        if key in ("events",):
            output[key] = {
                k: np.asarray(v)[mask] for k, v in value.items()
            }
        elif key == "common_mask":
            output[key] = np.asarray(value)[mask]
        elif key == "close_mask":
            output[key] = np.asarray(value)[mask]
        elif isinstance(value, np.ndarray) and value.shape[0] == len(mask):
            output[key] = value[mask]
        else:
            output[key] = value
    return output
