"""Stage-2h Goal-Anchor Belief (GAB).

An 8-bin egocentric target-bearing belief built only from observable current
features / 3x8x8 heatmap and executed actions. No true distance, visibility,
goal coordinates, oracle actions, or success labels are read.

The belief is causal and reset per episode. LEFT/RIGHT rotate the prior; FORWARD
never uses a fixed-pixel translation warp. Missing evidence decays confidence
toward a uniform belief instead of keeping stale over-confidence.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import FORWARD, LEFT, RIGHT  # noqa: E402
from v83_spatial_stage2.stage2g5_data import (  # noqa: E402
    reconstruct_current_maps,
)

BINS = 8
FOV_DEG = 90.0
TURN_DEG = 15.0
ALPHA = 0.50
BETA = 0.25
FORWARD_BETA = 0.15
MAX_AGE = 16
RELIABILITY_PERCENTILE = 60.0
GAB_FEATURE_NAMES = (
    [f"gab_belief_{index}" for index in range(BINS)]
    + [f"gab_last_reliable_{index}" for index in range(BINS)]
    + [
        "gab_confidence",
        "gab_entropy",
        "gab_last_seen_age",
        "gab_valid",
    ]
)


def uniform_belief():
    return np.full(BINS, 1.0 / BINS, dtype=np.float64)


def normalized_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / max(probabilities.sum(), 1e-12)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum()
    return float(entropy / math.log(len(probabilities)))


def current_evidence(maps, temperature=0.10):
    """Bearing evidence and confidence from the native 3x8x8 heatmap."""
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim == 4:
        maps = maps[0]
    heat = maps[:3].max(axis=0)
    flat = heat.reshape(-1)
    logits = flat / float(temperature)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= max(probabilities.sum(), 1e-12)
    grid = probabilities.reshape(8, 8)
    bins = grid.sum(axis=0)
    bins /= max(bins.sum(), 1e-12)
    return {
        "bins": bins.astype(np.float64),
        "confidence": float(probabilities.max()),
        "top1_column_mass": float(bins.max()),
        "entropy": normalized_entropy(bins),
        "peak_x": int(np.argmax(grid) % 8),
        "peak_y": int(np.argmax(grid) // 8),
    }


def shift_belief(probabilities, action, turn_deg=TURN_DEG, fov_deg=FOV_DEG):
    """Rotate an egocentric belief after a yaw action.

    Positive bin index is rightward in the image. Turning LEFT moves a world
    target to the right, so the prior shifts by +turn/bin_width; RIGHT shifts
    negative. FORWARD is no-op.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if action not in (LEFT, RIGHT):
        return probabilities.copy()
    bin_width = fov_deg / BINS
    shift = float(turn_deg) / bin_width
    if action == RIGHT:
        shift = -shift
    source_index = np.arange(BINS, dtype=np.float64) - shift
    lower = np.floor(source_index).astype(int)
    fraction = source_index - lower
    output = (
        (1.0 - fraction) * probabilities[lower % BINS]
        + fraction * probabilities[(lower + 1) % BINS]
    )
    return output / max(output.sum(), 1e-12)


@dataclass
class GABConfig:
    reliability_threshold: float
    alpha: float = ALPHA
    beta: float = BETA
    forward_beta: float = FORWARD_BETA
    max_age: int = MAX_AGE
    bins: int = BINS
    fov_deg: float = FOV_DEG
    turn_deg: float = TURN_DEG


class GABState:
    def __init__(self, config):
        self.config = config
        self.belief = uniform_belief()
        self.last_reliable = uniform_belief()
        self.confidence = 0.0
        self.last_seen_age = 0
        self.valid = 0.0

    def update(self, maps, previous_action):
        prior = shift_belief(
            self.belief, previous_action,
            self.config.turn_deg, self.config.fov_deg,
        )
        evidence = current_evidence(maps)
        reliable = evidence["confidence"] >= self.config.reliability_threshold
        if reliable:
            self.belief = (
                (1.0 - self.config.alpha) * prior
                + self.config.alpha * evidence["bins"]
            )
            self.belief /= max(self.belief.sum(), 1e-12)
            self.last_reliable = evidence["bins"].copy()
            self.confidence = (
                (1.0 - self.config.alpha) * self.confidence
                + self.config.alpha * evidence["confidence"]
            )
            self.last_seen_age = 0
            self.valid = 1.0
        else:
            decay = (
                self.config.forward_beta
                if previous_action == FORWARD
                else self.config.beta
            )
            self.belief = (
                (1.0 - decay) * prior
                + decay * uniform_belief()
            )
            self.belief /= max(self.belief.sum(), 1e-12)
            self.confidence *= (1.0 - decay)
            self.last_seen_age = min(
                self.last_seen_age + 1, self.config.max_age
            )
            if self.last_seen_age >= self.config.max_age:
                self.valid = 0.0
        return evidence

    def features(self):
        return np.concatenate([
            self.belief,
            self.last_reliable,
            np.asarray([
                self.confidence,
                normalized_entropy(self.belief),
                self.last_seen_age / max(self.config.max_age, 1),
                self.valid,
            ], dtype=np.float64),
        ]).astype(np.float32)


def fit_reliability(train_current_features, percentile=RELIABILITY_PERCENTILE):
    """Fit the reliability threshold on Train observable heatmaps only."""
    maps = reconstruct_current_maps(train_current_features)
    confidences = np.asarray([
        current_evidence(maps[index])["confidence"]
        for index in range(len(maps))
    ], dtype=np.float64)
    threshold = float(np.percentile(confidences, percentile))
    return {
        "threshold": threshold,
        "percentile": float(percentile),
        "n": int(len(confidences)),
        "mean": float(confidences.mean()),
        "quantiles": {
            str(q): float(np.quantile(confidences, q))
            for q in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        },
    }


def build_gab_config(train_current_features, percentile=RELIABILITY_PERCENTILE):
    stats = fit_reliability(train_current_features, percentile)
    return GABConfig(reliability_threshold=stats["threshold"]), stats


def build_gab_features(data, config, common_mask=None):
    """Causal per-episode GAB features for a concatenated episode split."""
    maps = reconstruct_current_maps(data["current_features"])
    episodes = np.asarray(data["episode_id"])
    steps = np.asarray(data["step"])
    actions = np.asarray(data["action"], dtype=np.int64)
    n = len(episodes)
    features = np.zeros((n, len(GAB_FEATURE_NAMES)), dtype=np.float32)
    confidences = np.zeros(n, dtype=np.float32)
    ages = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.int8)
    state = None
    current_episode = None
    previous_action = -1
    for index in range(n):
        episode = int(episodes[index])
        if episode != current_episode:
            state = GABState(config)
            current_episode = episode
            previous_action = -1
        # If a split is not contiguous, equality of step is checked.
        if index > 0 and episode == int(episodes[index - 1]):
            if int(steps[index]) != int(steps[index - 1]) + 1:
                state = GABState(config)
                previous_action = -1
        state.update(maps[index], previous_action)
        features[index] = state.features()
        confidences[index] = state.confidence
        ages[index] = state.last_seen_age
        valid[index] = int(state.valid)
        previous_action = int(actions[index])
    return {
        "features": features,
        "names": list(GAB_FEATURE_NAMES),
        "confidence": confidences,
        "age": ages,
        "valid": valid,
        "config": {
            "reliability_threshold": float(config.reliability_threshold),
            "alpha": float(config.alpha),
            "beta": float(config.beta),
            "forward_beta": float(config.forward_beta),
            "max_age": int(config.max_age),
            "bins": int(config.bins),
            "fov_deg": float(config.fov_deg),
            "turn_deg": float(config.turn_deg),
        },
    }


def build_gab_features_online(maps, actions, config):
    """Convenience for tests/offline sequences of one episode."""
    state = GABState(config)
    output = []
    previous_action = -1
    for index in range(len(maps)):
        state.update(maps[index], previous_action)
        output.append(state.features())
        previous_action = int(actions[index])
    return np.asarray(output, dtype=np.float32)
