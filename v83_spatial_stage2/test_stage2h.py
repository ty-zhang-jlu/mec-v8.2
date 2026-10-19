"""CPU-only tests for Stage-2h GAB and correction helpers."""

import unittest

import numpy as np

from env import FORWARD, LEFT, RIGHT
from v83_spatial_stage2.stage2g5_data import BASE_INDICES, COH_INDICES, HEAT_INDICES
from v83_spatial_stage2.stage2h_gab import (
    GABConfig,
    GABState,
    GAB_FEATURE_NAMES,
    build_gab_features,
    current_evidence,
    shift_belief,
    uniform_belief,
)
from v83_spatial_stage2.stage2h_r3_train import threshold_grid


def make_data(n=20):
    current = np.zeros((n, 2858), dtype=np.float32)
    for channel in range(3):
        current[:, HEAT_INDICES[channel]] = np.linspace(
            0.1, 0.9, n
        )[:, None]
    current[:, BASE_INDICES] = 0.5
    current[:, COH_INDICES] = 0.2
    return {
        "episode_id": np.repeat([0, 1], n // 2),
        "step": np.tile(np.arange(n // 2), 2),
        "action": np.tile([FORWARD, LEFT], n // 2)[:n],
        "current_features": current,
        "oracle_move_action": np.zeros(n, dtype=int),
        "distance_m": np.ones(n),
        "target_visible": np.ones(n, dtype=int),
        "heading_error_deg": np.zeros(n),
        "common_mask": np.ones(n, dtype=bool),
    }


class TestGAB(unittest.TestCase):
    def test_evidence_and_shift_sign(self):
        maps = np.zeros((5, 8, 8), dtype=np.float32)
        maps[0, :, 1] = 1.0
        evidence = current_evidence(maps)
        self.assertAlmostEqual(float(evidence["bins"].sum()), 1.0)
        self.assertGreater(evidence["confidence"], 0.0)
        belief = uniform_belief()
        belief[1] = 1.0
        left = shift_belief(belief, LEFT)
        right = shift_belief(belief, RIGHT)
        self.assertGreater(np.argmax(left), 1)
        self.assertLess(np.argmax(right), 1)
        np.testing.assert_allclose(shift_belief(belief, FORWARD), belief)

    def test_state_forward_and_decay(self):
        config = GABConfig(reliability_threshold=0.99)
        state = GABState(config)
        maps = np.zeros((5, 8, 8), dtype=np.float32)
        state.update(maps, FORWARD)
        before = state.confidence
        state.update(maps, FORWARD)
        self.assertLessEqual(state.confidence, before)
        self.assertEqual(state.last_seen_age, 2)

    def test_episode_reset_and_no_label_leak(self):
        first = make_data()
        second = make_data()
        second["target_visible"] = 1 - second["target_visible"]
        second["distance_m"] = second["distance_m"] + 5
        second["heading_error_deg"] = second["heading_error_deg"] + 5
        second["oracle_move_action"] = (second["oracle_move_action"] + 1) % 3
        config = GABConfig(reliability_threshold=1.1)
        a = build_gab_features(first, config)["features"]
        b = build_gab_features(second, config)["features"]
        np.testing.assert_array_equal(a, b)
        # Episode 1 starts from a fresh uniform belief.
        start1 = len(first["step"]) // 2
        np.testing.assert_allclose(
            a[start1, :8], a[0, :8], atol=1e-6
        )


class TestCorrection(unittest.TestCase):
    def test_positive_net_selection(self):
        # M0 wrong on two states; probe correct and confident there.
        m0 = np.array([0, 0, 0, 0])
        oracle = np.array([1, 1, 0, 0])
        logits = np.array([
            [0.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ])
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        result = threshold_grid(probabilities, m0, oracle)
        self.assertGreater(result["selected"]["net_agreement"], 0)
        self.assertGreater(result["selected"]["corrections"], 0)

    def test_no_positive_net_means_no_correction(self):
        m0 = np.array([0, 0])
        oracle = np.array([0, 0])
        probabilities = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = threshold_grid(probabilities, m0, oracle)
        self.assertEqual(result["selected"]["corrections"], 0)
        self.assertEqual(result["selected"]["threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
