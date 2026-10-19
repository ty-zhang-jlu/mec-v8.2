"""CPU-only tests for Stage-2g.5 movement data/metrics helpers."""

import unittest

import numpy as np

from v83_spatial_stage2.stage2g5_data import (
    BASE_INDICES,
    COH_INDICES,
    HEAT_INDICES,
    SPATIAL_CHANNELS,
    build_movement_inputs,
    build_temporal_spatial,
    class_weights,
    reconstruct_current_maps,
    subset_masks,
    yaw_warp_maps,
)
from v83_spatial_stage2.stage2g5_metrics import (
    all_model_failure,
    classification_metrics,
    confusion_matrix,
)


def make_data(n=24):
    episode = np.repeat([0, 1], n // 2)
    step = np.tile(np.arange(n // 2), 2)
    actions = np.zeros(n, dtype=np.int64)
    actions[1] = 2  # previous turn at step 1
    actions[10] = 1
    current = np.zeros((n, 2858), dtype=np.float32)
    for channel in range(3):
        current[:, HEAT_INDICES[channel]] = np.linspace(
            0.0, 1.0 + channel, n
        )[:, None]
    current[:, BASE_INDICES] = 0.5
    current[:, COH_INDICES] = 0.2
    return {
        "episode_id": episode,
        "band": np.zeros(n, dtype=np.int64),
        "step": step,
        "action": actions,
        "stop": np.zeros(n, dtype=np.int64),
        "oracle_move_action": np.tile([0, 1, 2], n // 3 + 1)[:n],
        "distance_m": np.full(n, 50.0),
        "target_visible": np.tile([1, 0], n // 2),
        "since_localizable": np.tile([0, 3], n // 2),
        "heading_error_deg": np.zeros(n),
        "current_features": current,
        "compact_features": current[:, :74].copy(),
        "common_mask": np.ones(n, dtype=bool),
    }


class TestSpatialMaps(unittest.TestCase):
    def test_reconstruct_shapes(self):
        data = make_data()
        maps = reconstruct_current_maps(data["current_features"])
        self.assertEqual(maps.shape, (len(data["step"]), 5, 8, 8))
        self.assertTrue(np.allclose(
            maps[:, 3], data["current_features"][:, BASE_INDICES].reshape(-1, 8, 8)
        ))

    def test_yaw_warp_validity(self):
        maps = np.zeros((3, 5, 8, 8), dtype=np.float32)
        maps[:, :, :, 1] = 1.0
        warped, valid = yaw_warp_maps(maps, np.array([0, 1, 2]))
        self.assertEqual(valid.tolist(), [False, True, True])
        self.assertEqual(float(np.abs(warped[0].numpy()).sum()), 0.0)
        self.assertGreater(float(np.abs(warped[1].numpy()).sum()), 0.0)

    def test_temporal_fallback(self):
        data = make_data()
        temporal = build_temporal_spatial(data)
        self.assertEqual(temporal["spatial"].shape, (
            len(data["step"]), SPATIAL_CHANNELS, 8, 8
        ))
        # Forward-preceded steps have no yaw-valid history and use current.
        forward_steps = [
            i for i in range(1, len(data["step"]))
            if data["episode_id"][i] == data["episode_id"][i - 1]
            and data["step"][i] == data["step"][i - 1] + 1
            and data["action"][i - 1] == 0
        ]
        for index in forward_steps:
            self.assertFalse(temporal["valid"][index])
            self.assertTrue(np.allclose(
                temporal["spatial"][index, 10:15],
                temporal["current"][index],
            ))


class TestMovementInputs(unittest.TestCase):
    def test_variant_shapes_and_no_label_leak(self):
        first = make_data()
        second = make_data()
        # Change every supervision/diagnostic label; observables unchanged.
        second["target_visible"] = 1 - second["target_visible"]
        second["distance_m"] = second["distance_m"] + 10
        second["heading_error_deg"] = second["heading_error_deg"] + 5
        second["oracle_move_action"] = (second["oracle_move_action"] + 1) % 3
        cache_a = {}
        cache_b = {}
        a1 = build_movement_inputs(first, first["common_mask"], "B1", cache_a)
        b1 = build_movement_inputs(second, second["common_mask"], "B1", cache_b)
        np.testing.assert_array_equal(a1["tabular"], b1["tabular"])
        a2 = build_movement_inputs(first, first["common_mask"], "B2", cache_a)
        b2 = build_movement_inputs(second, second["common_mask"], "B2", cache_b)
        np.testing.assert_array_equal(a2["tabular"], b2["tabular"])
        a3 = build_movement_inputs(first, first["common_mask"], "B3", cache_a)
        b3 = build_movement_inputs(second, second["common_mask"], "B3", cache_b)
        np.testing.assert_array_equal(a3["tabular"], b3["tabular"])
        np.testing.assert_array_equal(a3["spatial"], b3["spatial"])
        self.assertEqual(a3["spatial"].shape[1], SPATIAL_CHANNELS)

    def test_class_weights(self):
        weights, counts = class_weights(np.array([0, 0, 1, 2]))
        self.assertEqual(counts.tolist(), [2, 1, 1])
        self.assertGreater(weights[1], weights[0])


class TestMetrics(unittest.TestCase):
    def test_confusion_and_metrics(self):
        matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertEqual(matrix.tolist(), [
            [1, 1, 0], [0, 1, 0], [0, 0, 1]
        ])
        metrics = classification_metrics(
            [0, 0, 1, 2], [0, 1, 1, 2],
            np.array([
                [5, 0, -5], [0, 5, -5],
                [-5, 0, 5], [-5, 0, 5],
            ], dtype=float),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("lr_auc", metrics)

    def test_all_model_failure(self):
        data = make_data()
        labels = np.asarray(data["oracle_move_action"])
        preds = {
            "B1": labels.copy(), "B2": labels.copy(), "B3": labels.copy()
        }
        preds["B1"][0] = (preds["B1"][0] + 1) % 3
        preds["B2"][0] = (preds["B2"][0] + 1) % 3
        preds["B3"][0] = (preds["B3"][0] + 1) % 3
        result = all_model_failure(data, preds, base_pred=labels)
        self.assertEqual(result["all_wrong"], 1)
        self.assertIn("invisible", result["groups"])


if __name__ == "__main__":
    unittest.main()
