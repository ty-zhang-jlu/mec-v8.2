"""Stage-2g.5 movement diagnosis analysis and frozen decision rule."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    load_event_split,
    view_mask,
)
from v83_spatial_stage2.stage2g5_data import (  # noqa: E402
    BAND_NAMES,
    FEATURE_INDEX,
    MOVEMENT_NAMES,
    build_temporal_spatial,
    movement_view,
    subset_masks,
)
from v83_spatial_stage2.stage2g5_metrics import (  # noqa: E402
    all_model_failure,
    classification_metrics,
    episode_bootstrap,
    paired_bootstrap,
    per_episode_table,
    subset_metrics,
)


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def lr_balanced_recall(metrics):
    left = metrics["per_class"]["LEFT"]["recall"]
    right = metrics["per_class"]["RIGHT"]["recall"]
    return float((left + right) / 2.0)


def safe_diff(a, b):
    if a is None or b is None:
        return None
    return float(a - b)


def feature_means(data, mask):
    if not np.any(mask):
        return {}
    features = data["current_features"]
    names = {
        "sim_max_all_top5": "sim_max_all_top5",
        "heat_0_top1": "heat_0_top1",
        "base_max": "base_max",
        "coh_max": "coh_max",
    }
    output = {}
    for label, name in names.items():
        output[label] = float(
            features[mask, FEATURE_INDEX[name]].mean()
        )
    output["visible_rate"] = float(
        (np.asarray(data["target_visible"])[mask] == 1).mean()
    )
    output["mean_distance_m"] = float(
        np.asarray(data["distance_m"])[mask].mean()
    )
    output["mean_abs_heading_deg"] = float(
        np.abs(np.asarray(data["heading_error_deg"])[mask]).mean()
    )
    since = np.asarray(data["since_localizable"])[mask]
    output["mean_since_localizable"] = float(
        since[since >= 0].mean()
    ) if np.any(since >= 0) else float("nan")
    output["never_localizable_rate"] = float((since < 0).mean())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading data", flush=True)
    val_raw = load_event_split(args.val_dir)
    val_data = movement_view(val_raw)
    results = json.loads(
        (args.probe_dir / "probe_results.json").read_text(encoding="utf-8")
    )
    predictions = np.load(args.probe_dir / "predictions.npz")
    labels = np.asarray(val_data["oracle_move_action"], dtype=int)
    episodes = np.asarray(val_data["episode_id"])
    bands = np.asarray(val_data["band"])

    masks = subset_masks(val_data)
    for low, high in ((0, 20), (20, 50), (50, 100), (100, 200),
                      (200, 400), (400, 10000)):
        masks[f"distance_{low}_{high}"] = (
            (np.asarray(val_data["distance_m"]) >= low)
            & (np.asarray(val_data["distance_m"]) < high)
        )
    invisible = masks["invisible"]
    since = np.asarray(val_data["since_localizable"])
    near = np.asarray(val_data["distance_m"]) < 100.0
    masks["invisible_since_1_4"] = (
        invisible & (since >= 1) & (since <= 4)
    )
    masks["invisible_since_5_16"] = (
        invisible & (since >= 5) & (since <= 16)
    )
    masks["invisible_since_never_gt16"] = (
        invisible & ((since < 0) | (since > 16))
    )
    masks["near_invisible_since_1_4"] = (
        near & invisible & (since >= 1) & (since <= 4)
    )
    masks["near_invisible_since_5_16"] = (
        near & invisible & (since >= 5) & (since <= 16)
    )
    for code, name in BAND_NAMES.items():
        masks[f"band_{name}"] = bands == code
    temporal = build_temporal_spatial(val_raw)
    temporal_valid = np.asarray(temporal["valid"])[
        val_raw["common_mask"]
    ]
    masks["spatial_valid"] = temporal_valid
    masks["spatial_invalid"] = ~temporal_valid

    model_pred = {
        "B0": np.asarray(predictions["m0_action"], dtype=int),
        "B1": np.asarray(predictions["B1_val_pred"], dtype=int),
        "B2": np.asarray(predictions["B2_val_pred"], dtype=int),
        "B3": np.asarray(predictions["B3_val_pred"], dtype=int),
    }
    model_logits = {
        "B0": None,
        "B1": np.asarray(predictions["B1_val_logits"]),
        "B2": np.asarray(predictions["B2_val_logits"]),
        "B3": np.asarray(predictions["B3_val_logits"]),
    }

    analysis = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "probe_dir": str(args.probe_dir),
            "no_training_performed": True,
            "final_test_accessed": False,
        },
        "data": {
            "val_states": int(len(labels)),
            "val_episodes": int(np.unique(episodes).size),
            "class_counts": {
                MOVEMENT_NAMES[code]: int((labels == code).sum())
                for code in range(3)
            },
            "temporal_valid_states": int(temporal_valid.sum()),
            "temporal_valid_rate": float(temporal_valid.mean()),
        },
        "models": {},
        "audit": {},
        "paired": {},
        "bootstrap": {},
        "valid_split": {},
        "failure": {},
        "per_episode": {},
    }

    for name in ("B0", "B1", "B2", "B3"):
        pred = model_pred[name]
        logits = model_logits[name]
        entry = {
            "overall": classification_metrics(labels, pred, logits),
            "subsets": {
                key: subset_metrics(val_data, mask, pred, logits, key)
                for key, mask in masks.items()
            },
            "lr_balanced_recall": lr_balanced_recall(
                classification_metrics(labels, pred, logits)
            ),
        }
        analysis["models"][name] = entry
        analysis["bootstrap"][name] = {
            key: episode_bootstrap(
                val_data, mask, pred, logits,
                n_boot=args.bootstrap,
            )
            for key, mask in {
                "all": np.ones(len(labels), dtype=bool),
                "visible": masks["visible"],
                "invisible": masks["invisible"],
                "near_invisible": masks["near_invisible"],
            }.items()
        }
        analysis["per_episode"][name] = {
            "accuracy": float(np.mean([
                row["accuracy"] for row in per_episode_table(val_data, pred)
            ])),
            "table": per_episode_table(val_data, pred),
        }
        if name != "B0":
            saved = results["variants"][name]
            analysis["audit"][name] = {
                "auroc_saved": saved["val"]["accuracy"],
                "accuracy_recomputed": entry["overall"]["accuracy"],
                "balanced_saved": saved["val"]["balanced_accuracy_present"],
                "balanced_recomputed": entry["overall"][
                    "balanced_accuracy_present"
                ],
                "macro_f1_saved": saved["val"]["macro_f1_present"],
                "macro_f1_recomputed": entry["overall"]["macro_f1_present"],
            }

    def episode_wins(a, b):
        rows_a = {
            row["episode_id"]: row["accuracy"]
            for row in analysis["per_episode"][a]["table"]
        }
        rows_b = {
            row["episode_id"]: row["accuracy"]
            for row in analysis["per_episode"][b]["table"]
        }
        diffs = [
            rows_a[key] - rows_b[key] for key in rows_a
            if key in rows_b
        ]
        return {
            "wins": int(sum(value > 1e-12 for value in diffs)),
            "losses": int(sum(value < -1e-12 for value in diffs)),
            "ties": int(sum(abs(value) <= 1e-12 for value in diffs)),
            "mean_delta": float(np.mean(diffs)),
        }

    for a, b in (("B2", "B1"), ("B3", "B1"), ("B3", "B2")):
        analysis["paired"][f"{a}_minus_{b}"] = {
            key: paired_bootstrap(
                val_data, mask,
                model_pred[b], model_logits[b],
                model_pred[a], model_logits[a],
                n_boot=args.bootstrap,
            )
            for key, mask in {
                "all": np.ones(len(labels), dtype=bool),
                "visible": masks["visible"],
                "invisible": masks["invisible"],
                "near_invisible": masks["near_invisible"],
            }.items()
        }

    analysis["valid_split"] = {}
    for name in ("B1", "B2", "B3"):
        analysis["valid_split"][name] = {
            "valid": subset_metrics(
                val_data, temporal_valid, model_pred[name],
                model_logits[name], "spatial_valid",
            ),
            "invalid": subset_metrics(
                val_data, ~temporal_valid, model_pred[name],
                model_logits[name], "spatial_invalid",
            ),
        }

    analysis["failure"] = all_model_failure(
        val_data,
        {name: model_pred[name] for name in ("B1", "B2", "B3")},
        base_pred=model_pred["B0"],
    )
    wrong = np.stack([
        model_pred[name] != labels for name in ("B1", "B2", "B3")
    ], axis=0).all(axis=0)
    correct = np.stack([
        model_pred[name] == labels for name in ("B1", "B2", "B3")
    ], axis=0).all(axis=0)
    analysis["failure"]["all_wrong_feature_means"] = feature_means(
        val_data, wrong
    )
    analysis["failure"]["all_correct_feature_means"] = feature_means(
        val_data, correct
    )

    # Frozen decision rule.
    invisible = "invisible"
    near_invisible = "near_invisible"
    def get(model, subset, key):
        return analysis["models"][model]["subsets"][subset][key]
    def paired_get(a, b, subset, key):
        pair = analysis["paired"][f"{a}_minus_{b}"][subset]
        return pair.get(key, {"mean": float("nan"), "low": float("nan")})
    b1_bal = get("B1", invisible, "balanced_accuracy_present")
    b1_macro = get("B1", invisible, "macro_f1_present")
    b2_bal = get("B2", invisible, "balanced_accuracy_present")
    b2_macro = get("B2", invisible, "macro_f1_present")
    b3_bal = get("B3", invisible, "balanced_accuracy_present")
    b3_macro = get("B3", invisible, "macro_f1_present")
    b3_lr = (
        analysis["models"]["B3"]["subsets"][invisible]["per_class"]["LEFT"]["recall"]
        + analysis["models"]["B3"]["subsets"][invisible]["per_class"]["RIGHT"]["recall"]
    ) / 2
    b2_lr = (
        analysis["models"]["B2"]["subsets"][invisible]["per_class"]["LEFT"]["recall"]
        + analysis["models"]["B2"]["subsets"][invisible]["per_class"]["RIGHT"]["recall"]
    ) / 2
    pair_b3_b1_bal = paired_get("B3", "B1", invisible, "balanced_accuracy_present")
    pair_b3_b1_macro = paired_get("B3", "B1", invisible, "macro_f1_present")
    pair_b3_b2_bal = paired_get("B3", "B2", invisible, "balanced_accuracy_present")
    pair_b3_b2_macro = paired_get("B3", "B2", invisible, "macro_f1_present")
    pair_b2_b1_bal = paired_get("B2", "B1", invisible, "balanced_accuracy_present")
    pair_b2_b1_macro = paired_get("B2", "B1", invisible, "macro_f1_present")
    b3_vs_b1_gain = max(b3_bal - b1_bal, b3_macro - b1_macro)
    b3_vs_b2_gain = max(b3_bal - b2_bal, b3_macro - b2_macro)
    b2_vs_b1_gain = max(b2_bal - b1_bal, b2_macro - b1_macro)
    b3_b1_low = max(
        pair_b3_b1_bal.get("low", float("-inf")),
        pair_b3_b1_macro.get("low", float("-inf")),
    )
    b3_b2_low = max(
        pair_b3_b2_bal.get("low", float("-inf")),
        pair_b3_b2_macro.get("low", float("-inf")),
    )
    b2_b1_low = max(
        pair_b2_b1_bal.get("low", float("-inf")),
        pair_b2_b1_macro.get("low", float("-inf")),
    )
    pass_spatial = (
        b3_vs_b1_gain >= 0.02
        and b3_vs_b2_gain >= 0.02
        and b3_b1_low > 0
        and b3_b2_low > 0
        and (b3_lr - b2_lr) >= 0.02
    )
    partial_history = (
        not pass_spatial
        and b2_vs_b1_gain >= 0.02
        and b2_b1_low > 0
    )
    analysis["decision"] = {
        "invisible_balanced_accuracy": {
            name: get(name, invisible, "balanced_accuracy_present")
            for name in ("B0", "B1", "B2", "B3")
        },
        "invisible_macro_f1_present": {
            name: get(name, invisible, "macro_f1_present")
            for name in ("B0", "B1", "B2", "B3")
        },
        "invisible_lr_balanced_recall": {
            "B1": (
                analysis["models"]["B1"]["subsets"][invisible]["per_class"]["LEFT"]["recall"]
                + analysis["models"]["B1"]["subsets"][invisible]["per_class"]["RIGHT"]["recall"]
            ) / 2,
            "B2": b2_lr,
            "B3": b3_lr,
        },
        "near_invisible_balanced_accuracy": {
            name: get(name, near_invisible, "balanced_accuracy_present")
            for name in ("B0", "B1", "B2", "B3")
        },
        "b2_vs_b1_invisible_gain": b2_vs_b1_gain,
        "b3_vs_b1_invisible_gain": b3_vs_b1_gain,
        "b3_vs_b2_invisible_gain": b3_vs_b2_gain,
        "b3_vs_b1_paired_ci": pair_b3_b1_bal,
        "b3_vs_b1_macro_paired_ci": pair_b3_b1_macro,
        "b3_vs_b2_paired_ci": pair_b3_b2_bal,
        "b3_vs_b2_macro_paired_ci": pair_b3_b2_macro,
        "b2_vs_b1_paired_ci": pair_b2_b1_bal,
        "b2_vs_b1_macro_paired_ci": pair_b2_b1_macro,
        "episode_wins": {
            "B2_vs_B1": episode_wins("B2", "B1"),
            "B3_vs_B2": episode_wins("B3", "B2"),
        },
        "b3_b1_ci_low_best": b3_b1_low,
        "b3_b2_ci_low_best": b3_b2_low,
        "b2_b1_ci_low_best": b2_b1_low,
        "pass_spatial_belief": bool(pass_spatial),
        "partial_history_only": bool(partial_history),
        "decision": (
            "PASS: report controlled M0 + Temporal Spatial Belief MEC "
            "ablation plan only"
            if pass_spatial else
            "PARTIAL: long history helps but current spatial-map fusion "
            "does not add value"
            if partial_history else
            "FAIL: stop the image-matching history route; report the "
            "observability limit and alternatives"
        ),
    }

    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "B0": {
            "accuracy": analysis["models"]["B0"]["overall"]["accuracy"],
            "balanced": analysis["models"]["B0"]["overall"][
                "balanced_accuracy_present"
            ],
        },
        "models": {
            name: {
                "accuracy": analysis["models"][name]["overall"]["accuracy"],
                "balanced": analysis["models"][name]["overall"][
                    "balanced_accuracy_present"
                ],
                "invisible_balanced": analysis["models"][name]["subsets"][
                    "invisible"
                ]["balanced_accuracy_present"],
            }
            for name in ("B1", "B2", "B3")
        },
        "decision": analysis["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
