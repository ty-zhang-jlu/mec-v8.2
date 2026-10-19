"""Stage-2h: train the R3 movement branch (16-step history + GAB).

Reuses the Stage-2g.5 B2 tabular input definition and adds 20 GAB features.
Only lightweight movement supervision is used; M0/M1/MEC are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import assign_episode_folds  # noqa: E402
from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    load_event_split,
)
from v83_spatial_stage2.stage2g5_data import (  # noqa: E402
    BAND_NAMES,
    MOVEMENT_NAMES,
    build_movement_inputs,
    class_weights,
    movement_view,
)
from v83_spatial_stage2.stage2g5_metrics import (  # noqa: E402
    classification_metrics,
)
from v83_spatial_stage2.stage2g5_probe import (  # noqa: E402
    MovementMLP,
    predict_logits,
    save_checkpoint,
    standardize_tabular,
    train_one,
)
from v83_spatial_stage2.stage2h_gab import (  # noqa: E402
    GABConfig,
    GAB_FEATURE_NAMES,
    build_gab_features,
    fit_reliability,
)


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def threshold_grid(probabilities, m0_action, oracle, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 0.99, 100)
    confidence = probabilities.max(axis=1)
    proposal = probabilities.argmax(axis=1)
    base_accuracy = float((m0_action == oracle).mean())
    rows = []
    for threshold in grid:
        correct = (
            (proposal != m0_action)
            & (confidence >= float(threshold))
        )
        final = np.where(correct, proposal, m0_action)
        accuracy = float((final == oracle).mean())
        helpful = int(
            (correct & (proposal == oracle) & (m0_action != oracle)).sum()
        )
        harmful = int(
            (correct & (proposal != oracle) & (m0_action == oracle)).sum()
        )
        neutral = int(correct.sum()) - helpful - harmful
        rows.append({
            "threshold": float(threshold),
            "corrections": int(correct.sum()),
            "helpful": helpful,
            "harmful": harmful,
            "neutral": neutral,
            "net_agreement": helpful - harmful,
            "accuracy": accuracy,
            "base_accuracy": base_accuracy,
            "net_accuracy": accuracy - base_accuracy,
        })
    positive = [row for row in rows if row["net_agreement"] > 0]
    if positive:
        best = max(
            positive,
            key=lambda row: (
                row["net_agreement"],
                row["accuracy"],
                row["threshold"],
            ),
        )
    else:
        best = {
            "threshold": 1.0,
            "corrections": 0,
            "helpful": 0,
            "harmful": 0,
            "neutral": 0,
            "net_agreement": 0,
            "accuracy": base_accuracy,
            "base_accuracy": base_accuracy,
            "net_accuracy": 0.0,
        }
    return {"base_accuracy": base_accuracy, "grid": rows, "selected": best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    print("loading data", flush=True)
    train_raw = load_event_split(args.train_dir)
    val_raw = load_event_split(args.val_dir)
    train_data = movement_view(train_raw)
    val_data = movement_view(val_raw)
    train_y = np.asarray(train_data["oracle_move_action"], dtype=np.int64)
    val_y = np.asarray(val_data["oracle_move_action"], dtype=np.int64)
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], 5, args.seed
    )
    train_cache = {}
    val_cache = {}
    train_b2 = build_movement_inputs(
        train_raw, train_raw["common_mask"], "B2", train_cache
    )
    val_b2 = build_movement_inputs(
        val_raw, val_raw["common_mask"], "B2", val_cache
    )
    reliability = fit_reliability(
        train_raw["current_features"][train_raw["common_mask"]]
    )
    config = GABConfig(reliability_threshold=reliability["threshold"])
    print("building GAB", flush=True)
    train_gab = build_gab_features(train_raw, config)
    val_gab = build_gab_features(val_raw, config)
    train_x = np.concatenate([
        train_b2["tabular"],
        train_gab["features"][train_raw["common_mask"]],
    ], axis=1).astype(np.float32)
    val_x = np.concatenate([
        val_b2["tabular"],
        val_gab["features"][val_raw["common_mask"]],
    ], axis=1).astype(np.float32)
    names = list(train_b2["tabular_names"]) + list(GAB_FEATURE_NAMES)
    if train_x.shape[1] != len(names):
        raise RuntimeError("R3 input/name dimension mismatch")

    weights, counts = class_weights(train_y)
    oof_logits = np.zeros((len(train_x), 3), dtype=np.float64)
    fold_info = {}
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std, mean, std = standardize_tabular(
            train_x[train_mask], train_x[valid_mask]
        )
        model = MovementMLP(
            train_x.shape[1], args.hidden
        )
        logits = train_one(
            model, train_std, None, train_y[train_mask],
            valid_std, None, weights, args, device,
        )
        oof_logits[valid_mask] = logits
        fold_info[str(fold)] = {
            "train_states": int(train_mask.sum()),
            "valid_states": int(valid_mask.sum()),
            "train_class_counts": {
                MOVEMENT_NAMES[int(code)]: int((train_y[train_mask] == code).sum())
                for code in range(3)
            },
        }
        if args.save_models:
            save_checkpoint(
                models_dir / f"R3_fold{fold}.pt",
                model, mean, std, None, None, "R3", fold,
                names, None, weights,
                {
                    "hidden": int(args.hidden), "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size), "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "gab": reliability,
                },
                {"train_dir": str(args.train_dir), "val_dir": str(args.val_dir)},
            )
    train_std, val_std, mean, std = standardize_tabular(train_x, val_x)
    model = MovementMLP(train_x.shape[1], args.hidden)
    train_one(
        model, train_std, None, train_y, val_std, None,
        weights, args, device,
    )
    val_logits = predict_logits(model, val_std, None, device)
    if args.save_models:
        save_checkpoint(
            models_dir / "R3_external.pt",
            model, mean, std, None, None, "R3", -1,
            names, None, weights,
            {
                "hidden": int(args.hidden), "epochs": int(args.epochs),
                "batch_size": int(args.batch_size), "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "gab": reliability,
            },
            {"train_dir": str(args.train_dir), "val_dir": str(args.val_dir)},
        )
    m0_train = np.asarray(train_data["action"], dtype=int)
    correction = threshold_grid(
        softmax(oof_logits), m0_train, train_y
    )
    results = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "no_mec_training": True,
            "final_test_accessed": False,
        },
        "config": {
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "input_dim": int(train_x.shape[1]),
        },
        "data": {
            "train_states": int(len(train_x)),
            "train_episodes": int(np.unique(train_data["episode_id"]).size),
            "val_states": int(len(val_x)),
            "val_episodes": int(np.unique(val_data["episode_id"]).size),
            "train_class_counts": {
                MOVEMENT_NAMES[int(code)]: int((train_y == code).sum())
                for code in range(3)
            },
            "val_class_counts": {
                MOVEMENT_NAMES[int(code)]: int((val_y == code).sum())
                for code in range(3)
            },
        },
        "gab": {
            "reliability": reliability,
            "config": {
                "alpha": config.alpha,
                "beta": config.beta,
                "forward_beta": config.forward_beta,
                "max_age": config.max_age,
                "bins": config.bins,
                "fov_deg": config.fov_deg,
                "turn_deg": config.turn_deg,
            },
            "feature_names": list(GAB_FEATURE_NAMES),
            "train_valid_rate": float(
                train_gab["valid"][train_raw["common_mask"]].mean()
            ),
            "val_valid_rate": float(
                val_gab["valid"][val_raw["common_mask"]].mean()
            ),
            "train_mean_age": float(
                train_gab["age"][train_raw["common_mask"]].mean()
            ),
            "val_mean_age": float(
                val_gab["age"][val_raw["common_mask"]].mean()
            ),
        },
        "oof": classification_metrics(
            train_y, oof_logits.argmax(axis=1), oof_logits
        ),
        "val": classification_metrics(
            val_y, val_logits.argmax(axis=1), val_logits
        ),
        "fold_info": fold_info,
        "correction_selection": correction,
    }
    (output_dir / "r3_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "r3_predictions.npz",
        oof_logits=oof_logits,
        val_logits=val_logits,
        val_pred=val_logits.argmax(axis=1),
        oof_pred=oof_logits.argmax(axis=1),
        episode_id=np.asarray(val_data["episode_id"]),
        band=np.asarray(val_data["band"]),
        step=np.asarray(val_data["step"]),
        oracle_move_action=val_y,
        m0_action=np.asarray(val_data["action"]),
        target_visible=np.asarray(val_data["target_visible"]),
        distance_m=np.asarray(val_data["distance_m"]),
        since_localizable=np.asarray(val_data["since_localizable"]),
        heading_error_deg=np.asarray(val_data["heading_error_deg"]),
        gab_features=val_gab["features"][val_raw["common_mask"]],
        gab_valid=val_gab["valid"][val_raw["common_mask"]],
        gab_age=val_gab["age"][val_raw["common_mask"]],
    )
    print(json.dumps({
        "output": str(output_dir),
        "input_dim": int(train_x.shape[1]),
        "gab": results["gab"],
        "oof": {
            "accuracy": results["oof"]["accuracy"],
            "balanced": results["oof"]["balanced_accuracy_present"],
        },
        "val": {
            "accuracy": results["val"]["accuracy"],
            "balanced": results["val"]["balanced_accuracy_present"],
        },
        "correction_selected": correction["selected"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
