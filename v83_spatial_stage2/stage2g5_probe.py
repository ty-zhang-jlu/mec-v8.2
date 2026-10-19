"""Stage-2g.5 movement diagnosis probe.

B0: frozen M0 saved movement action (no training).
B1: current 2,858 observable features -> F/L/R MLP.
B2: current + 16-step compact history -> F/L/R MLP.
B3: B2 tabular + 8x8 coarse temporal spatial belief -> hybrid CNN/MLP.

The B3 spatial branch is explicitly 8x8 because the original 24x24 maps are
not saved for real M0 closed-loop states. It never globally pools the map; a
2x2 spatial pool preserves left/right information. FORWARD history maps are
invalid (no yaw warp) and fall back to current evidence.

No MEC training, no checkpoint modification, no controller, Final Test sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import assign_episode_folds  # noqa: E402
from v83_spatial_stage2.stage2g5_data import (  # noqa: E402
    BAND_NAMES,
    MOVEMENT_NAMES,
    SPATIAL_CHANNEL_NAMES,
    SPATIAL_CHANNELS,
    class_weights,
    build_movement_inputs,
    load_event_split,
    movement_view,
    subset_masks,
)
from v83_spatial_stage2.stage2g5_metrics import (  # noqa: E402
    classification_metrics,
    subset_metrics,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


class MovementMLP(nn.Module):
    def __init__(self, tabular_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(tabular_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, tabular, spatial=None):
        return self.net(tabular)


class SpatialMovementNet(nn.Module):
    def __init__(self, tabular_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.tabular = nn.Sequential(
            nn.Linear(int(tabular_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden) + 64, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 3),
        )

    def forward(self, tabular, spatial):
        return self.head(torch.cat([
            self.tabular(tabular), self.spatial(spatial)
        ], dim=1))


def build_model(variant, tabular_dim, hidden, seed):
    torch.manual_seed(int(seed))
    if variant in ("B1", "B2"):
        return MovementMLP(tabular_dim, hidden)
    if variant == "B3":
        return SpatialMovementNet(tabular_dim, hidden)
    raise ValueError(variant)


def standardize_tabular(train_x, valid_x):
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (
        np.clip((train_x - mean) / std, -20.0, 20.0).astype(np.float32),
        np.clip((valid_x - mean) / std, -20.0, 20.0).astype(np.float32),
        mean, std,
    )


def standardize_spatial(train_x, valid_x):
    mean = train_x.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    # Keep the validity plane as a 0/1 indicator.
    mean[-1] = 0.0
    std[-1] = 1.0
    return (
        np.clip((train_x - mean[None, :, None, None]) / std[None, :, None, None],
                -20.0, 20.0).astype(np.float32),
        np.clip((valid_x - mean[None, :, None, None]) / std[None, :, None, None],
                -20.0, 20.0).astype(np.float32),
        mean, std,
    )


def train_one(model, train_tab, train_spatial, train_y, valid_tab,
              valid_spatial, weights, args, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    train_tab_t = torch.from_numpy(train_tab).to(device)
    train_spatial_t = (
        None if train_spatial is None
        else torch.from_numpy(train_spatial).to(device)
    )
    train_y_t = torch.from_numpy(train_y.astype(np.int64)).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    n = len(train_y_t)
    for _epoch in range(int(args.epochs)):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, int(args.batch_size)):
            indices = order[start:start + int(args.batch_size)]
            logits = model(
                train_tab_t[indices],
                None if train_spatial_t is None else train_spatial_t[indices],
            )
            loss = loss_fn(logits, train_y_t[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        valid_tab_t = torch.from_numpy(valid_tab).to(device)
        valid_spatial_t = (
            None if valid_spatial is None
            else torch.from_numpy(valid_spatial).to(device)
        )
        logits = model(valid_tab_t, valid_spatial_t).cpu().numpy()
    return logits


def save_checkpoint(path, model, tab_mean, tab_std, spatial_mean,
                    spatial_std, variant, fold, tab_names, spatial_names,
                    weights, config, data_dirs):
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "tabular_mean": np.asarray(tab_mean, dtype=np.float32),
        "tabular_std": np.asarray(tab_std, dtype=np.float32),
        "spatial_mean": (
            None if spatial_mean is None
            else np.asarray(spatial_mean, dtype=np.float32)
        ),
        "spatial_std": (
            None if spatial_std is None
            else np.asarray(spatial_std, dtype=np.float32)
        ),
        "variant": variant,
        "fold": int(fold),
        "tabular_names": list(tab_names),
        "spatial_names": (
            None if spatial_names is None else list(spatial_names)
        ),
        "class_weights": np.asarray(weights, dtype=np.float32),
        "config": dict(config),
        "data": dict(data_dirs),
        "git_commit": git_commit(),
        "spatial_note": (
            "8x8 coarse belief reconstructed from saved features; "
            "24x24 maps are not available for real M0 states"
            if variant == "B3" else None
        ),
    }
    torch.save(checkpoint, path)


def predict_logits(model, tab, spatial, device):
    model.eval()
    with torch.no_grad():
        tab_t = torch.from_numpy(tab).to(device)
        spatial_t = None if spatial is None else torch.from_numpy(spatial).to(device)
        logits = model(tab_t, spatial_t).cpu().numpy()
    return logits


def run_variant(variant, train_inputs, val_inputs, train_y, args,
                folds, device, output_dir, models_dir, data_dirs):
    train_tab = train_inputs["tabular"]
    val_tab = val_inputs["tabular"]
    train_spatial = train_inputs["spatial"]
    val_spatial = val_inputs["spatial"]
    tab_names = train_inputs["tabular_names"]
    spatial_names = train_inputs["spatial_names"]
    weights, counts = class_weights(train_y)
    oof_logits = np.zeros((len(train_tab), 3), dtype=np.float64)
    fold_info = {}
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std, tab_mean, tab_std = standardize_tabular(
            train_tab[train_mask], train_tab[valid_mask]
        )
        if train_spatial is not None:
            train_sp_std, valid_sp_std, sp_mean, sp_std = standardize_spatial(
                train_spatial[train_mask], train_spatial[valid_mask]
            )
        else:
            train_sp_std = valid_sp_std = None
            sp_mean = sp_std = None
        model = build_model(
            variant, train_tab.shape[1], args.hidden, args.seed + int(fold)
        )
        logits = train_one(
            model, train_std, train_sp_std, train_y[train_mask],
            valid_std, valid_sp_std, weights, args, device,
        )
        oof_logits[valid_mask] = logits
        fold_info[str(fold)] = {
            "train_states": int(train_mask.sum()),
            "valid_states": int(valid_mask.sum()),
            "train_class_counts": {
                MOVEMENT_NAMES[int(code)]: int((train_y[train_mask] == code).sum())
                for code in range(3)
            },
            "valid_class_counts": {
                MOVEMENT_NAMES[int(code)]: int((train_y[valid_mask] == code).sum())
                for code in range(3)
            },
            "class_weights": weights.tolist(),
        }
        if args.save_models:
            save_checkpoint(
                models_dir / f"{variant}_fold{fold}.pt",
                model, tab_mean, tab_std, sp_mean, sp_std,
                variant, fold, tab_names, spatial_names, weights,
                {
                    "hidden": int(args.hidden), "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size), "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                }, data_dirs,
            )
    train_std, val_std, tab_mean, tab_std = standardize_tabular(
        train_tab, val_tab
    )
    if train_spatial is not None:
        train_sp_std, val_sp_std, sp_mean, sp_std = standardize_spatial(
            train_spatial, val_spatial
        )
    else:
        train_sp_std = val_sp_std = None
        sp_mean = sp_std = None
    model = build_model(
        variant, train_tab.shape[1], args.hidden, args.seed + 1000
    )
    train_one(
        model, train_std, train_sp_std, train_y,
        val_std, val_sp_std, weights, args, device,
    )
    val_logits = predict_logits(model, val_std, val_sp_std, device)
    if args.save_models:
        save_checkpoint(
            models_dir / f"{variant}_external.pt",
            model, tab_mean, tab_std, sp_mean, sp_std,
            variant, -1, tab_names, spatial_names, weights,
            {
                "hidden": int(args.hidden), "epochs": int(args.epochs),
                "batch_size": int(args.batch_size), "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            }, data_dirs,
        )
    oof_pred = oof_logits.argmax(axis=1)
    val_pred = val_logits.argmax(axis=1)
    return {
        "variant": variant,
        "input_dim": int(train_tab.shape[1]),
        "spatial_channels": (
            None if train_spatial is None else int(train_spatial.shape[1])
        ),
        "class_weights": weights.tolist(),
        "train_class_counts": {
            MOVEMENT_NAMES[int(code)]: int((train_y == code).sum())
            for code in range(3)
        },
        "oof_logits": oof_logits,
        "val_logits": val_logits,
        "oof_pred": oof_pred,
        "val_pred": val_pred,
        "fold_info": fold_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["B1", "B2", "B3"])
    parser.add_argument("--folds", type=int, default=5)
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
    train_y = np.asarray(
        train_data["oracle_move_action"], dtype=np.int64
    )
    val_y = np.asarray(val_data["oracle_move_action"], dtype=np.int64)
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], args.folds, args.seed
    )
    data_dirs = {
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "train_manifest_sha256": sha256(Path(args.train_dir) / "manifest.json"),
        "val_manifest_sha256": sha256(Path(args.val_dir) / "manifest.json"),
    }
    base_pred = np.asarray(val_data["action"], dtype=int)
    base_metrics = classification_metrics(val_y, base_pred)
    base_subsets = {
        name: subset_metrics(val_data, mask, base_pred, None, name)
        for name, mask in subset_masks(val_data).items()
    }
    base_bands = {
        BAND_NAMES[int(code)]: subset_metrics(
            val_data, np.asarray(val_data["band"]) == code,
            base_pred, None, BAND_NAMES[int(code)],
        )
        for code in range(3)
    }
    results = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "no_mec_training": True,
            "no_controller": True,
            "final_test_accessed": False,
        },
        "config": {
            "variants": list(args.variants),
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "common_mask": "step>=16 and action!=STOP",
            "class_weight": "inverse class frequency",
            "spatial_belief": (
                "8x8 coarse maps reconstructed from saved features; "
                "24x24 real closed-loop maps unavailable"
            ),
        },
        "data": {
            "train_states": int(len(train_data["step"])),
            "train_episodes": int(np.unique(train_data["episode_id"]).size),
            "val_states": int(len(val_data["step"])),
            "val_episodes": int(np.unique(val_data["episode_id"]).size),
            "train_manifest_sha256": data_dirs["train_manifest_sha256"],
            "val_manifest_sha256": data_dirs["val_manifest_sha256"],
            "class_counts_train": {
                MOVEMENT_NAMES[int(code)]: int((train_y == code).sum())
                for code in range(3)
            },
            "class_counts_val": {
                MOVEMENT_NAMES[int(code)]: int((val_y == code).sum())
                for code in range(3)
            },
            "folds": {
                str(fold): int((folds == fold).sum())
                for fold in sorted(set(folds.tolist()))
            },
        },
        "B0": {
            "description": "frozen M0 saved movement action",
            "overall": base_metrics,
            "subsets": base_subsets,
            "bands": base_bands,
        },
        "variants": {},
    }
    train_history_cache = {}
    val_history_cache = {}
    predictions = {
        "episode_id": np.asarray(val_data["episode_id"]),
        "band": np.asarray(val_data["band"]),
        "step": np.asarray(val_data["step"]),
        "distance_m": np.asarray(val_data["distance_m"]),
        "target_visible": np.asarray(val_data["target_visible"]),
        "since_localizable": np.asarray(val_data["since_localizable"]),
        "heading_error_deg": np.asarray(val_data["heading_error_deg"]),
        "oracle_move_action": val_y,
        "m0_action": base_pred,
    }
    for variant in args.variants:
        print(f"building {variant}", flush=True)
        train_inputs = build_movement_inputs(
            train_raw, train_raw["common_mask"], variant, train_history_cache
        )
        val_inputs = build_movement_inputs(
            val_raw, val_raw["common_mask"], variant, val_history_cache
        )
        print(f"training {variant}", flush=True)
        result = run_variant(
            variant, train_inputs, val_inputs, train_y, args, folds,
            device, output_dir, models_dir, data_dirs,
        )
        entry = {
            "variant": variant,
            "input_dim": result["input_dim"],
            "spatial_channels": result["spatial_channels"],
            "class_weights": result["class_weights"],
            "train_class_counts": result["train_class_counts"],
            "oof": classification_metrics(
                train_y, result["oof_pred"], result["oof_logits"]
            ),
            "val": classification_metrics(
                val_y, result["val_pred"], result["val_logits"]
            ),
            "fold_info": result["fold_info"],
            "subsets": {
                name: subset_metrics(
                    val_data, mask, result["val_pred"],
                    result["val_logits"], name,
                )
                for name, mask in subset_masks(val_data).items()
            },
            "bands": {
                BAND_NAMES[int(code)]: subset_metrics(
                    val_data, np.asarray(val_data["band"]) == code,
                    result["val_pred"], result["val_logits"],
                    BAND_NAMES[int(code)],
                )
                for code in range(3)
            },
        }
        results["variants"][variant] = entry
        predictions[f"{variant}_oof_logits"] = result["oof_logits"]
        predictions[f"{variant}_val_logits"] = result["val_logits"]
        predictions[f"{variant}_oof_pred"] = result["oof_pred"]
        predictions[f"{variant}_val_pred"] = result["val_pred"]
        spatial_valid = (
            train_inputs["spatial"] is not None
            and train_inputs["spatial"].shape[1] == SPATIAL_CHANNELS
        )
        results["variants"][variant]["spatial_note"] = (
            "8x8 coarse spatial belief, 2x2 pooled; FORWARD history invalid"
            if spatial_valid else None
        )
    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    provenance = {
        "created_utc": results["meta"]["created_utc"],
        "git_commit": results["meta"]["git_commit"],
        "command": sys.argv,
        "data": data_dirs,
        "config": results["config"],
        "data_counts": results["data"],
        "models": sorted(
            str(path.relative_to(output_dir))
            for path in models_dir.glob("*.pt")
        ) if args.save_models else [],
        "spatial_channel_names": list(SPATIAL_CHANNEL_NAMES),
        "spatial_audit": {
            "full_24x24_available": False,
            "reason": (
                "Stage-2a 24x24 cache matches real M0 states only at step 0; "
                "coherent MSE grows to ~0.18-0.28 after step 1. The 8x8 "
                "heatmap and pooled base/coherent maps are saved in the "
                "2,858-d current features and are used here."
            ),
        },
    }
    (output_dir / "probe_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "data": results["data"],
        "B0": {
            "accuracy": base_metrics["accuracy"],
            "balanced_accuracy_present": base_metrics["balanced_accuracy_present"],
            "macro_f1_present": base_metrics["macro_f1_present"],
        },
        "variants": {
            name: {
                "accuracy": entry["val"]["accuracy"],
                "balanced_accuracy_present": entry["val"][
                    "balanced_accuracy_present"
                ],
                "macro_f1_present": entry["val"]["macro_f1_present"],
            }
            for name, entry in results["variants"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
