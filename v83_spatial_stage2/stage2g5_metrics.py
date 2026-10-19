"""Stage-2g.5 multiclass movement metrics and episode-level inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from v83_spatial_stage2.stage2g_probe import auc_rank
from v83_spatial_stage2.stage2g5_data import BAND_NAMES, MOVEMENT_NAMES


def confusion_matrix(y_true, y_pred, num_classes=3):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(
        np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    ):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true, pred] += 1
    return matrix


def classification_metrics(y_true, y_pred, logits=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    matrix = confusion_matrix(y_true, y_pred)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class = {}
    recalls = []
    f1s = []
    f1s_present = []
    for code, name in MOVEMENT_NAMES.items():
        tp = int(matrix[code, code])
        fn = int(support[code] - tp)
        fp = int(predicted[code] - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "support": int(support[code]),
            "predicted": int(predicted[code]),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if support[code] > 0:
            recalls.append(recall)
            f1s_present.append(f1)
        f1s.append(f1)
    present = support > 0
    result = {
        "n": int(n),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy_present": (
            float(np.mean(recalls)) if recalls else float("nan")
        ),
        "balanced_accuracy_all3": float(
            np.mean([
                per_class[MOVEMENT_NAMES[code]]["recall"]
                for code in range(3)
            ])
        ),
        "macro_f1_present": (
            float(np.mean(f1s_present)) if f1s_present else float("nan")
        ),
        "macro_f1_all3": float(np.mean(f1s)),
        "oracle_agreement": float((y_true == y_pred).mean()),
        "confusion": matrix.tolist(),
        "per_class": per_class,
        "support": {
            MOVEMENT_NAMES[code]: int(support[code]) for code in range(3)
        },
        "predicted": {
            MOVEMENT_NAMES[code]: int(predicted[code]) for code in range(3)
        },
        "oracle_turn_rate": float(
            np.mean(np.isin(y_true, [1, 2]))
        ),
        "predicted_turn_rate": float(
            np.mean(np.isin(y_pred, [1, 2]))
        ),
        "classes_present": [
            MOVEMENT_NAMES[code] for code in range(3) if present[code]
        ],
    }
    if logits is not None:
        logits = np.asarray(logits, dtype=float)
        lr_mask = np.isin(y_true, [1, 2])
        if lr_mask.any() and logits.shape[1] >= 3:
            score = logits[:, 1] - logits[:, 2]
            result["lr_auc"] = auc_rank(
                (y_true[lr_mask] == 1).astype(int), score[lr_mask]
            )
            result["lr_balanced_accuracy_at_0"] = float(np.mean(
                (score[lr_mask] > 0) == (y_true[lr_mask] == 1)
            ))
            result["lr_n"] = int(lr_mask.sum())
        else:
            result["lr_auc"] = float("nan")
            result["lr_balanced_accuracy_at_0"] = float("nan")
            result["lr_n"] = int(lr_mask.sum())
    return result


def subset_metrics(data, mask, y_pred, logits=None, label="all"):
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(data["oracle_move_action"], dtype=int)[mask]
    predictions = np.asarray(y_pred, dtype=int)[mask]
    subset_logits = None if logits is None else np.asarray(logits)[mask]
    metrics = classification_metrics(labels, predictions, subset_logits)
    metrics["subset"] = label
    return metrics


def per_episode_table(data, y_pred):
    episodes = np.asarray(data["episode_id"])
    labels = np.asarray(data["oracle_move_action"], dtype=int)
    rows = []
    for episode in np.unique(episodes):
        mask = episodes == episode
        rows.append({
            "episode_id": int(episode),
            "n": int(mask.sum()),
            "accuracy": float((labels[mask] == y_pred[mask]).mean()),
            "oracle_turn_rate": float(
                np.mean(np.isin(labels[mask], [1, 2]))
            ),
            "predicted_turn_rate": float(
                np.mean(np.isin(y_pred[mask], [1, 2]))
            ),
        })
    return rows


def _bootstrap_indices(episodes, rng):
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    sample = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([
        index_by_episode[int(episode)] for episode in sample
    ])


def episode_bootstrap(data, mask, y_pred, logits=None, n_boot=1000, seed=0):
    episodes = np.asarray(data["episode_id"])
    labels = np.asarray(data["oracle_move_action"], dtype=int)
    mask = np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(int(seed))
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        indices = _bootstrap_indices(episodes, rng)
        selected = indices[mask[indices]]
        if len(selected) == 0:
            continue
        y = labels[selected]
        pred = y_pred[selected]
        sub_logits = None if logits is None else logits[selected]
        metrics = classification_metrics(y, pred, sub_logits)
        draws["accuracy"].append(metrics["accuracy"])
        draws["balanced_accuracy_present"].append(
            metrics["balanced_accuracy_present"]
        )
        draws["macro_f1_present"].append(metrics["macro_f1_present"])
        if "lr_auc" in metrics and np.isfinite(metrics["lr_auc"]):
            draws["lr_auc"].append(metrics["lr_auc"])
        draws["predicted_turn_rate"].append(metrics["predicted_turn_rate"])
    output = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            output[name] = {
                "n_boot": int(len(values)),
                "mean": float(values.mean()),
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
            }
        else:
            output[name] = {"n_boot": 0}
    return output


def paired_bootstrap(data, mask, pred_a, logits_a, pred_b, logits_b,
                     n_boot=1000, seed=0):
    episodes = np.asarray(data["episode_id"])
    labels = np.asarray(data["oracle_move_action"], dtype=int)
    mask = np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(int(seed))
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        indices = _bootstrap_indices(episodes, rng)
        selected = indices[mask[indices]]
        if len(selected) == 0:
            continue
        y = labels[selected]
        ma = classification_metrics(
            y, pred_a[selected],
            None if logits_a is None else logits_a[selected],
        )
        mb = classification_metrics(
            y, pred_b[selected],
            None if logits_b is None else logits_b[selected],
        )
        draws["accuracy"].append(mb["accuracy"] - ma["accuracy"])
        draws["balanced_accuracy_present"].append(
            mb["balanced_accuracy_present"]
            - ma["balanced_accuracy_present"]
        )
        draws["macro_f1_present"].append(
            mb["macro_f1_present"] - ma["macro_f1_present"]
        )
        if (
            "lr_auc" in ma and "lr_auc" in mb
            and np.isfinite(ma["lr_auc"]) and np.isfinite(mb["lr_auc"])
        ):
            draws["lr_auc"].append(mb["lr_auc"] - ma["lr_auc"])
    output = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            output[name] = {
                "n_boot": int(len(values)),
                "mean": float(values.mean()),
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "fraction_le_0": float((values <= 0).mean()),
            }
        else:
            output[name] = {"n_boot": 0}
    return output


def all_model_failure(data, predictions, base_pred=None):
    labels = np.asarray(data["oracle_move_action"], dtype=int)
    variants = [name for name, pred in predictions.items() if pred is not None]
    wrong = np.stack([
        np.asarray(predictions[name]) != labels for name in variants
    ], axis=0)
    all_wrong = wrong.all(axis=0)
    base_wrong = (
        None if base_pred is None else (np.asarray(base_pred) != labels)
    )
    masks = {
        "visible": np.asarray(data["target_visible"]) == 1,
        "invisible": np.asarray(data["target_visible"]) == 0,
        "near_invisible": (
            (np.asarray(data["distance_m"]) < 100.0)
            & (np.asarray(data["target_visible"]) == 0)
        ),
        "far_invisible": (
            (np.asarray(data["distance_m"]) >= 100.0)
            & (np.asarray(data["target_visible"]) == 0)
        ),
    }
    for low, high in ((0, 20), (20, 50), (50, 100), (100, 200),
                      (200, 400), (400, 10000)):
        value = f"distance_{low}_{high}"
        masks[value] = (
            (np.asarray(data["distance_m"]) >= low)
            & (np.asarray(data["distance_m"]) < high)
        )
    since = np.asarray(data["since_localizable"])
    masks["since_1_4"] = (since >= 1) & (since <= 4)
    masks["since_5_16"] = (since >= 5) & (since <= 16)
    masks["since_never_or_gt16"] = since < 0
    for code, name in BAND_NAMES.items():
        masks[f"band_{name}"] = np.asarray(data["band"]) == code
    output = {
        "variants": variants,
        "all_wrong": int(all_wrong.sum()),
        "all_wrong_rate": float(all_wrong.mean()),
        "base_also_wrong": (
            None if base_wrong is None
            else int((all_wrong & base_wrong).sum())
        ),
        "groups": {},
    }
    for name, mask in masks.items():
        count = int(mask.sum())
        failures = int((all_wrong & mask).sum())
        output["groups"][name] = {
            "n": count,
            "all_wrong": failures,
            "rate": failures / max(count, 1),
        }
    return output
