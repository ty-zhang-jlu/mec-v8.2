"""Select R1/R2/R3 movement-correction thresholds on Train OOF only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    load_event_split,
)
from v83_spatial_stage2.stage2g5_data import movement_view  # noqa: E402
from v83_spatial_stage2.stage2h_r3_train import threshold_grid  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--stage2g5-probe-dir", type=Path, required=True)
    parser.add_argument("--r3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_raw = load_event_split(args.train_dir)
    train = movement_view(train_raw)
    labels = np.asarray(train["oracle_move_action"], dtype=int)
    m0 = np.asarray(train["action"], dtype=int)
    stage2g5 = np.load(args.stage2g5_probe_dir / "predictions.npz")
    r3 = np.load(args.r3_dir / "r3_predictions.npz")
    logits = {
        "R1": np.asarray(stage2g5["B1_oof_logits"]),
        "R2": np.asarray(stage2g5["B2_oof_logits"]),
        "R3": np.asarray(r3["oof_logits"]),
    }
    output = {
        "data": {
            "train_states": int(len(labels)),
            "m0_accuracy": float((m0 == labels).mean()),
        },
        "schemes": {},
    }
    from v83_spatial_stage2.stage2h_r3_train import softmax
    for name, values in logits.items():
        if len(values) != len(labels):
            raise RuntimeError(
                f"{name} OOF length {len(values)} != labels {len(labels)}"
            )
        selection = threshold_grid(softmax(values), m0, labels)
        output["schemes"][name] = selection
        selected = selection["selected"]
        output["schemes"][name]["selected_summary"] = {
            "threshold": selected["threshold"],
            "corrections": selected["corrections"],
            "helpful": selected["helpful"],
            "harmful": selected["harmful"],
            "net_agreement": selected["net_agreement"],
            "accuracy": selected["accuracy"],
            "base_accuracy": selected["base_accuracy"],
            "net_accuracy": selected["net_accuracy"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "schemes": {
            name: entry["selected_summary"]
            for name, entry in output["schemes"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
