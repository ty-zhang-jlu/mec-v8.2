"""Stage-2h closed-loop R0/R1/R2/R3 analysis and decision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_training.eval_closedloop_spatial import paired_comparison


BANDS = ("short", "medium", "long")
DEFAULT_R0_CSV = (
    "runs/v83_spatial_stage2/stage2d/r4_20261019T002626Z/"
    "threshold_0.7_per_episode.csv"
)


def summarize_rows(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "sr": float(np.mean([row["success"] for row in rows])),
        "osr": float(np.mean([row["osr"] for row in rows])),
        "spl": float(np.mean([row["spl"] for row in rows])),
        "ndtw": float(np.mean([row["ndtw"] for row in rows])),
        "premature_stop": float(np.mean([
            row["premature_stop"] for row in rows
        ])),
        "timeout": float(np.mean([
            row["max_steps_termination"] for row in rows
        ])),
        "steps": float(np.mean([row["steps"] for row in rows])),
    }


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value):
    return int(float(value))


def movement_stats(rows):
    move = [row for row in rows if as_int(row["action"]) != 3]
    corrected = [row for row in move if as_int(row["corrected"])]
    def accuracy(subset):
        return (
            float(np.mean([
                as_int(row["action"]) == as_int(row["oracle_move_action"])
                for row in subset
            ])) if subset else float("nan")
        )
    def base_accuracy(subset):
        return (
            float(np.mean([
                as_int(row["base_action"]) == as_int(row["oracle_move_action"])
                for row in subset
            ])) if subset else float("nan")
        )
    visible = [row for row in move if as_int(row["goal_visible"])]
    invisible = [row for row in move if not as_int(row["goal_visible"])]
    corr_visible = [
        row for row in corrected if as_int(row["goal_visible"])
    ]
    corr_invisible = [
        row for row in corrected if not as_int(row["goal_visible"])
    ]
    return {
        "move_states": len(move),
        "base_accuracy": base_accuracy(move),
        "final_accuracy": accuracy(move),
        "visible_states": len(visible),
        "visible_base_accuracy": base_accuracy(visible),
        "visible_final_accuracy": accuracy(visible),
        "invisible_states": len(invisible),
        "invisible_base_accuracy": base_accuracy(invisible),
        "invisible_final_accuracy": accuracy(invisible),
        "corrections": len(corrected),
        "helpful": int(sum(as_int(row["helpful"]) for row in corrected)),
        "harmful": int(sum(as_int(row["harmful"]) for row in corrected)),
        "net": int(
            sum(as_int(row["helpful"]) - as_int(row["harmful"])
                for row in corrected)
        ),
        "corrections_visible": len(corr_visible),
        "corrections_invisible": len(corr_invisible),
        "helpful_visible": int(sum(as_int(row["helpful"]) for row in corr_visible)),
        "harmful_visible": int(sum(as_int(row["harmful"]) for row in corr_visible)),
        "helpful_invisible": int(sum(as_int(row["helpful"]) for row in corr_invisible)),
        "harmful_invisible": int(sum(as_int(row["harmful"]) for row in corr_invisible)),
    }


def episode_rows(metrics, scheme):
    rows = metrics["episodes"][scheme]
    return {
        as_int(row["episode_index"]): row for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--stage2g5-probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r0-csv", type=Path, default=Path(DEFAULT_R0_CSV))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.eval_dir / "metrics.json").read_text())
    analysis = {
        "meta": {
            "eval_dir": str(args.eval_dir),
            "episodes": metrics["meta"]["episodes"],
            "correction_thresholds": metrics["meta"]["correction_thresholds"],
            "gab": metrics["meta"]["gab"],
            "final_test_accessed": False,
        },
        "overall": {},
        "bands": {},
        "paired_vs_R0": metrics.get("paired", {}),
        "paired_R3_vs_R2": {},
        "movement": {},
        "r0_movement": {},
        "decision": {},
    }
    for scheme in ("R0", "R1", "R2", "R3"):
        if scheme in metrics["summaries"]:
            analysis["overall"][scheme] = metrics["summaries"][scheme]
        if scheme in metrics["bands"]:
            analysis["bands"][scheme] = metrics["bands"][scheme]
    rows_by_scheme = {
        scheme: episode_rows(metrics, scheme)
        for scheme in ("R1", "R2", "R3")
        if scheme in metrics["episodes"]
    }
    if "R0" in metrics["episodes"]:
        r0_rows = {
            as_int(row["episode_index"]): {
                "episode_index": as_int(row["episode_index"]),
                "band": row["band"],
                "success": as_int(row["success"]),
                "osr": as_int(row["osr"]),
                "premature_stop": as_int(row["premature_stop"]),
                "max_steps_termination": as_int(row["max_steps_termination"]),
                "steps": as_int(row["steps"]),
                "spl": float(row["spl"]),
                "ndtw": float(row["ndtw"]),
                "min_distance_m": float(row["min_distance_m"]),
            }
            for row in metrics["episodes"]["R0"]
        }
    else:
        r0_rows = {
            as_int(row["episode_index"]): {
                "episode_index": as_int(row["episode_index"]),
                "band": row["band"],
                "success": as_int(row["success"]),
                "osr": as_int(row["osr"]),
                "premature_stop": as_int(row["premature_stop"]),
                "max_steps_termination": as_int(row["max_steps_termination"]),
                "steps": as_int(row["steps"]),
                "spl": float(row["spl"]),
                "ndtw": float(row["ndtw"]),
                "min_distance_m": float(row["min_distance_m"]),
            }
            for row in read_csv(args.r0_csv)
        }
    rows_by_scheme["R0"] = r0_rows
    analysis["overall"]["R0"] = summarize_rows(
        [row for row in r0_rows.values()
         if row["episode_index"] in {
             as_int(item["episode_index"])
             for item in metrics["episodes"].get(
                 "R1", [{"episode_index": row["episode_index"]}]
             )
         }]
    )
    analysis["bands"]["R0"] = {
        band: summarize_rows([
            row for row in r0_rows.values() if row["band"] == band
        ]) for band in BANDS
    }
    rows_by_scheme_lists = {
        name: list(rows.values()) for name, rows in rows_by_scheme.items()
    }
    analysis["paired_vs_R0"] = {
        scheme: paired_comparison(
            rows_by_scheme_lists, base="R0", other=scheme
        )
        for scheme in ("R1", "R2", "R3")
        if scheme in rows_by_scheme_lists
    }
    if "R2" in rows_by_scheme_lists and "R3" in rows_by_scheme_lists:
        analysis["paired_R3_vs_R2"] = paired_comparison(
            rows_by_scheme_lists, base="R2", other="R3"
        )
    for scheme in ("R1", "R2", "R3"):
        step_path = args.eval_dir / f"{scheme}_steps.csv"
        if step_path.is_file():
            analysis["movement"][scheme] = movement_stats(
                read_csv(step_path)
            )
    # R0 movement reference from Stage-2g.5 B0 metrics.
    b0_path = args.stage2g5_probe_dir / "probe_results.json"
    if b0_path.is_file():
        probe = json.loads(b0_path.read_text(encoding="utf-8"))
        analysis["r0_movement"] = probe["B0"]

    # Frozen decision rule using SR/SPL/nDTW and paired CIs.
    def summary(scheme):
        return analysis["overall"].get(scheme, {})
    def paired(base, other):
        if base == "R0":
            return analysis.get("paired_vs_R0", {}).get(other, {})
        return analysis.get("paired_R3_vs_R2", {}) if (
            base == "R2" and other == "R3"
        ) else {}
    def delta(entry, metric):
        if metric == "sr":
            return entry.get("success", {}).get("delta_rate")
        if metric == "spl":
            return entry.get("spl_delta")
        if metric == "ndtw":
            return entry.get("ndtw_delta")
        return None
    r0, r1, r2, r3 = (summary(name) for name in ("R0", "R1", "R2", "R3"))
    p_r2 = paired("R0", "R2")
    p_r3 = paired("R0", "R3")
    p_r3_r2 = paired("R2", "R3")
    def describe(pair, metric):
        key = {
            "sr": ("success", "delta_ci95"),
            "spl": ("spl", "spl_delta_ci95"),
            "ndtw": ("ndtw", "ndtw_delta_ci95"),
        }[metric]
        if metric == "sr":
            delta_value = pair.get("success", {}).get("delta_rate")
            ci = pair.get("success", {}).get("delta_ci95")
        else:
            delta_value = pair.get(f"{metric}_delta")
            ci = pair.get(f"{metric}_delta_ci95")
        return {"delta": delta_value, "ci95": ci}
    evidence = {
        "R2_vs_R0": {
            metric: describe(p_r2, metric)
            for metric in ("sr", "spl", "ndtw")
        },
        "R3_vs_R0": {
            metric: describe(p_r3, metric)
            for metric in ("sr", "spl", "ndtw")
        },
        "R3_vs_R2": {
            metric: describe(p_r3_r2, metric)
            for metric in ("sr", "spl", "ndtw")
        },
    }
    def positive(entry):
        delta_value = entry.get("delta")
        ci = entry.get("ci95")
        return (
            delta_value is not None and delta_value > 0
            and ci is not None and ci[0] is not None and ci[0] > 0
        )
    def nondegraded(entry):
        ci = entry.get("ci95")
        return ci is None or ci[1] is None or ci[1] >= -0.01
    r2_gain = any(
        positive(evidence["R2_vs_R0"][metric])
        for metric in ("sr", "spl")
    )
    r3_over_r2 = any(
        positive(evidence["R3_vs_R2"][metric])
        for metric in ("sr", "spl")
    ) and nondegraded(evidence["R3_vs_R2"]["ndtw"])
    r3_over_r0 = any(
        positive(evidence["R3_vs_R0"][metric])
        for metric in ("sr", "spl")
    ) and nondegraded(evidence["R3_vs_R0"]["ndtw"])
    if r3_over_r2 and r3_over_r0:
        decision = (
            "R3 passes the SR/SPL gain and nDTW non-degradation check; "
            "report a controlled M0 + Temporal Spatial Belief MEC ablation "
            "plan only."
        )
    elif r2_gain and not r3_over_r2:
        decision = (
            "R2 improves over R0 but R3 does not improve over R2: keep the "
            "history branch as a diagnostic, stop the current GAB design."
        )
    elif not r2_gain and not r3_over_r0:
        decision = (
            "No clear adapter gain: stop this adapter route; do not start "
            "full backbone retraining."
        )
    else:
        decision = (
            "Evidence is mixed/insufficient (small deltas or wide CIs); do "
            "not claim a gain and do not start full backbone retraining."
        )
    analysis["decision"] = {
        "evidence": evidence,
        "r2_gain": bool(r2_gain),
        "r3_over_r2": bool(r3_over_r2),
        "r3_over_r0": bool(r3_over_r0),
        "decision": decision,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "overall": {
            name: {
                key: analysis["overall"][name].get(key)
                for key in ("sr", "osr", "spl", "ndtw", "premature_stop",
                            "timeout", "steps", "corrections", "helpful",
                            "harmful", "net")
            }
            for name in ("R0", "R1", "R2", "R3")
            if name in analysis["overall"]
        },
        "movement": analysis["movement"],
        "decision": analysis["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
