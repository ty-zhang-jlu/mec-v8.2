"""Stage-2h closed-loop pilot: R0/R1/R2/R3 Movement corrections.

R0 is the frozen M0@0.7 baseline (reused only if seeds/bands exactly match).
R1/R2/R3 keep the original M0 Stop decision and only correct Movement actions
when the corresponding movement probe is confident enough. The correction
threshold is selected on Train OOF only.

No MEC training, no checkpoint modification, no Final Test access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV
from env import FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv, wrap_angle
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import static_grounding
from v83_spatial_stage2.stage2d_common import (
    STAGE2_ROOT,
    VAL_SPECS,
    build_oracle_cache,
    load_dev_specs,
    load_encoder,
)
from v83_spatial_stage2.stage2f_collect import (
    M0_CKPT,
    load_model,
    oracle_move_action,
    projection_vector,
)
from v83_spatial_stage2.stage2g1_common import (
    compact_indices,
    one_hot_actions,
)
from v83_spatial_stage2.stage2g_features import (
    compute_feature_vector,
    feature_names,
)
from v83_spatial_stage2.stage2g5_data import (
    reconstruct_current_maps,
)
from v83_spatial_stage2.stage2g5_probe import MovementMLP
from v83_spatial_stage2.stage2h_gab import GABConfig, GABState
from v83_spatial_training.eval_closedloop_spatial import paired_comparison


BANDS = ("short", "medium", "long")
STOP_THRESHOLD = 0.7
R0_CSV = (
    STAGE2_ROOT / "stage2d/r4_20261019T002626Z/threshold_0.7_per_episode.csv"
)
COMPACT_IDX = compact_indices(feature_names())


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_r0_rows(path=R0_CSV):
    with Path(path).open(encoding="utf-8") as handle:
        return {
            int(row["episode_index"]): row
            for row in csv.DictReader(handle)
        }


def check_r0_alignment(specs, r0_rows):
    mismatches = []
    for spec in specs:
        row = r0_rows.get(int(spec["episode_index"]))
        if row is None:
            mismatches.append({"episode_index": spec["episode_index"],
                               "error": "missing"})
            continue
        if int(row["seed"]) != int(spec["seed"]) or row["band"] != spec["band"]:
            mismatches.append({
                "episode_index": int(spec["episode_index"]),
                "seed": [int(spec["seed"]), int(row["seed"])],
                "band": [spec["band"], row["band"]],
            })
    return mismatches


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_movement_probe(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    dim = len(checkpoint["tabular_names"])
    hidden = int(checkpoint["config"]["hidden"])
    model = MovementMLP(dim, hidden)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(checkpoint["tabular_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["tabular_std"], dtype=np.float32),
        "variant": checkpoint["variant"],
        "input_dim": dim,
    }


def probe_probs(probe, tabular, device):
    x = (np.asarray(tabular, dtype=np.float32) - probe["mean"]) / probe["std"]
    x = np.clip(x, -20.0, 20.0)
    with torch.no_grad():
        logits = probe["model"](
            torch.from_numpy(x).unsqueeze(0).to(device), None
        ).cpu().numpy()[0]
    return softmax(logits), logits


def history_vector(past_compact, past_actions):
    horizon = 16
    history = np.zeros((horizon, len(COMPACT_IDX)), dtype=np.float32)
    actions = np.full(horizon, -1, dtype=np.int64)
    dt = np.zeros(horizon, dtype=np.float32)
    for offset in range(1, horizon + 1):
        if len(past_compact) >= offset:
            history[offset - 1] = past_compact[-offset]
            actions[offset - 1] = past_actions[-offset]
            dt[offset - 1] = 1.0
    one_hot = one_hot_actions(actions.reshape(-1)).reshape(horizon * 4)
    return history.reshape(-1), one_hot, dt


def movement_error_label(action, oracle):
    if action == oracle:
        return "correct"
    if oracle in (LEFT, RIGHT) and action == FORWARD:
        return "missed_turn"
    if oracle in (LEFT, RIGHT) and action in (LEFT, RIGHT):
        return "wrong_turn"
    if oracle == FORWARD and action in (LEFT, RIGHT):
        return "spurious_turn"
    return "other"


def run_episode(spec, scheme, threshold, probe, encoder, m0_model,
                device, oracle_info, gab_config):
    env = ToyUAVEnv()
    obs = env.reset(
        seed=spec["seed"],
        goal_distance_range_m=spec["distance_range"],
        render_observation=True,
    )
    nodes = build_random_gbs_layout(
        seed=spec["network_layout_seed"], verbose=False
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device, dtype=torch.float32)
    gab_state = GABState(gab_config) if scheme == "R3" else None
    past_compact = []
    past_actions = []
    previous_action = -1
    trajectory = [env.position.copy()]
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    rows = []

    for step in range(int(ENV.max_steps)):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype("float32")
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            baseline, coherent = static_grounding(current_tokens, goal_tokens)
            components = m0_model.base.forward_components(
                current_tokens, goal_tokens, heatmap, network
            )
            feature_tensor = compute_feature_vector(
                current_tokens, goal_tokens, heatmap, baseline, coherent
            )
            features = feature_tensor.squeeze(0).cpu().numpy()
            arrival_probability = float(
                torch.sigmoid(components["arrival_logit"]).item()
            )
            base_action = int(
                torch.argmax(components["move_logits"], -1).item()
            )
            projection = projection_vector(env)
            dx = float(env.goal[0] - env.x)
            dy = float(env.goal[1] - env.y)
            heading_error = wrap_angle(
                math.atan2(dy, dx) - env.yaw
            )
            oracle_action = oracle_move_action(env)

        current_compact = features[COMPACT_IDX]
        tabular_b2 = None
        gab_features = None
        if scheme in ("R2", "R3"):
            hist_features, hist_onehot, hist_dt = history_vector(
                past_compact, past_actions
            )
            tabular_b2 = np.concatenate([
                features, hist_features, hist_onehot, hist_dt
            ])
        if scheme == "R3":
            maps = reconstruct_current_maps(features[None])[0]
            gab_state.update(maps, previous_action)
            gab_features = gab_state.features()
            tabular_r3 = np.concatenate([tabular_b2, gab_features])
        else:
            tabular_r3 = None

        proposed = base_action
        confidence = 1.0
        corrected = False
        if arrival_probability >= STOP_THRESHOLD:
            action = STOP
        else:
            if scheme == "R1":
                tabular = features
            elif scheme == "R2":
                tabular = tabular_b2
            elif scheme == "R3":
                tabular = tabular_r3
            else:
                tabular = None
            if scheme != "R0" and step >= 16 and tabular is not None:
                probs, _ = probe_probs(probe, tabular, device)
                proposed = int(np.argmax(probs))
                confidence = float(np.max(probs))
                if proposed != base_action and confidence >= threshold:
                    action = proposed
                    corrected = True
                else:
                    action = base_action
            else:
                action = base_action

        helpful = int(
            corrected and proposed == oracle_action
            and base_action != oracle_action
        )
        harmful = int(
            corrected and proposed != oracle_action
            and base_action == oracle_action
        )
        distance = float(env.distance_to_goal)
        rows.append({
            "policy": scheme,
            "episode_index": int(spec["episode_index"]),
            "seed": int(spec["seed"]),
            "band": str(spec["band"]),
            "step": int(step),
            "distance_m": distance,
            "goal_visible": int(projection is not None),
            "goal_px": projection["goal_px"] if projection else float("nan"),
            "goal_py": projection["goal_py"] if projection else float("nan"),
            "heading_error_deg": math.degrees(heading_error),
            "oracle_move_action": int(oracle_action),
            "base_action": int(base_action),
            "proposed_action": int(proposed),
            "action": int(action),
            "probe_confidence": float(confidence),
            "corrected": int(corrected),
            "helpful": helpful,
            "harmful": harmful,
            "arrival_probability": arrival_probability,
            "gab_valid": int(gab_state.valid) if gab_state else 0,
            "gab_confidence": float(gab_state.confidence) if gab_state else 0.0,
            "gab_age": int(gab_state.last_seen_age) if gab_state else 0,
            "movement_label": (
                movement_error_label(int(action), int(oracle_action))
                if action != STOP else "stop"
            ),
        })

        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        trajectory.append(env.position.copy())
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        past_compact.append(current_compact)
        past_actions.append(int(action))
        previous_action = int(action)
        if done:
            break

    trajectory = np.asarray(trajectory, dtype=np.float32)
    path_length = (
        float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
        if len(trajectory) > 1 else 0.0
    )
    shortest = float(oracle_info["shortest_distance_m"])
    spl = shortest / max(shortest, path_length, 1e-8) if success else 0.0
    ndtw = ndtw_score(
        trajectory=trajectory[:, :2],
        reference=oracle_info["reference"],
        distance_threshold=env.stop_radius,
    )
    move_rows = [row for row in rows if row["action"] != STOP]
    corrections = [row for row in rows if row["corrected"]]
    episode = {
        "policy": scheme,
        "episode_index": int(spec["episode_index"]),
        "seed": int(spec["seed"]),
        "band": str(spec["band"]),
        "success": int(success),
        "osr": int(osr),
        "premature_stop": int(bool(ended_by_stop and not success)),
        "max_steps_termination": int(
            bool((not ended_by_stop) and (not success) and done)
        ),
        "steps": int(len(trajectory) - 1),
        "spl": float(spl),
        "ndtw": float(ndtw),
        "min_distance_m": float(min_distance),
        "final_distance_m": float(env.distance_to_goal),
        "reached_radius_not_success": int(osr and not success),
        "corrections": len(corrections),
        "helpful": sum(row["helpful"] for row in corrections),
        "harmful": sum(row["harmful"] for row in corrections),
        "net": (
            sum(row["helpful"] for row in corrections)
            - sum(row["harmful"] for row in corrections)
        ),
        "move_states": len(move_rows),
        "base_move_accuracy": (
            float(np.mean([
                row["base_action"] == row["oracle_move_action"]
                for row in move_rows
            ])) if move_rows else float("nan")
        ),
        "final_move_accuracy": (
            float(np.mean([
                row["action"] == row["oracle_move_action"]
                for row in move_rows
            ])) if move_rows else float("nan")
        ),
        "visible_move_accuracy": (
            float(np.mean([
                row["action"] == row["oracle_move_action"]
                for row in move_rows if row["goal_visible"]
            ])) if any(row["goal_visible"] for row in move_rows) else float("nan")
        ),
        "invisible_move_accuracy": (
            float(np.mean([
                row["action"] == row["oracle_move_action"]
                for row in move_rows if not row["goal_visible"]
            ])) if any(not row["goal_visible"] for row in move_rows) else float("nan")
        ),
    }
    return episode, rows


def summarize(episodes):
    if not episodes:
        return {"n": 0}
    return {
        "n": len(episodes),
        "sr": float(np.mean([row["success"] for row in episodes])),
        "osr": float(np.mean([row["osr"] for row in episodes])),
        "spl": float(np.mean([row["spl"] for row in episodes])),
        "ndtw": float(np.mean([row["ndtw"] for row in episodes])),
        "premature_stop": float(np.mean([
            row["premature_stop"] for row in episodes
        ])),
        "timeout": float(np.mean([
            row["max_steps_termination"] for row in episodes
        ])),
        "steps": float(np.mean([row["steps"] for row in episodes])),
        "corrections": float(np.mean([
            row.get("corrections", 0) for row in episodes
        ])),
        "helpful": float(np.mean([
            row.get("helpful", 0) for row in episodes
        ])),
        "harmful": float(np.mean([
            row.get("harmful", 0) for row in episodes
        ])),
        "net": float(np.mean([
            row.get("net", 0) for row in episodes
        ])),
        "base_move_accuracy": float(np.nanmean([
            row.get("base_move_accuracy", float("nan"))
            for row in episodes
        ])),
        "final_move_accuracy": float(np.nanmean([
            row.get("final_move_accuracy", float("nan"))
            for row in episodes
        ])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--r0-csv", type=Path, default=R0_CSV)
    parser.add_argument(
        "--schemes", nargs="+", default=["R0", "R1", "R2", "R3"]
    )
    parser.add_argument("--stage2g5-probe-dir", type=Path, required=True)
    parser.add_argument("--r3-dir", type=Path, required=True)
    parser.add_argument("--correction-json", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.smoke:
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    specs = load_dev_specs(args.val_specs, 180)
    if args.max_episodes is not None:
        specs = specs[: int(args.max_episodes)]
    r0_rows = load_r0_rows(args.r0_csv)
    mismatches = check_r0_alignment(specs, r0_rows)
    if mismatches:
        raise SystemExit(f"R0 CSV does not match specs: {mismatches[:5]}")
    correction = json.loads(Path(args.correction_json).read_text())
    thresholds = {
        name: float(entry["selected_threshold"])
        if "selected_threshold" in entry
        else float(entry["selected"]["threshold"])
        for name, entry in correction["schemes"].items()
    }
    r3_results = json.loads((args.r3_dir / "r3_results.json").read_text())
    gab_cfg_json = r3_results["gab"]
    gab_config = GABConfig(
        reliability_threshold=float(gab_cfg_json["reliability"]["threshold"]),
        alpha=float(gab_cfg_json["config"]["alpha"]),
        beta=float(gab_cfg_json["config"]["beta"]),
        forward_beta=float(gab_cfg_json["config"]["forward_beta"]),
        max_age=int(gab_cfg_json["config"]["max_age"]),
        bins=int(gab_cfg_json["config"]["bins"]),
        fov_deg=float(gab_cfg_json["config"]["fov_deg"]),
        turn_deg=float(gab_cfg_json["config"]["turn_deg"]),
    )
    probes = {
        "R1": load_movement_probe(
            args.stage2g5_probe_dir / "models" / "B1_external.pt"
        ),
        "R2": load_movement_probe(
            args.stage2g5_probe_dir / "models" / "B2_external.pt"
        ),
        "R3": load_movement_probe(args.r3_dir / "models" / "R3_external.pt"),
    }
    for probe in probes.values():
        probe["model"].to(device)
    print("loading encoder/M0/oracle", flush=True)
    encoder = load_encoder(device)
    m0_model = load_model("M0", M0_CKPT, device)
    oracle_cache = build_oracle_cache(specs)
    output = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "schemes": list(args.schemes),
            "episodes": len(specs),
            "max_episodes": args.max_episodes,
            "device": str(device),
            "stop_threshold": STOP_THRESHOLD,
            "correction_thresholds": thresholds,
            "gab": gab_cfg_json,
            "r0_csv": str(args.r0_csv),
            "no_mec_training": True,
            "final_test_accessed": False,
        },
        "r0_alignment_mismatches": mismatches,
        "episodes": {},
        "summaries": {},
        "bands": {},
        "paired": {},
    }
    rows_by_scheme = {}
    for scheme in args.schemes:
        print(f"running {scheme}", flush=True)
        episodes = []
        all_rows = []
        for position, spec in enumerate(specs):
            episode, step_rows = run_episode(
                spec, scheme, thresholds.get(scheme, 1.0),
                probes.get(scheme), encoder, m0_model, device,
                oracle_cache[int(spec["episode_index"])], gab_config,
            )
            episodes.append(episode)
            all_rows.extend(step_rows)
            if (position + 1) % 20 == 0:
                print(f"  {scheme}: {position + 1}/{len(specs)}", flush=True)
        rows_by_scheme[scheme] = episodes
        output["episodes"][scheme] = episodes
        output["summaries"][scheme] = summarize(episodes)
        output["bands"][scheme] = {
            band: summarize([
                row for row in episodes if row["band"] == band
            ])
            for band in BANDS
        }
        write_csv(
            args.output_dir / f"{scheme}_per_episode.csv", episodes
        )
        write_csv(
            args.output_dir / f"{scheme}_steps.csv", all_rows
        )
        print(
            json.dumps({
                "scheme": scheme,
                "summary": output["summaries"][scheme],
                "bands": output["bands"][scheme],
            }, indent=2),
            flush=True,
        )
    # R0 rows are loaded from CSV for full runs; convert for paired comparison.
    if "R0" not in rows_by_scheme and r0_rows:
        rows_by_scheme["R0"] = [
            {
                "episode_index": int(row["episode_index"]),
                "seed": int(row["seed"]),
                "band": row["band"],
                "success": int(row["success"]),
                "osr": int(row["osr"]),
                "premature_stop": int(row["premature_stop"]),
                "max_steps_termination": int(row["max_steps_termination"]),
                "steps": int(row["steps"]),
                "spl": float(row["spl"]),
                "ndtw": float(row["ndtw"]),
                "min_distance_m": float(row["min_distance_m"]),
            }
            for row in r0_rows.values()
            if int(row["episode_index"]) in {
                int(spec["episode_index"]) for spec in specs
            }
        ]
        output["episodes"]["R0"] = rows_by_scheme["R0"]
        output["summaries"]["R0"] = summarize(rows_by_scheme["R0"])
        output["bands"]["R0"] = {
            band: summarize([
                row for row in rows_by_scheme["R0"]
                if row["band"] == band
            ])
            for band in BANDS
        }
    for scheme in ("R1", "R2", "R3"):
        if scheme in rows_by_scheme and "R0" in rows_by_scheme:
            output["paired"][scheme] = paired_comparison(
                rows_by_scheme, base="R0", other=scheme
            )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "summaries": output["summaries"],
        "paired": output["paired"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
