# V8.3 Stage-2g.5 — Temporal Spatial Belief / Coarse Movement Diagnosis Protocol

Status: frozen before full probe training (2026-10-19).

Branch: `experiment/v83-spatial-foundation`

Base commit: `cca0fb3`

## 1. Scope

- Reuse `runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z/` and
  `real_val_20261019T132000Z/`.
- Train only lightweight movement diagnostic probes. Do **not** train MEC,
  M1-Dir, M2, Depth, or a controller; do not modify old checkpoints; do not
  read Final Test 300.
- No new trajectories are collected.

## 2. Spatial-map audit (frozen before training)

| Source | What exists | Alignment verdict |
|---|---|---|
| Stage-2a `cache_20261018T043434Z` | 24×24 baseline/coherent/temporal/innovation, `valid`, `(episode_id, step_id)` | **Not usable for real M0 closed-loop states.** Maps at step 0 match exactly (MSE 0), but after step 1 coherent MSE rises to ~0.18–0.28 and baseline MSE to ~0.005–0.009. The cache is built from reference/oracle trajectories, not the M0 rollout. |
| Stage-2e replay `replay_20261019T015500Z` | M0/M1 logits, scalar spatial statistics, 16/64 pooled conv features | M0 rows (26,628) align exactly with real Val; full 24×24 maps are **not stored**. |
| Stage-2f collect `collect_20261019T024000Z` | scalar heatmap/baseline/coherent statistics for M0/M1 Val | M0 rows align exactly (26,628); full 24×24 maps are **not stored**. |
| Stage-2g.3 real NPZ | 2,858 current features, including native 3×8×8 heatmap and 8×8 pooled baseline/coherent maps | **8×8 maps are reconstructable**; the original 24×24 maps are not recoverable. |
| Stage-2g.1 supplement | full 24×24 maps | Synthetic turn-away states, not real M0 closed-loop; not used. |

Consequences:

- B3 is explicitly **B3-8×8**: it uses the reconstructable 8×8 heatmap and
  8×8 pooled baseline/coherent maps. It does not claim 24×24 alignment.
- The 8×8 maps preserve left/right columns. B3's spatial branch uses a 2×2
  spatial pool, never a global 1×1 pool.
- Yaw alignment uses the same pinhole geometry as the verified
  `yaw_warp_previous`, applied at 8×8. Only the immediate previous state is
  used, and only when the previous action is LEFT/RIGHT. FORWARD history is
  marked invalid (no fixed-pixel warp) and B3 falls back to current evidence.
- The validity plane is an explicit input channel; invalid history is zero,
  not fabricated.

## 3. Data and labels

- Common set: `step >= 16` and `action != STOP`.
  - Train: 71,576 states / 563 episodes.
  - Dev Val: 23,665 states / 171 episodes.
- Oracle Movement label: `oracle_move_action` ∈ {FORWARD, LEFT, RIGHT}; used
  only as the offline 3-class target.
- No oracle action, true goal position, true distance, true heading, true
  visibility, or success is a model input.
- Visible/invisible, distance, heading error, since-localizable and bands are
  evaluation-only labels.
- Class distribution (common set):

| Split | FORWARD | LEFT | RIGHT |
|---|---:|---:|---:|
| Train | 35,929 | 17,270 | 18,377 |
| Dev Val | 6,435 | 8,250 | 8,980 |

- Previous-turn spatial validity: Train 46,899/71,576 (65.5%);
  Dev Val 14,385/23,665 (60.8%). Invalid states use current maps only.

## 4. Variants

| Variant | Input |
|---|---|
| B0 | frozen M0 saved movement action (argmax of M0 movement head), no training |
| B1 | current 2,858 observable features → 3-class MLP |
| B2 | current + 16-step compact history (74 compact + 4 action + dt per frame) → 3-class MLP |
| B3 | B2 tabular + 8×8 coarse temporal spatial belief (16 channels) → hybrid CNN/MLP |

B3 spatial channels: current heat_0/1/2, current base, current coherent;
previous-turn yaw-warped versions; blended belief; validity plane.
The spatial branch is Conv(16→16,3×3) → ReLU → Conv(16→16,3×3) → ReLU →
2×2 average pool → 64-d; it is concatenated with the 128-d tabular branch.

Training: 5-fold **episode-level** OOF; external model on all Train;
30 epochs, hidden 128, batch 8,192, Adam lr 1e-3 / weight decay 1e-4,
inverse-frequency class weights, seed 0 (+fold, +1000 for external).
No threshold selection: 3-class argmax.

## 5. Required evaluation

Overall and per band (Short/Medium/Long):

- Accuracy, balanced accuracy, macro-F1, 3×3 confusion, oracle agreement.
- Visible vs invisible subsets.
- Near-target invisible (`distance<100 & invisible`), far visible/invisible.
- Since-localizable 1–4 and 5–16 steps.
- Left/Right balanced recall and L-R AUC from `logit_LEFT - logit_RIGHT`;
  predicted turn rate vs oracle turn rate.
- Episode-level bootstrap 95% CI and paired episode bootstrap for
  B2−B1, B3−B1, B3−B2.
- All-model failure groups (where B1/B2/B3 are all wrong), with visibility,
  distance, since-localizable and band breakdown.
- B3 valid vs invalid temporal-map states, to test whether any gain is
  confined to valid spatial history.

## 6. Decision rule (frozen)

- **Pass (spatial belief adds value):**
  On Dev Val invisible states, B3 improves balanced accuracy or macro-F1 over
  both B1 and B2 by at least **+2 percentage points**, with paired
  episode-bootstrap 95% CI lower bound > 0; and B3 improves LEFT/RIGHT
  balanced recall by at least **+2 pp** over B2. If pass, report only a
  controlled M0 + Temporal Spatial Belief MEC ablation plan; do not train it.
- **Partial (history only):** B2 improves over B1 by at least +2 pp on
  invisible states, but B3 does not improve over B2 → long history is useful,
  current spatial-map fusion adds no value.
- **Fail (stop image-matching history):** neither B2 nor B3 shows a stable
  invisible-state gain over B1 → stop the image-matching history route and
  report the observational limit plus alternatives.
- All decisions use episode-level CIs; continuous steps are not treated as
  independent samples.

## 7. Compliance

- No MEC/M1-Dir/M2/Depth training.
- No controller integration.
- No old checkpoint/data/report modification.
- Final Test 300 sealed.
- Large maps/weights/predictions stay under `runs/`.
