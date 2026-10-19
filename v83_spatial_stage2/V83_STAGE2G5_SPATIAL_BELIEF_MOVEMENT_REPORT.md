# V8.3 Stage-2g.5 — Temporal Spatial Belief / Coarse Movement Diagnosis

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

前序提交：`cca0fb3`
本轮性质：**真实 M0 闭环数据上的轻量 Movement 诊断；不训练 MEC、不接控制器、不读 Final Test**

---

## 0. TL;DR（中文结论）

1. **长历史（B2）确实改善 Movement 方向预测，但改善不在近目标不可见状态。**
   Dev Val 总体：B0 M0 基线 accuracy **0.4402** / balanced **0.4431**；
   B1 **0.3914** / **0.4054**；B2 **0.4545** / **0.4652**；
   B3 **0.4359** / **0.4482**。
   目标不可见子集：B1 balanced **0.323** → B2 **0.403**
   （+7.97 pp，Episode 配对 95% CI `[+0.62, +17.66]`）。
   但近目标不可见（<100 m 且不可见）：
   B1 **0.260** → B2 **0.235**（**没有改善**，CI 跨 0）。
2. **B2 的增益集中在“远距/持续不可见”和“最近可靠观测 1–4 步”**：
   - 不可见且 `since_localizable` 1–4：B1 balanced 0.436 → B2 **0.512**，
     L/R AUC 0.589 → **0.808**；
   - 不可见且 `since_localizable` 5–16：B1 0.202 → B2 0.243；
   - 不可见且 never/>16：B1 0.319 → B2 0.400（但 L/R AUC 几乎不变）。
   近目标不可见 + since 1–4（n=82）有改善（B2 balanced 0.277，
   L/R AUC 0.741 vs B1 0.535），但样本少；近目标不可见 + since 5–16
   （n=246）没有改善（B2 0.239 vs B1 0.244，L/R AUC 均为 0.475）。
3. **B3 的 8×8 coarse temporal spatial belief 没有额外价值。**
   B3 相对 B2：总体 balanced **-1.72 pp**（95% CI `[-4.76, +1.25]`），
   不可见 balanced **-1.32 pp**（`[-7.31, +4.08]`），
   L/R AUC **-1.54 pp**。即使只看 temporal map 有效的状态，
   B3 balanced **0.397** 仍低于 B2 **0.421**；无效状态也低于 B2
   （0.533 vs 0.547）。B3 的 Train OOF accuracy 0.579 高于 B2 0.563，
   但 Dev Val 更低（0.436 vs 0.455），呈现过拟合/负迁移。
4. **B3 没有真正改善 Left/Right。**
   总体 L/R AUC：B1 0.673、B2 0.696、B3 0.688；
   不可见子集：B1 0.597、B2 0.608、B3 0.594；
   近目标不可见：B1 0.488、B2 0.422、B3 0.408。
   B2 的增益主要来自最近丢失（since 1–4）的 L/R 方向；
   B2 的预测转向率 **0.581** 仍低于 oracle 的 **0.728**，
   不是简单“多转向”。
5. **所有模型都无法预测的状态有明确画像。**
   B1/B2/B3 全部错误的状态占 **41.9%**（9,925/23,665）；
   近目标不可见中 **58.7%** 全部错误，`since 5–16` 中 **67.7%** 全部错误。
   这些状态的可见率 54.8%（全对状态 66.7%）、平均 |heading error|
   60.4°（全对 50.3°）、never-localizable 率 35.9%（全对 16.0%）；
   视觉相似度并不低（`sim_max_all_top5` 0.926 vs 0.903），
   说明失败主要是**方向观测缺失/左右歧义**，不是单纯低置信度。
6. **明确决策：PARTIAL。**
   - 长 16 步历史（B2）对远距/持续不可见和最近丢失（1–4 步）的 Movement
     有稳定增益，值得保留为诊断/特征；
   - **当前图像匹配空间图融合（B3-8×8）没有额外价值，停止这条融合路线**；
   - 近目标不可见整体仍未改善，不能据此训练 MEC 或接 Recovery；
   - 完整 24×24 B3 **无法从现有产物验证**（见 §1），不能把 B3-8×8 的失败
     等同于“24×24 一定失败”；若要继续，需对已保存 M0 状态做增量重渲染
     得到真正的 24×24 图，再做同一协议检验，而不是继续堆 MLP/历史帧。

---

## 1. 空间图与标签对齐审计

### 1.1 可用产物（观测）

| 来源 | 内容 | 与真实 M0 状态的对齐结论 |
|---|---|---|
| Stage-2a `cache_20261018T043434Z` | 17,234 val / 55,565 train 行；24×24 baseline/coherent/temporal/innovation + `valid` + `(episode_id, step_id)` | **不可直接复用。** step 0 完全匹配（coherent/base MSE=0），step 1 起 coherent MSE 约 **0.18–0.28**、base MSE 约 **0.005–0.009**；cache 来自 reference/oracle 轨迹，而真实状态来自 M0 闭环。 |
| Stage-2e `replay_20261019T015500Z` | M0/M1 logits、标量空间统计、16/64 维 pooled conv 特征 | M0 26,628 行与真实 Dev Val **精确对齐**；**没有 24×24 图**。 |
| Stage-2f `collect_20261019T024000Z` | Val M0/M1 的标量 heatmap/baseline/coherent 统计、M0 logits | M0 26,628 行精确对齐；**没有 24×24 图**。 |
| Stage-2g.3 真实 NPZ | 2,858 当前可观测特征；包含原生 3×8×8 heatmap、8×8 pooled base/coherent | **可重建 8×8 图**；原始 24×24 不可恢复。 |
| Stage-2g.1 supplement | 完整 24×24 图 | 合成 turn-away 状态，不是真实 M0 闭环；不使用。 |

Stage-2f 对齐核查：M0 rows = 26,628，`(episode_index, step)` 与 Stage-2g.3
Dev Val 完全重合；`oracle_move_action` 逐状态一致；`owner_move_action` 仅
在 150 个 STOP 状态上不同（STOP 不进入 Movement 评测）。

### 1.2 结论与 B3 的定义（审计）

- **完整 24×24 B3 在本轮不可验证**：真实 M0 状态的 24×24 图没有保存，
  Stage-2a cache 又无法在 step≥1 对齐。不会用错误对齐的 cache，
  也不会把 8×8 上采样后声称“24×24”。
- **B3-8×8** 使用：
  - 原生 3×8×8 Goal-Current heatmap；
  - 8×8 pooled base/coherent maps；
  - 当前 + 上一帧（仅 LEFT/RIGHT 时）yaw-aligned map；
  - 0.7/0.3 temporal blend；
  - 显式 validity plane。
- 8×8 保留左右列；B3 的空间分支使用 **2×2 pool**，不是 global 1×1 pool。
- FORWARD 历史不 warp；标记 invalid，B3 回退当前证据。
- Dev Val temporal-map 有效率 **60.8%**（14,385/23,665）；Train 65.5%。
- 这是“coarse spatial belief”，不是完整 24×24 结论；报告在 §10 明确限制。

---

## 2. 数据与监督

公共 Movement 集：`step >= 16` 且 `action != STOP`。

| Split | States | Episodes | FORWARD | LEFT | RIGHT |
|---|---:|---:|---:|---:|---:|
| Train | 71,576 | 563 | 35,929 | 17,270 | 18,377 |
| Dev Val | 23,665 | 171 | 6,435 | 8,250 | 8,980 |

- 标签：`oracle_move_action` ∈ {F,L,R}；只做离线 3-class 监督。
- 输入：当前 2,858 可观测特征（B1）；16 步 compact 历史（B2）；
  B2 + 8×8 temporal spatial belief（B3）。
- **禁止输入**：oracle action、真实目标坐标、距离、heading、可见性、
  success。visible/distance/heading/since 只用于事后分组。
- 训练：5-fold Episode OOF + 全部 Train 的 external；30 epochs，
  hidden 128，batch 8,192，Adam 1e-3 / wd 1e-4，inverse-frequency class
  weights，seed 0。
- B0：M0 保存的 movement argmax，不训练。
- 所有 B1/B2/B3 权重保存；没有训练 MEC/M1-Dir/M2/Depth。

---

## 3. 总体结果（Dev Val，统计）

| 模型 | Accuracy | Balanced Acc | Macro-F1 | Pred turn | Oracle turn | L/R AUC |
|---|---:|---:|---:|---:|---:|---:|
| B0（冻结 M0） | **0.4402** | 0.4431 | 0.4401 | 0.609 | 0.728 | — |
| B1（当前特征） | 0.3914 | 0.4054 | 0.3914 | 0.506 | 0.728 | 0.673 |
| B2（16 步历史） | **0.4545** | **0.4652** | **0.4531** | 0.581 | 0.728 | **0.696** |
| B3（+8×8 spatial） | 0.4359 | 0.4482 | 0.4363 | 0.545 | 0.728 | 0.688 |

**Confusion（rows=true F/L/R；cols=pred F/L/R）：**

B0：

```text
[[3030, 1358, 2047],
 [3223, 3621, 1406],
 [2989, 2224, 3767]]
```

B2：

```text
[[3531,  943, 1961],
 [2984, 4289,  977],
 [3396, 2648, 2936]]
```

B3：

```text
[[3598,  933, 1904],
 [3406, 3791, 1053],
 [3767, 2287, 2926]]
```

- B0 在 visible 上明显更强（B0 .557 vs B2 .488），但在 invisible 上明显更弱
  （B0 .253 vs B2 .403）。
- B2 总体略高于 B0（+1.4 pp accuracy / +2.2 pp balanced），
  但这是 visible/invisible 的**互补权衡**，不是全面超越。
  使用真实可见性做 oracle routing（B0 visible + B2 invisible）的
  上限仅 **0.4968** accuracy；可见性本身不可直接获得。
- B3 总体低于 B2；OOF 更高（B3 .579 vs B2 .563）但 Val 更低，
  说明空间分支过拟合/负迁移。

---

## 4. 目标可见 vs 不可见（统计）

| 子集 | 模型 | n | Accuracy | Balanced Acc | Macro-F1 | Pred turn | L/R AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| visible | B0 | 14,573 | **0.557** | **0.575** | **0.559** | 0.670 | — |
| visible | B1 | 14,573 | 0.435 | 0.412 | 0.417 | 0.450 | 0.761 |
| visible | B2 | 14,573 | 0.488 | 0.480 | 0.483 | 0.500 | 0.805 |
| visible | B3 | 14,573 | 0.466 | 0.450 | 0.457 | 0.460 | **0.816** |
| invisible | B0 | 9,092 | 0.253 | 0.253 | 0.333 | 0.520 | — |
| invisible | B1 | 9,092 | 0.322 | 0.323 | 0.402 | 0.600 | 0.597 |
| invisible | B2 | 9,092 | **0.400** | **0.403** | **0.463** | 0.700 | **0.608** |
| invisible | B3 | 9,092 | 0.387 | 0.390 | 0.457 | 0.680 | 0.594 |

- **B2 相对 B1 在 invisible 上 balanced +7.97 pp**
  （95% CI `[+0.62, +17.66]`；macro-F1 +6.16 pp，CI `[-0.63, +15.02]`）。
  这是一个中等证据的增益，不是压倒性。
- B3 在 invisible 上不如 B2（-1.32 pp balanced，CI 跨 0）；
  即便 visible L/R AUC 略高（0.816 vs 0.805），这不是整体增益。
- B2 的 invisible turn rate 0.700 仍低于 oracle 1.000；B1 0.600。
  B2 不是简单“全部转向”，而是部分恢复 L/R（尤其 left）。

### 4.1 Short / Medium / Long（统计）

| Band | B0 acc/bal/mf1 | B1 | B2 | B3 |
|---|---|---|---|---|
| short | 0.411 / 0.409 / 0.413 | 0.390 / 0.388 / 0.386 | **0.484 / 0.469 / 0.468** | 0.470 / 0.461 / 0.457 |
| medium | 0.432 / 0.455 / 0.433 | 0.437 / 0.467 / 0.435 | **0.470 / 0.503 / 0.473** | 0.449 / 0.482 / 0.451 |
| long | **0.476 / 0.482 / 0.476** | 0.346 / 0.362 / 0.341 | 0.410 / 0.423 / 0.408 | 0.390 / 0.399 / 0.392 |

B2 在 short/medium 最好；B0 在 long 仍最好；B3 没有超过 B2。

### 4.2 距离段（统计）

| Distance | B0 acc/bal | B1 acc/bal | B2 acc/bal | B3 acc/bal |
|---|---:|---:|---:|---:|
| 0–20 m (n=87) | 0.460 / 0.599 | 0.460 / 0.404 | 0.621 / 0.609 | **0.644 / 0.613** |
| 20–50 m (n=1,237) | **0.733 / 0.749** | 0.479 / 0.516 | 0.703 / 0.711 | 0.697 / 0.698 |
| 50–100 m (n=3,542) | **0.553 / 0.578** | 0.376 / 0.396 | 0.455 / 0.471 | 0.445 / 0.451 |
| 100–200 m (n=7,209) | **0.471 / 0.475** | 0.425 / 0.417 | 0.451 / 0.446 | 0.428 / 0.421 |
| 200–400 m (n=9,004) | 0.404 / 0.432 | 0.348 / 0.426 | 0.382 / 0.453 | 0.383 / 0.442 |
| ≥400 m (n=2,586) | 0.186 / 0.348 | 0.427 / 0.547 | **0.593 / 0.552** | 0.499 / 0.477 |

- B2 的增益主要在 **≥400 m**（0.593 vs B0 0.186）和 **0–20 m**；
  B0 仍在 20–200 m 更强。
- 近目标不可见（<100 m 且 invisible）是另一回事，见 §5。

---

## 5. 近目标 / 持续不可见 / since 分桶（统计）

| 子集 | n | B0 bal | B1 bal | B2 bal | B3 bal | B1 L/R AUC | B2 L/R AUC | B3 L/R AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| near-invisible（<100 m） | 743 | 0.124 | 0.260 | **0.235** | 0.231 | 0.488 | 0.422 | 0.408 |
| invisible since 1–4 | 373 | 0.410 | 0.436 | **0.512** | 0.485 | 0.589 | **0.808** | 0.774 |
| invisible since 5–16 | 468 | 0.118 | 0.202 | **0.243** | 0.237 | 0.534 | **0.586** | 0.544 |
| invisible never/>16 | 8,251 | 0.246 | 0.319 | **0.400** | 0.387 | 0.595 | **0.599** | 0.587 |
| near-invisible since 1–4 | 82 | 0.098 | 0.218 | **0.277** | 0.230 | 0.535 | **0.741** | 0.702 |
| near-invisible since 5–16 | 246 | 0.105 | **0.244** | 0.239 | 0.221 | **0.475** | **0.475** | 0.433 |

**回答“改善是否集中在近目标/持续不可见/1–4 或 5–16 步”：**

- **是的，B2 的改善集中在不可见的最近丢失（since 1–4）**：
  balanced +7.6 pp，L/R AUC +0.219（0.589 → 0.808）。
- 对 since 5–16 有小幅增益（+4.1 pp balanced，L/R AUC +0.052）。
- 对 never/>16，balanced +8.1 pp，但 L/R AUC 几乎不变（0.595 → 0.599）；
  主要是转向/前进决策改善，不是左右方向。
- **近目标不可见整体没有改善**：B2 balanced 0.235 < B1 0.260；
  只有 n=82 的 near-invisible since 1–4 有改善（0.277 vs 0.218，
  L/R AUC 0.741 vs 0.535），但 n=246 的 since 5–16 完全持平。
- B3 在所有上述近目标/持续不可见子集上均未超过 B2。

---

## 6. Left/Right 方向分析

| 指标 | B1 | B2 | B3 |
|---|---:|---:|---:|
| Overall L/R AUC | 0.673 | **0.696** | 0.688 |
| Visible L/R AUC | 0.761 | 0.805 | **0.816** |
| Invisible L/R AUC | 0.597 | **0.608** | 0.594 |
| Near-invisible L/R AUC | **0.488** | 0.422 | 0.408 |
| Invisible since 1–4 L/R AUC | 0.589 | **0.808** | 0.774 |
| Predicted turn rate（overall） | 0.506 | 0.581 | 0.545 |
| Oracle turn rate | 0.728 | 0.728 | 0.728 |
| LEFT recall（overall） | 0.348 | **0.520** | 0.460 |
| RIGHT recall（overall） | 0.316 | **0.327** | 0.326 |

- B2 真正改善了 **LEFT** 与最近丢失的 L/R 排序（invisible since 1–4
  L/R AUC 0.808），不是简单增加转向比例：turn rate 0.581 仍低于
  oracle 0.728。
- **RIGHT 仍然很差**：B1 0.316、B2 0.327、B3 0.326；
  B2 的增益主要是 LEFT + 最近丢失。近目标不可见 L/R AUC B2 0.422
  低于 B1 0.488，说明近距左右方向仍不可靠。
- B3 的 visible L/R AUC 高（0.816），但 invisible/近目标没有；
  说明 8×8 空间分支只在可见状态有效，未解决不可见方向的歧义。

---

## 7. Episode 级置信区间与配对改善（统计）

Bootstrap 按 Episode 重采样 1,000 次（连续 Step 不当作独立样本）。

| 比较（invisible 子集） | ΔBalanced Acc | 95% CI | ΔMacro-F1 | 95% CI | ΔL/R AUC |
|---|---:|---|---:|---|---:|
| B2 − B1 | **+0.0797** | `[+0.0062, +0.1766]` | +0.0616 | `[-0.0063, +0.1502]` | +0.0101 |
| B3 − B1 | +0.0665 | `[+0.0098, +0.1300]` | +0.0545 | `[-0.0012, +0.1169]` | -0.0053 |
| B3 − B2 | **-0.0132** | `[-0.0731, +0.0408]` | -0.0070 | `[-0.0650, +0.0495]` | -0.0154 |

总体（all）：

| 比较 | ΔBalanced Acc | 95% CI | ΔMacro-F1 | 95% CI |
|---|---:|---|---:|---|
| B2 − B1 | **+0.0596** | `[+0.0142, +0.1028]` | **+0.0616** | `[+0.0156, +0.1087]` |
| B3 − B2 | -0.0172 | `[-0.0476, +0.0125]` | -0.0170 | `[-0.0471, +0.0122]` |

逐 Episode accuracy（171 个 Episode）：

| 比较 | Wins | Losses | Ties | Mean Δ |
|---|---:|---:|---:|---:|
| B2 vs B1 | **119** | 42 | 10 | **+8.52 pp** |
| B3 vs B2 | 58 | **81** | 32 | **-0.88 pp** |

**结论：** B2 的增益在 Episode 级是广泛的正向（119 胜/42 负），
不是少数 Episode 拉动；但不可见子集的 macro-F1 CI 仍跨 0，
属于中等证据。B3 在更多 Episode 上变差（58 胜/81 负），不支持空间融合。

---

## 8. B3 有效性拆分与过拟合

| 模型 | temporal valid（14,410）Balanced | temporal invalid（9,255）Balanced | OOF Balanced | Val Balanced |
|---|---:|---:|---:|---:|
| B1 | 0.364 | 0.475 | 0.415 | 0.405 |
| B2 | **0.421** | **0.547** | 0.544 | **0.465** |
| B3 | 0.397 | 0.533 | **0.551** | 0.448 |

- 即使只在 temporal map 有效的状态，B3 也低于 B2（0.397 vs 0.421）；
  无效状态同样低于 B2（0.533 vs 0.547）。
- B3 的 OOF balanced 0.551 高于 B2 0.544，但 Val 更低（0.448 vs 0.465）；
  说明 8×8 空间 CNN 分支没有泛化，且增加了过拟合。
- 因此 **不是“历史图没对齐导致 B3 无效”**：对齐有效时也没有增益。
- 但必须强调：这是 **8×8 coarse** 结果；完整 24×24 没有可验证数据（§1）。

---

## 9. 所有模型都无法预测的状态

B1/B2/B3 全部错误：**9,925 / 23,665 = 41.9%**；
B0 也错误的有 7,578（76.4%）。

| 子集 | n | all-wrong | rate |
|---|---:|---:|---:|
| visible | 14,573 | 5,439 | 37.3% |
| invisible | 9,092 | 4,486 | 49.3% |
| near-invisible | 743 | 436 | **58.7%** |
| far-invisible | 8,349 | 4,050 | 48.5% |
| invisible since 1–4 | 373 | 131 | 35.1% |
| invisible since 5–16 | 468 | 317 | **67.7%** |
| invisible never/>16 | 8,251 | 3,561 | 54.0% |
| distance 0–20 m | 87 | 24 | 27.6% |
| distance 20–50 m | 1,237 | 250 | 20.2% |
| distance 50–100 m | 3,542 | 1,359 | 38.4% |
| distance 100–200 m | 7,209 | 2,959 | 41.0% |
| distance 200–400 m | 9,004 | 4,385 | 48.7% |
| distance ≥400 m | 2,586 | 948 | 36.7% |

**失败状态 vs 全对状态（均值）：**

| 特征 | all-wrong | all-correct |
|---|---:|---:|
| `sim_max_all_top5` | 0.926 | 0.903 |
| `heat_0_top1` | 0.755 | 0.757 |
| `base_max` | 0.951 | 0.924 |
| `coh_max` | 2.851 | 2.567 |
| visible rate | 54.8% | 66.7% |
| mean distance | 235.2 m | 253.6 m |
| mean abs heading | **60.4°** | **50.3°** |
| mean since localizable | 49.1 | 64.7 |
| never localizable rate | **35.9%** | **16.0%** |

**解释（推断）：**

- 所有模型都失败的状态并不是“视觉置信度低”：sim/base/coh 均值甚至更高。
  这说明当前可观测视觉匹配度不能直接给出左右方向；失败更多来自
  **目标不可见且最近可靠观测缺失（since 5–16 / never）、heading 歧义大**。
- 近目标不可见 all-wrong 58.7%、since 5–16 67.7%，是 Recovery 最关心的
  区域；这里没有模型能可靠预测。
- 这也解释了为什么增加 8×8 空间分支没有帮助：空间图仍然来自当前/上一帧
  的可观测匹配，不能凭空恢复“目标在哪一侧”。

---

## 10. 决策与建议

### 决策：**PARTIAL（长历史有效；当前空间图融合无额外价值）**

依据冻结协议：

- **Pass（B3 空间信念）FAIL**：
  B3 vs B2 在不可见子集 balanced **-1.32 pp**，L/R AUC **-1.54 pp**；
  即便 temporal map 有效也低于 B2。没有 +2 pp 稳定增益。
- **Partial（B2 长历史）PASS**：
  B2 vs B1 不可见 balanced **+7.97 pp**，Episode 配对 95% CI 下限 +0.62 pp；
  逐 Episode 119 胜/42 负；L/R AUC 在最近丢失（since 1–4）从 0.589
  提升到 0.808。长历史是真实有效的。
- **但增益不在近目标不可见整体**：B2 near-invisible balanced 0.235
  vs B1 0.260；只有 n=82 的 near-invisible since 1–4 有改善。
- **B3-8×8 没有增益，且 OOF/Val 过拟合**；完整 24×24 因缺图未能验证。

### 因此：

- **不训练完整 MEC，不接入控制器，不训练 M1-Dir/M2/Depth。**
- 停止当前“图像匹配空间图 + CNN 融合”路线（B3-8×8）。
- B2（16 步 compact 历史）保留为**远距/持续不可见/最近丢失**的 Movement
  诊断特征；它不是近目标 Recovery 触发器。
- 下一步应做**显式目标空间信念 / 粗导航定位**，而不是继续堆 MLP 或
  历史帧。若坚持验证 24×24 空间图，最小实验是：
  1. 对已保存的 M0 状态做**增量重渲染**（只重算状态，不重跑轨迹），
     生成真正的 24×24 baseline/coherent/temporal + validity；
  2. 用同一 B0–B3 协议、同一 Episode 划分复测；
  3. 只有在 24×24 上能稳定超过 B2，才考虑 M0 + Spatial Belief 的受控
     MEC 消融。
- 在没有 24×24 和近目标不可见增益之前，不应把该方向接入 Recovery 或
  完整 MEC。

---

## 11. 缺失与限制

1. **完整 24×24 图缺失**：B3 是 8×8 coarse；Stage-2a 24×24 cache 只在
   step 0 对齐，step≥1 后发散，不能静默复用。报告不把 8×8 结果当作
   24×24 结论。
2. temporal map 只使用**紧邻上一帧**且上一动作是 LEFT/RIGHT 的情况；
   FORWARD 历史不 warp；有效率为 Val 60.8%。更长的 turn-valid 历史
   没有尝试，以避免未经验证的复合 yaw/平移。
3. 8×8 base/coherent 是 24×24 的平均池化，空间细节有损；空间分支只有
   16 通道/两层 conv，架构未做超参搜索。
4. B3 OOF > Val（0.551 vs 0.448 balanced），存在过拟合；本轮没有为追求
   结果做正则化/早停调参，以免过拟合验证集。
5. 近目标不可见 n=743，near-invisible since 1–4 只有 n=82，
   since 5–16 只有 n=246；CI 较宽。
6. B0 是 M0 保存的 movement argmax；没有训练或修改 M0。B0 的 L/R AUC
   未从 logits 计算（只有 argmax），因此表中为 —。
7. 单 seed；没有多 seed 复现。
8. Final Test 300 未读取；没有修改旧数据/checkpoint/报告。

---

## 12. 产物与复现

### 12.1 主探针

```
runs/v83_spatial_stage2/stage2g5/probe_20261019T081000Z/
├── probe_results.json
├── probe_provenance.json
├── predictions.npz
└── models/                     # B1/B2/B3 × (5 folds + external) = 18
```

### 12.2 分析

```
runs/v83_spatial_stage2/stage2g5/analysis_20261019T083000Z/analysis.json
```

### 12.3 命令

```bash
python v83_spatial_stage2/stage2g5_probe.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --output-dir runs/v83_spatial_stage2/stage2g5/probe_<stamp> \
  --variants B1 B2 B3 --epochs 30 --device cuda --save-models

python v83_spatial_stage2/stage2g5_analyze.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --probe-dir runs/v83_spatial_stage2/stage2g5/probe_<stamp> \
  --output-dir runs/v83_spatial_stage2/stage2g5/analysis_<stamp> \
  --bootstrap 1000
```

### 12.4 测试

```bash
python -m unittest -v v83_spatial_stage2.test_stage2g5
```

本轮：**7 tests OK**（空间图重建、yaw warp validity、FORWARD invalid、
无标签泄漏、class weights、confusion/LR AUC、all-model failure）。

### 12.5 提交内容

- 报告：`v83_spatial_stage2/V83_STAGE2G5_SPATIAL_BELIEF_MOVEMENT_REPORT.md`
- 协议：`v83_spatial_stage2/V83_STAGE2G5_PROTOCOL.md`
- 源码：`stage2g5_data.py`、`stage2g5_metrics.py`、
  `stage2g5_probe.py`、`stage2g5_analyze.py`
- 测试：`test_stage2g5.py`
- 不提交大文件；checkpoint/predictions/analysis 留在 `runs/`。

---

## Appendix A. 关键数字速查

- B0 总体：acc 0.4402 / bal 0.4431 / mf1 0.4401。
- B1 总体：0.3914 / 0.4054 / 0.3914。
- B2 总体：0.4545 / 0.4652 / 0.4531。
- B3 总体：0.4359 / 0.4482 / 0.4363。
- Invisible balanced：B0 0.253 / B1 0.323 / B2 0.403 / B3 0.390。
- Near-invisible balanced：B0 0.124 / B1 0.260 / B2 0.235 / B3 0.231。
- Invisible since 1–4 L/R AUC：B1 0.589 / B2 0.808 / B3 0.774。
- B2−B1 invisible balanced：+0.0797，CI `[+0.0062,+0.1766]`。
- B3−B2 invisible balanced：-0.0132，CI `[-0.0731,+0.0408]`。
- B3 temporal valid：0.397 vs B2 0.421；invalid：0.533 vs 0.547。
- 逐 Episode：B2 vs B1 119W/42L/10T；B3 vs B2 58W/81L/32T。
- All-model failure：41.9% overall；near-invisible 58.7%；since 5–16 67.7%。
