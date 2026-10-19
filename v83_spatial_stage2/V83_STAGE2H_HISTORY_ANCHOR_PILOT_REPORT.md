# V8.3 Stage-2h — History + Goal-Anchor Belief Integration Pilot

日期：2026-10-19

分支：`experiment/v83-spatial-foundation`

前序提交：`4633562`

本轮性质：**冻结 M0@0.7 的轻量 Movement 修正 pilot；不训练完整 MEC，不接 Recovery，不读 Final Test**

---

## 0. TL;DR（中文结论）

1. **没有获得可宣称的闭环增益。** 在完整 Dev Val 180 上：
   - R0：SR **34.44%**、OSR 38.33%、SPL **0.3432**、nDTW **0.3573**；
   - R1：SR 35.00%、SPL 0.3488、nDTW 0.3554；
   - R2：SR 35.56%、SPL 0.3544、nDTW 0.3660；
   - R3：SR **36.67%**、OSR **40.00%**、SPL **0.3625**、nDTW **0.3576**。
2. **R3 的 SR/SPL 点估计最高，但证据不足。** R3 vs R0：
   ΔSR **+2.22 pp**，95% CI `[-2.78, +7.78]`；
   ΔSPL **+1.93 pp**，CI `[-3.17, +7.11]`；
   ΔnDTW **+0.03 pp**，CI `[-3.50, +3.51]`。
   CI 全部跨 0，nDTW 基本不变；不能宣称有效。
3. **R3 没有比 R2 更好。** R3 vs R2：
   ΔSR +1.11 pp `[-3.33, +5.56]`；
   ΔSPL +0.82 pp `[-3.74, +5.33]`；
   ΔnDTW **-0.84 pp** `[-3.26, +1.45]`。
   SR/SPL 点估计小幅正向，nDTW 点估计反而下降；不满足“SR/SPL 改善且 nDTW 不退化”的 GAB 增益判据。
4. **所有方案的运行时 Movement 修正净收益都是负的。**
   - R1：203 次修正，helpful 48 / harmful 115，净 **-67**；
   - R2：1,239 次，helpful 272 / harmful 372，净 **-100**；
   - R3：2,594 次，helpful 575 / harmful 1,083，净 **-508**。
   修正主要帮助 invisible（R1 +35、R2 +67、R3 +15），但严重伤害 visible
   （R1 -102、R2 -167、R3 -523）。当前探针没有可用的可见性/修正精度门控。
5. **R2/R3 的导航点估计变化来自 Stop/Timeout/路径动态，不是 Movement 变好。**
   R2 的 premature-stop 增加 **+6.11 pp**（CI `[+0.56,+11.67]`），
   timeout 降低 **-7.22 pp**（CI `[-12.22,-2.22]`），平均步数从 147.9
   降到 131.8；这抵消成 SR/SPL 的小幅正向，但 Movement 修正本身净负。
   R3 类似：premature-stop +3.33 pp，timeout -5.56 pp，步数 128.6。
6. **GAB 离线 Movement 分类确实有提升，但没有转化为闭环收益。**
   R3 Val movement accuracy **0.4707** / balanced **0.4776**，
   高于 B2 的 0.4545 / 0.4652；但在闭环修正中，R3 的 helpful/harmful
   为 575/1083，比 R2 的 272/372 更差，visible 修正伤害更大。
7. **明确判断：停止本轮 adapter/Movement 修正路线；不启动完整主干重训。**
   现有证据不足以证明 R1/R2/R3 任一方案有效；R3 不优于 R2；GAB 没有
   提供额外闭环价值。若未来重新尝试，必须先把修正精度门控（尤其
   visible 状态）和 Episode 级收益做出来，而不是继续加分支/调阈值。
8. **本轮没有训练完整 MEC、M1-Dir、M2、Depth；没有接控制器；没有读
   Final Test 300；没有修改任何旧 checkpoint。**

---

## 1. 公开工作借鉴与不可照搬的原因

### 1.1 AerialVLN

AerialVLN（Liu et al., ICCV 2023, arXiv:2308.06735）把无人机连续 3D
城市导航表述为视觉-语言导航：智能体根据自然语言指令和第一人称视觉
观测行动，没有预建导航图；其 baseline 还引入了 look-ahead guidance。
它强调的是**语言指令驱动的长程视觉导航**。

### 1.2 CityNav

CityNav（Lee et al., ICCV 2025, arXiv:2406.14240）是真实世界航拍 VLN
数据集，核心辅助表示是 **Geo-Semantic Map (GSM)**：包含当前视场、
已探索区域、OpenStreetMap 地标、潜在目的地和周边物体；地标通道对
成功最关键。其目标描述来自 CityRefer，包含地标及空间关系。

### 1.3 LookasideVLN

LookasideVLN（Ning et al., CVPR 2026, arXiv:2604.17190）提出
**direction-aware** 的 lookaside 范式：

- Egocentric Lookaside Graph（ELG）：围绕无人机当前位置编码
  instruction-relevant landmarks 及其方向关系；
- Spatial Landmark Knowledge Base（SLKB）：轻量地标空间记忆；
- MLLM navigation agent：对齐视觉、指令与方向地标信息。

它强调的是**方向线索 + egocentric landmark graph + 短程 lookaside**，
而不是全局城市图。

### 1.4 本轮 GAB 借鉴了什么、为什么不能照搬

**借鉴的机制类型：**

- egocentric directional belief（LookasideVLN 的方向感知/短程 lookaside）；
- 历史视觉记忆与“最近可靠观测”的时间衰减（AerialVLN/CityNav 的
  视觉历史/空间推理思想）；
- 用显式空间分布而不是单一全局池化向量表达方向（CityNav GSM 的
  空间通道思想，但本轮不引入地理地图）。

**不能直接照搬的原因：**

- 当前 ToyUAVEnv 是 **Image-Goal** 环境，没有自然语言指令、没有
  CityRefer 式地标描述、没有 OpenStreetMap/Geo-Semantic Map。
- CityNav 的 GSM 依赖真实地标坐标与语义地图；在 ToyUAVEnv 中构造它
  只能读取 `env.goal`/`env.goal_landmark.position` 等模拟器真值，
  这会构成监督泄漏，本轮明确禁止。
- AerialVLN/LookasideVLN 依赖指令中的方向词、landmark graph 和
  MLLM 语义对齐；本轮没有语言输入，也没有训练 MLLM 的预算。
- 因此 GAB 只能是**纯可观测视觉历史 + 动作**的 egocentric belief；
  它不是地理地图，也不伪造“目标附近地标坐标”。真正的 map-assisted
  研究应另设独立实验。

---

## 2. GAB 设计（无标签、因果、Episode reset）

### 2.1 输入与特征

GAB 只使用：

- 当前 3×8×8 Goal-Current heatmap（Saved 特征中的 heat_0/1/2）；
- 当前匹配图的空间分布、峰值、熵与匹配置信度；
- 最近一次可靠匹配的 8-bin 相对方位分布；
- 最近可靠匹配距当前的步数（last-seen age）；
- 最近实际执行的 LEFT/RIGHT/FORWARD 动作及 dt（通过历史动作链）；
- 当前信念的 validity/uncertainty。

输出 = **8-bin egocentric target-bearing belief** + 8-bin last-reliable
belief + confidence + entropy + age + valid，共 **20 维**。

### 2.2 更新规则

- 当前 evidence 从 heatmap 的 8×8 分布得到 8 个横向 bin。
- 置信度 = heatmap softmax 峰值；可靠性阈值取 **Train observable
  heatmap 的 60th percentile**（仅 Train，不使用标签）。
- 当前证据可靠：`belief = (1-α)·rotated_prior + α·current_evidence`，
  更新 last-reliable，age=0，valid=1。
- LEFT/RIGHT：按已验证的 yaw 几何旋转 prior（8 bins，90° FOV，
  15° turn → 1.333 bin 的线性重分配）；**FORWARD 不进行固定像素
  translation warp**。
- 证据不可靠：belief 向 uniform 衰减，confidence 衰减，age 增加；
  age 达到 16 时 valid=0，不允许永久保留过度自信的旧方位。
- 每个 Episode reset 全部 belief/age/valid。

可观测统计：Train GAB valid rate **55.5%**，Val **51.8%**；
Val mean age 8.2 步。没有使用真实距离、可见性、目标坐标、Oracle
Action、成功标签或未来动作。

### 2.3 GAB 的训练与修正

- R3 = Stage-2g.5 B2 tabular（current + 16-step compact history）
  + GAB 20 维 → 3-class Movement MLP（hidden 128, dropout 0.2,
  30 epochs, batch 8,192, Adam 1e-3 / wd 1e-4, seed 0）。
- R1/R2 直接复用 Stage-2g.5 B1/B2 external 权重；没有重训。
- 所有方案共用同一修正规则：
  - Stop 决策完全保留 M0@0.7；
  - M0 选择 Movement 时，probe top1 与 M0 不同且
    `max_prob >= τ` 才允许替换；
  - τ 只在 **Train OOF** 上选择，选择“helpful - harmful”净收益
    最大的阈值；Dev Val 不参与阈值选择。
- 选择结果（Train OOF）：

| 方案 | τ | corrections | helpful | harmful | net | OOF accuracy 变化 |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 0.99 | 834 | 476 | 335 | +141 | +0.20 pp |
| R2 | 0.98 | 3,090 | 1,551 | 923 | +628 | +0.88 pp |
| R3 | 0.95 | 5,221 | 2,725 | 1,930 | +795 | +1.11 pp |

OOF 上均为正；但运行时（见 §5）全部转为负净收益，说明修正状态分布
发生了闭环偏移。

---

## 3. 数据与复用审计

- Train 600 真实闭环：81,303 states；Dev Val 180：26,628 states。
- Movement 评测集：`step>=16 & action!=STOP`，Train 71,576 / Val 23,665。
- R0 复用 `stage2d/r4_20261019T002626Z/threshold_0.7_per_episode.csv`；
  180 个 episode_index/seed/band 与 Dev Val specs **完全一致**
  （mismatch=0），因此无需重跑 R0。
- B1/B2 external checkpoints 从 Stage-2g.5 复用：
  - B1 input 2,858；B2 input 4,122；均值/方差与输入定义已核验。
- R3 新训练 6 个 checkpoint（5 fold + external），输入 4,142
  （B2 4,122 + GAB 20）。
- R0 闭环原始动作与 Stage-2d baseline 一致；2-Episode smoke 中
  R0 outcome 与 CSV 完全一致。
- **24×24 空间图缺失**：Stage-2a cache 与真实 M0 状态只在 step 0
  完全对齐，step≥1 后 coherent MSE 约 0.18–0.28；本轮没有再引入
  错误对齐的 24×24 图。GAB 使用 8×8 heatmap/pooled 空间结构。
- 没有新采集数据；没有修改旧数据/checkpoint。

---

## 4. 闭环总体结果（Dev Val 180，严格配对）

| 方案 | SR | OSR | SPL | nDTW | Premature Stop | Timeout | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0（冻结 M0） | 34.44% | 38.33% | 0.3432 | 0.3573 | 48.89% | 16.67% | 147.93 |
| R1（B1 current） | 35.00% | 38.33% | 0.3488 | 0.3554 | 48.33% | 16.67% | 146.58 |
| R2（B2 history） | 35.56% | 38.33% | 0.3544 | **0.3660** | 55.00% | 9.44% | 131.83 |
| R3（history+GAB） | **36.67%** | **40.00%** | **0.3625** | 0.3576 | 52.22% | 11.11% | 128.64 |

**R3 的 SR/SPL 点估计最高，但 nDTW 与 R0 基本持平，且 R3 的 SR/SPL
配对 CI 均跨 0。** R2 的 nDTW 点估计最高，但 premature-stop 显著增加，
timeout 显著减少，SR/SPL CI 跨 0。

### 4.1 分 Band

| Band | 方案 | SR | OSR | SPL | nDTW | Premature | Timeout | Steps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| short | R0 | 48.33% | 56.67% | 0.4833 | 0.5179 | 36.67% | 15.00% | 141.03 |
| short | R1 | 48.33% | 56.67% | 0.4833 | 0.5164 | 38.33% | 13.33% | 139.35 |
| short | R2 | 48.33% | 55.00% | 0.4833 | **0.5291** | 45.00% | 6.67% | 110.93 |
| short | R3 | 43.33% | 53.33% | 0.4313 | 0.4899 | 45.00% | 11.67% | 123.78 |
| medium | R0 | 33.33% | 36.67% | 0.3312 | 0.3372 | 51.67% | 15.00% | 153.93 |
| medium | R1 | 33.33% | 35.00% | 0.3312 | 0.3413 | 53.33% | 13.33% | 142.82 |
| medium | R2 | 35.00% | 36.67% | 0.3479 | 0.3502 | 53.33% | 11.67% | 139.07 |
| medium | R3 | **43.33%** | **43.33%** | **0.4295** | **0.3843** | 50.00% | 6.67% | 117.13 |
| long | R0 | 21.67% | 21.67% | 0.2152 | 0.2169 | 58.33% | 20.00% | 148.83 |
| long | R1 | 23.33% | 23.33% | 0.2318 | 0.2083 | 53.33% | 23.33% | 157.58 |
| long | R2 | 23.33% | 23.33% | 0.2319 | 0.2186 | 66.67% | 10.00% | 145.48 |
| long | R3 | 23.33% | 23.33% | 0.2268 | 0.1986 | 61.67% | 15.00% | 145.02 |

**R3 的改善集中在 medium band（SR 33.33% → 43.33%），但 short band
SR 从 48.33% 降到 43.33%，long 的 nDTW 从 0.2169 降到 0.1986。**
这不是一个一致的 band-wise 增益。

---

## 5. Movement 修正分析（运行时）

| 方案 | Move states | Base acc | Final acc | Corrections | Helpful | Harmful | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 26,235 | 0.4611 | 0.4585 | 203 | 48 | 115 | **-67** |
| R2 | 23,566 | 0.4551 | 0.4508 | 1,239 | 272 | 372 | **-100** |
| R3 | 22,996 | 0.4742 | 0.4521 | 2,594 | 575 | 1,083 | **-508** |

**Visible / invisible 分解：**

| 方案 | 子集 | Move states | Base acc | Final acc | Corrections | Helpful | Harmful |
|---|---|---:|---:|---:|---:|---:|---:|
| R1 | visible | 17,639 | 0.5562 | 0.5504 | 108 | 1 | 103 |
| R1 | invisible | 8,596 | 0.2659 | 0.2700 | 95 | 47 | 12 |
| R2 | visible | 15,648 | 0.5753 | 0.5646 | 518 | 154 | 321 |
| R2 | invisible | 7,918 | 0.2175 | 0.2259 | 721 | 118 | 51 |
| R3 | visible | 15,355 | 0.5960 | 0.5620 | 1,282 | 331 | 854 |
| R3 | invisible | 7,641 | 0.2293 | 0.2313 | 1,312 | 244 | 229 |

- **修正对 invisible 有小幅正收益，但对 visible 伤害更大**；所有方案
  的总净收益为负。
- R3 增加了修正数量但没有提高选择性；visible 修正 harmful 854，
  远高于 helpful 331。
- 这说明“Probe 在离线分类上变好”并没有转化为“修正更可信”；
  当前缺少可用的可见性/修正精度门控。

---

## 6. Episode 级配对比较（vs R0）

| 比较 | SR Δ（W/L/T） | SR 95% CI | SPL Δ | SPL 95% CI | nDTW Δ | nDTW 95% CI |
|---|---|---|---:|---|---:|---|---|
| R1 vs R0 | +0.56 pp（1/0/179） | `[0.00, +1.67]` | +0.0056 | `[0.00, +1.67]` | -0.0020 | `[-1.20, +0.50]` |
| R2 vs R0 | +1.11 pp（9/7/164） | `[-3.33, +5.56]` | +0.0111 | `[-3.24, +5.51]` | +0.0086 | `[-1.93, +3.72]` |
| R3 vs R0 | **+2.22 pp（14/10/156）** | `[-2.78, +7.78]` | **+0.0193** | `[-3.17, +7.11]` | +0.0003 | `[-3.50, +3.51]` |

R3 vs R2：

| 指标 | Δ | 95% CI | W/L/T |
|---|---:|---|---|
| SR | +1.11 pp | `[-3.33, +5.56]` | 10 / 8 / 162 |
| SPL | +0.82 pp | `[-3.74, +5.33]` | 10 / 11 / 159 |
| nDTW | **-0.84 pp** | `[-3.26, +1.45]` | 50 / 51 / 79 |

**所有关键 CI 都跨 0；R3 相对 R2 的 nDTW 点估计为负。** 因此不能
宣称 R2 或 R3 有可靠闭环增益，更不能宣称 GAB 提供了额外价值。

其他 Stop/Termination 指标：

- R2 vs R0：premature-stop **+6.11 pp**（CI `[+0.56,+11.67]`），
  timeout **-7.22 pp**（CI `[-12.22,-2.22]`）。
- R3 vs R0：premature-stop +3.33 pp（CI `[-2.78,+9.44]`），
  timeout -5.56 pp（CI `[-11.67,+0.56]`）。
- 这些变化说明 R2/R3 主要是把部分 timeout 转成了提前 Stop，而不是
  改善了 Movement 方向；SR/SPL 的小幅点估计不足以证明 adapter 有效。

---

## 7. R3 GAB 的离线表现与闭环表现

- R3 Val movement accuracy **0.4707** / balanced **0.4776**
  （B2：0.4545 / 0.4652），GAB 确实改善了离线 Movement 分类。
- R3 Train OOF accuracy **0.5702** / balanced **0.5457**。
- 但闭环中 R3 的 helpful/harmful 为 575/1083，R2 为 272/372；
  R3 的 visible 修正 harmful 854，说明 GAB 没有提供更好的修正选择性。
- R3 的 short band SR 反而低于 R0/R1/R2；medium 更高；long nDTW 更低。
  增益在 band 间不一致。
- 结论：**GAB 对离线 Movement 分类有正贡献，但没有转化为闭环空间
  引导收益；当前 GAB 设计不通过。**

---

## 8. 决策

### **决策：停止本轮 adapter / Movement 修正路线；不启动完整主干重训。**

依据：

1. R1/R2/R3 的 SR/SPL/nDTW 配对 95% CI 全部跨 0；变化小或证据不足。
2. R3 点估计 SR/SPL 最高，但 nDTW 与 R0 持平，且 R3 相对 R2 的
   nDTW 为负；不满足“SR/SPL 改善且 nDTW 不退化”。
3. 所有方案运行时修正净收益为负；visible 修正造成大量 harmful；
   当前探针没有可用的可见性/修正精度门控。
4. R2 的 nDTW/timeout 点估计改善伴随 premature-stop 显著增加，
   不能解释为 Movement 能力提升。
5. 单 seed、180 Episode；没有多 seed 复现；不能据此启动完整 MEC 或
   完整 Transformer-M 主干重训。

### 下一步建议（本轮未执行）

- 不要继续盲目增加 GAB 分支、历史长度或调 τ；当前瓶颈是**修正精度/
  可见性门控**，不是模型容量。
- 若未来重启 adapter 路线，最小前置条件是：
  1. 在 Train OOF 上按 visible/invisible 分别校准修正规则；
  2. 只在 M0 Movement 置信度低、probe 置信度高且修正方向与历史
     证据一致的子集修正；
  3. 用多 seed + Episode CI 验证；先要求 visible harmful 显著下降，
     再评估闭环 SR/SPL/nDTW；
  4. 若做不到，转向**显式目标空间信念 / 粗导航定位**，而不是继续
     叠加 adapter。
- 完整 24×24 空间图仍缺失；若要做 map-assisted，应另设独立实验，
  对已保存 M0 状态增量重渲染，而不是混入本轮 GAB。

---

## 9. 缺失与限制

1. 单 seed、单模型初始化；没有多 seed。
2. Dev Val 180 的 Episode 级 CI 较宽；SR 的 4 个 Episode 差异
   （R3）不足以支撑宣称。
3. R0 复用 Stage-2d CSV；已核验 seeds/bands 完全一致，但没有在本轮
   重新 rollout R0 全 180（2-Episode smoke 已核对 outcome 一致）。
4. R1/R2/R3 运行时的历史动作包含被修正后的动作，而训练数据来自
   M0 原始动作；存在 feedback distribution shift。这是 pilot 的真实
   运行条件，但也限制了离线修正在闭环中的可迁移性。
5. GAB 使用 8×8 coarse heatmap，不是完整 24×24；完整 24×24 的
   对齐与效果未验证。
6. 没有读取 Final Test 300；没有训练完整 MEC/M1-Dir/M2/Depth；
   没有修改旧 checkpoint/data。
7. R3 的 medium band SR 提升较大，但 short/long 不一致；不能把
   medium 的单 band 点估计当作总体结论。

---

## References（公开论文信息）

- AerialVLN: Liu et al., *AerialVLN: Vision-and-Language Navigation for UAVs*,
  ICCV 2023, <https://arxiv.org/abs/2308.06735>
- CityNav: Lee et al., *CityNav: A Large-Scale Dataset for Real-World Aerial
  Navigation*, ICCV 2025,
  <https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html>
- LookasideVLN: Ning et al., *LookasideVLN: Direction-Aware Aerial
  Vision-and-Language Navigation*, CVPR 2026,
  <https://www.openaccess.thecvf.com/content/CVPR2026/html/Ning_LookasideVLN_Direction-Aware_Aerial_Vision-and-Language_Navigation_CVPR_2026_paper.html>

---

## 10. 产物、命令与提交

### 10.1 数据/模型

```
runs/v83_spatial_stage2/stage2h/
├── r3_20261019T090500Z/          # R3 GAB movement branch (6 ckpt)
├── correction_20261019T091000Z/  # Train-OOF correction thresholds
├── smoke_20261019T092500Z/       # 2-episode smoke
├── eval_20261019T093000Z/        # full Dev Val 180 R1/R2/R3
│   ├── R1_per_episode.csv, R1_steps.csv
│   ├── R2_per_episode.csv, R2_steps.csv
│   ├── R3_per_episode.csv, R3_steps.csv
│   └── metrics.json
└── analysis_20261019T095000Z/analysis.json
```

### 10.2 命令

```bash
# R3 GAB movement branch
python v83_spatial_stage2/stage2h_r3_train.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --val-dir runs/v83_spatial_stage2/stage2g3/real_val_20261019T132000Z \
  --output-dir runs/v83_spatial_stage2/stage2h/r3_<stamp> \
  --epochs 30 --device cuda --save-models

# Train-OOF correction thresholds
python v83_spatial_stage2/stage2h_select_correction.py \
  --train-dir runs/v83_spatial_stage2/stage2g3/real_train_20261019T132500Z \
  --stage2g5-probe-dir runs/v83_spatial_stage2/stage2g5/probe_20261019T081000Z \
  --r3-dir runs/v83_spatial_stage2/stage2h/r3_<stamp> \
  --output runs/v83_spatial_stage2/stage2h/correction_<stamp>/correction_thresholds.json

# Closed-loop (R0 reused from exact Stage-2d CSV)
python v83_spatial_stage2/stage2h_eval.py \
  --output-dir runs/v83_spatial_stage2/stage2h/eval_<stamp> \
  --stage2g5-probe-dir runs/v83_spatial_stage2/stage2g5/probe_20261019T081000Z \
  --r3-dir runs/v83_spatial_stage2/stage2h/r3_<stamp> \
  --correction-json runs/v83_spatial_stage2/stage2h/correction_<stamp>/correction_thresholds.json \
  --schemes R1 R2 R3 --device cuda
```

### 10.3 测试

```bash
python -m unittest -v v83_spatial_stage2.test_stage2h
```

本轮 Stage-2h：**5 tests OK**；Stage-2g.4/2g.5/2h 合计 **18 tests OK**
（2-Episode smoke 另外验证 R0 与 Stage-2d baseline 一致）。
