# 曲率认知场的实验模块

《曲率认知场-实验设计》的 Phase 0–6 实现，加上十项方法验证。

**它从 `rl-scaffold` 里拆出来单独成一个仓库了。** 理由不是"代码不该和知识放在一起"，
而是：**实验的版本不该挂在产品的版本号后面**。原来它住在 `rl-scaffold/field/`，
于是那个仓库为了这个实验发了一个产品版本（v0.0.5）——那是不对的。

**它读语料的 `nodes/`，一个字都不写。** 这一点由 `phase5.py` 的快照比对和
`checks.py` 的 B-D5 钉着。

**中文** · [English](README.en.md) · 设计声明见 [`DECLARATION.md`](DECLARATION.md)

---

## 它怎么拿到语料

这个仓库**不含** `nodes/`。它要读一份 `rl-scaffold`（或任何同构的 Scaffold 语料）。
位置由 `corpus.py` 决定，优先级：

```
1. FIELD_CORPUS 环境变量
2. 默认：与本仓库并排的 ../rl-scaffold
```

```bash
# 默认布局（两个仓库并排）→ 什么都不用设
python checks.py

# 别处
FIELD_CORPUS=/path/to/rl-scaffold python checks.py
```

⚠️ **没有语料时不会静默通过。** 需要语料的六条检查（B-D1 / B-D2 / B-D7 /
B-D11 / B-D14 / B-D15）会**显式标成"跳过"**并计入单独的计数——因为
**跳过不等于通过，它们什么都没验**。

### 这个仓库有两半，其中一半根本不碰语料

| 半边 | 模块 | 需要语料吗 |
|---|---|---|
| **机制层** | `spectral` `geometry` `connectivity` `curvature` `curvature_split` `ollivier` `field` `ppr` `metrics` `conformal` `knockout` `synthetic` `direction` `directed` `attack` `invariants` | **不需要**，可以脱开任何语料单独用 |
| **实验层** | `graph` `checks` 与 `phase*.py` | 需要（`graph` 的纯图工具仍不需要） |

⚠️ 这条"机制层不需要语料"是**被自己的测试打掉过一次**才成立的：
原来 `graph.py` 在**模块导入时**就连语料，于是 `curvature.py` 只想拿一个
`degrees` 也被拖去要语料。改成**延迟挂载**之后才为真。见 `graph.py:load_nodes` 的注释。

---

## 它不是什么

- **不是产品，是实验记录。** 它读语料的 `nodes/`，**一个字节都不写** ——
  不改校验规则、不改检索、不进任何视图。
- **结论只对这张图负责**（36 节点 / 112 条处境 / 47 条节点间关系）。
  换一张图，下面的数字全部作废。
- **合成图只做机制验证**，不用来报任何性能数字。
- **本仓库跑不全**：没有语料时六条检查**显式跳过**。
  **跳过不等于通过** —— 它们什么都没验。
- **`B-D7` 是红的，且不由本仓库修。** 靶子在上游数据（`HELD_OUT` 里 8/17 是 cue 原文），
  修法需上游拍板。本仓库只报告，**不自行放宽、不删数据**。

---

## 一句话结论

**四个 kill test 里两个通过（谱、控制），一个通过但证据弱（可判据性），一个不成立（遗忘的一半）。
曲率那条线从"已被证伪"改判为"仅 1 条边支撑的待验项"。**

| 子命题 | 状态 | 关键数字 |
|---|---|---|
| C1 度量存在 | **成立** | 有效维数 19.1/35，λ_max/λ₁ = 85.9 |
| C2 意图能控制构造 | **成立**（t ≥ 2） | Mantel p = 0.0005（12/12 组） |
| C3 构造确定性 | **成立** | 逐元素相同 |
| C4 入口鲁棒性 | **成立** | 容忍约 2 跳（图直径 8） |
| C5 可判据性 | **成立，但「比分数好」已被推翻** | R1 自身 AUC 1.000 / p = 0.0028；**配对检验 R1 vs 强对照 p = 0.7852（不显著）** |
| C5 损伤实验 | **失败** | 效应是自己混淆放大的 |
| C6 遗忘 | **不成立** | "确定变多解"被证伪：不同读数 2.67 → 1.11 |
| C7 可用性 | **一半成立** | 无参数判据召回 9/9 vs 基线 7/9 |
| 曲率 Forman | **被证伪** | 度数项方差占比 1.054 |
| 曲率 Ollivier-Ricci | **未被证伪，仅 1 条边支撑** | 唯一界面桥 κ = −0.500 |
| 界面桥（合成图，n=10） | **成立** | 界面 vs 悬边 AUC = 1.000，四张图全部完全分开 |

---

## 怎么跑

```
python checks.py --expect-skipped 6   # B-D 否证检查（21 条产物 + 1 条数据；缺语料时恰好 6 条跳过）
python phase1.py              # 图 / 谱 / 有效电阻 / Forman / Fiedler 切 / 空间
python curvature_compare.py   # Ollivier-Ricci vs Forman（含三条手算自检）
python phase2c.py             # C2/C3 可控性（Mantel 置换检验）
python phase3.py              # C4 入口鲁棒性 + 匹配器排序诊断
python phase4.py              # C5 可判据性（go/no-go）
python phase5.py              # C6 遗忘
python phase6.py              # C7 可用性 + 消融

# 方法推荐 ⑤–⑩ 带进来的探针
python phase2d.py             # 五种读数正面对比（推荐①的判决）
python phase4b.py             # sweep 电导当判断统计量 + 配对检验（推荐②③）
python phase4c.py             # 谱模式敲除（推荐⑤）
python phase7a.py             # 方向一致性 + 多数派免疫（推荐⑦）
python phase7b.py             # 合成图界面桥机制验证（推荐⑥，不需要语料）
python phase7c.py             # 有向谱与有向 OR（推荐⑧）
python phase7d.py             # 攻击类型学 + 随机图不变量（推荐⑨⑩）
```

唯一的依赖是 **PyYAML**（为了复用上游的 front matter 解析器，不重写它）。
全部确定性，总耗时秒级。

`B-D7` 是**按设计为红**的（靶子在上游数据），所以**带语料**跑 `checks.py` 会**故意退出码 1** ——
那是检查在命中，不是仓库坏了。带上 `--product-only` 就只算产物检查。

（`phase2.py` 与 `phase2b_readout.py` 是过程记录：前者是**作废**的那版检验，
后者是诊断出它为什么作废的探针。留着是为了可审计，不要拿它们当结论。）

---

## 文件

### 产物代码

| 文件 | 干什么 |
|---|---|
| `graph.py` | 把 `nodes/` 建成图（只取 id/type/relations/cues，**不取任何正文**） |
| `spectral.py` | 拉普拉斯、Jacobi 特征分解、有效电阻距离、谱描述 |
| `geometry.py` | 扩散坐标 `Ψ_k = u_k/√λ_k`、Fiedler 切、切电导 |
| `connectivity.py` | 桥、2-边连通分量（Tarjan 低链） |
| `curvature.py` | Forman 曲率 |
| `curvature_split.py` | Forman 的项分解（用它证伪了 Forman） |
| `ollivier.py` | **Ollivier-Ricci 曲率**（整数化最小费用流算精确 W₁）+ 三条手算自检 |
| `field.py` | 场的演化：`K(t) = e^{−tL}δ_e + α(I − e^{−tL})L⁺S`、读数、距离 |
| `ppr.py` | **PPR + sweep cut**：局部质量分布、逐前缀电导、图自身的节点零分布校准 |
| `metrics.py` | 优超概率 AUC、**配对精确置换检验**、Holm 校正、位次/召回 |
| `conformal.py` | 分位数校准取代手写的线 + 覆盖机制验证 |
| `knockout.py` | **谱模式敲除**（图不动，替代删节点的损伤实验） |
| `synthetic.py` | 合成图生成器（团环、挂叶、随机树）+ 按埋点分组 |
| `direction.py` | **方向一致性**：均值 / 第一主成分 / max-min 三种规则 + 免疫测试 |
| `directed.py` | 有向邻接、平稳分布、强连通判定、拓扑层级、可达数、**有向 OR** |
| `attack.py` | ASPIC+/ABA 三分攻击类型 + **边具体化**（reify） |
| `invariants.py` | **随机图上的不变量测试**（六条，含反例上报） |
| `checks.py` | **B-D 否证检查**（22 条：21 产物 + 1 数据；`--product-only` 让退出码不背上游数据，`--expect-skipped N` 断言跳过数） |
| `corpus.py` | **语料定位**：`FIELD_CORPUS` > 默认并排的 `../rl-scaffold`；不改上游，只改 `fp.NODES` |
| `paths.py` | 报告落点（`docs/`）——**报告不许写在根目录** |

### 报告与判决（都在 `docs/`）

| 文件 | 内容 |
|---|---|
| `phase1_graph.md` | 图的规模、连通分量、cue 全是叶子 |
| `phase1_spectrum.md` | 完整谱、有效电阻分布、有效维数 |
| `phase1_curvature.md` | Forman 逐边 + 项分解 |
| `phase1_geometry.md` | 36 个节点的扩散坐标、Fiedler 切、跨切边 |
| `phase1b_curvature_compare.md` | OR vs Forman：度数代理程度、悬边/界面桥/核内边分组 |
| `phase2_controllability.md` | （作废）第一版可控性 |
| `phase2b_readout.md` | 诊断：三层意图恢复率，定位损失在哪一层 |
| `phase2c_controllability.md` | **有效的**可控性结果 |
| `phase3_robustness.md` | 匹配器排序诊断 + 入口扰动容忍度 |
| `phase4_grounding.md` | 共振/分数/桥三判据对照、针对性比较、损伤实验 |
| `phase5_decay.md` | 遗忘曲线 |
| `phase6_usability.md` | 可构造性判据的 2×2、样本内间隔、消融 |
| `phase2d_readout_compare.md` | **五种读数对比**：损失在「用集合比」不在「怎么选」 |
| `phase4b_sweep_statistic.md` | 五个统计量的 AUC、**配对精确检验**、conformal 应用 |
| `phase4c_knockout.md` | 谱模式敲除：Fiedler 模夺魁 16/36 |
| `phase7a_direction.md` | 方向一致性、噪声扫描、三种规则的免疫对比 |
| `phase7b_synthetic_interfaces.md` | 合成图界面桥：n=10、AUC 1.000、树作反面对照 |
| `phase7c_directed.md` | 有向拉普拉斯为何不适用 + DAG 工具 + 有向 OR |
| `phase7d_attack_and_invariants.md` | 三分攻击的可表达性 + 六条不变量 × 24 张随机图 |
| `raw_*.txt` | Phase 0 冻结的上游基线原始输出 |
| `phase1_verdict.md` | Phase 1 判决 + **§7 追加：对 kill test 2 的自我修正** |
| `phase2_verdict.md` | Phase 2 判决（含一次把自己的检验做废的完整记录） |
| `phase3_verdict.md` | Phase 3 判决（推翻上游"瓶颈在匹配算法"的结论） |
| `phase4_verdict.md` | Phase 4 判决（go/no-go，含损伤实验失败） |
| `phase5_6_verdict.md` | Phase 5/6 判决 + Phase 0–6 汇总表 |
| `methods_verdict.md` | **十条方法推荐的逐条判决**（含配对检验推翻 C5 的完整记录） |

---

## 门面文件

| 文件 | 是什么 |
|---|---|
| [`DECLARATION.md`](DECLARATION.md) | **设计声明**：它要解决什么问题、结构、四条纪律、结论、三条自我更正、逐条判决 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 参与方式（**先读它，再动手**）——含"不许做的事"与语料 ref 变更怎么办 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |
| [`NOTICE`](NOTICE) | 与 `rl-scaffold` 的关系、抽取方式、第三方 |
| `.github/workflows/ci.yml` | 两个 job：**不带语料**（门禁）与**带语料**（产物门禁，上游数据只报告） |

**代码平铺在根目录**，与 `arena` / `nested-traceable-discussion-graph` 同一形状；
**所有报告与判决收在 `docs/`**（31 份：19 报告 + 6 判决 + 6 基线），
这样根目录只剩代码与门面。

---

## 待办

1. **更大的正负例集**（正例 50 / 负例 20）重跑 C5 判据。现在样本是 9 vs 4，
   而配对检验已经证明**现有样本量下看不出差异**。
2. **conformal 需要可交换的校准集。** 机制已验（名义 0.95/0.90/0.80 →
   实测 0.9640/0.9167/0.8027，样本不足时**拒绝给门**），但真实数据上做不到同分布。
   **障碍是样本量，不是方法**——与第 1 条是同一件事。
3. **方向语义**：上游 `SPEC.md` **自己前后矛盾**——它的散文规则说
   「A 出现在 B 的 `relations` 里 ⟹ A 指向 B」，它自己的方向表说「论据 → 立场」
   且写进 `relations` 的是论据。实测数据支持**表**那一侧。
   三条出路已登记在 `rl-scaffold` 的 `GAPS.md` §7.2。
4. **`relations` 是有向无环图**（36 个强连通分量、每个节点自成一份）值得写进上游文档：
   它决定了哪些有向工具可用（Chung 拉普拉斯不行，DAG 层级 / 可达可以）。
5. **max-min 求解器是局部的**。只报了它达到的最小投影，没报与全局最优的差距。
6. **给 `checks.py` 的跳过态加一条 CI 断言**：现在 CI 只断言"退出码 0"，
   而缺语料时退出码也是 0（六条跳过）。**"跳过"应当被单独断言**，
   否则将来某条检查悄悄变成"永远跳过"没人会发现。

⚠️ 与上游 `rl-scaffold` 共有一条待决项：`B-D7` 命中——`HELD_OUT` 里 17 句有 8 句是
期望节点自己的 cue 原文。修法两条（加原文/改写标记，或给指标改名）都在上游拍板，
**本仓库只报告，不自行放宽、不删数据、不改语料**。
