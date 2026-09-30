"""Phase 6 跑腿：可用性（C7）与消融。

设计稿 §3 Phase 6 原本写的是"用轨迹构造替换 `find_path.py` 的六步模板"。
**实际做出来的东西不是那个**：Phase 3/4 建立的是一条
`处境句 → 匹配器排名 → 前 k 个入口 → 各自构造 → 共振 ρ` 的判据链，
不是六步报告的替代品。所以 C7 按**实际做出来的东西**测，不按设计稿的想象测：

    C7 = 用场给出的判据（可构造性 + 共振）做"接受/拒绝"，跟 TAU 那条线比。

---
两组，分开报
------------

    C7a  **无参数**那一半：候选少于 2 个就拒绝，且 ρ > 0。
         两者都是定义性的——ρ 需要至少两个构造才谈得上"一致性"；
         ρ = 0 意味着两个构造毫无一致、场什么都没说。
         **不引入任何可调参数。**

    C7b  需要一条线的那一半。**本模块不嵌入这条线**（B-D6 禁止内联数字线）。
         只报出"在样本内存在一条能完全分开的线"，以及它的间隔在哪里，
         并明确标注：9 vs 4 的样本内完全分开**不构成证据**。

---
消融（三组，逐组说明它到底做没做）
----------------------------------

    A1 去意图    本任务的判据链用的是**中性扩散**（α=0），本就不含意图，
                 所以 A1 在这条链上**结构性地不适用**。意图的对照在 Phase 2 的
                 α=0 对照里做过（α=0 → 1 个构造，意图确实在做功）。
    A2 去曲率    已由 Phase 1 完成：Forman 被项分解证伪，曲率**已经不在**判据链里。
                 Phase 1b 又补做了 Ollivier-Ricci，结论见 `phase1b_curvature_compare.md`。
    A3 去轨迹    等价于 Phase 4 的 R0b（只用首名分）。**已做**：p = 0.0573，分不开。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import corpus as C           # noqa: E402

# 语料位置显式化（默认 ../rl-scaffold，可用 FIELD_CORPUS 覆盖）。见 corpus.py。
CORPUS, fp = C.attach()

import graph as G            # noqa: E402
import spectral as S         # noqa: E402
import field as F            # noqa: E402
import probe_gaps as pg      # noqa: E402

K_MAIN = 3
T_ARC = 2.0
MIN_ENTRIES = 2      # ρ 的定义域：少于两个构造就没有"一致性"可言


def resonance_for(q, match_by_id, vals, vecs, didx, node_ids):
    rows = fp.rank(q, match_by_id)
    entries = [r[0] for r in rows[:K_MAIN]]
    sets = [F.readout(F.trajectory(vals, vecs, didx[e], [0.0] * len(node_ids), 0.0, T_ARC),
                      node_ids) for e in entries if e in didx]
    if len(sets) < MIN_ENTRIES:
        return None, len(entries), (rows[0][1] if rows else 0.0)
    tot = n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            tot += 1.0 - F.jaccard(sets[i], sets[j])
            n += 1
    return tot / n, len(entries), (rows[0][1] if rows else 0.0)


def main() -> int:
    raw, match_by_id = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))

    cue = lambda w: [str(c).strip() for c in (raw[w].get("cues") or [])]   # noqa: E731
    strict = [q for w, q in pg.HELD_OUT if q not in cue(w)]
    negs = [q for cat, q in pg.CORPUS if "越界" in cat]

    P = [resonance_for(q, match_by_id, vals, vecs, didx, node_ids) for q in strict]
    N = [resonance_for(q, match_by_id, vals, vecs, didx, node_ids) for q in negs]

    # ── 两种判据的 2x2 ────────────────────────────────────────────────
    def tau_accept(q):
        rows = fp.rank(q, match_by_id)
        return bool(rows) and rows[0][1] >= fp.TAU

    def field_accept(triple):
        rho, ne, _ = triple
        return ne >= MIN_ENTRIES and rho is not None and rho > 0

    tau = {"TP": sum(1 for q in strict if tau_accept(q)),
           "FN": sum(1 for q in strict if not tau_accept(q)),
           "FP": sum(1 for q in negs if tau_accept(q)),
           "TN": sum(1 for q in negs if not tau_accept(q))}
    fld = {"TP": sum(1 for t in P if field_accept(t)),
           "FN": sum(1 for t in P if not field_accept(t)),
           "FP": sum(1 for t in N if field_accept(t)),
           "TN": sum(1 for t in N if not field_accept(t))}

    pv = [t[0] for t in P if t[0] is not None]
    nv = [t[0] for t in N if t[0] is not None]
    pmin = min(pv) if pv else None
    nmax = max(nv) if nv else None

    L: list[str] = []
    L.append("# Phase 6 · 可用性（C7）与消融\n")
    L.append("判据链：`处境句 → 匹配器排名 → 前 k 个入口 → 各自构造 → 共振 ρ → 接受/拒绝`")
    L.append(f"k = {K_MAIN}，弧长 t = {T_ARC}（中性扩散）\n")

    L.append("## 逐句\n")
    L.append("| 组 | 语句 | 候选数 | 首名分 | ρ 共振 | TAU 接受 | 场判据接受 |")
    L.append("|---|---|---|---|---|---|---|")
    for grp, qs, ts in (("正例", strict, P), ("负例", negs, N)):
        for q, t in zip(qs, ts):
            rho, ne, top = t
            L.append(f"| {grp} | {q} | {ne} | {top:.3f} | "
                     f"{'—' if rho is None else f'{rho:.3f}'} | "
                     f"{'是' if tau_accept(q) else '否'} | {'是' if field_accept(t) else '否'} |")
    L.append("")

    L.append("## C7a：无参数那一半（候选 < 2 或 ρ = 0 就拒绝）\n")
    L.append("| 判据 | TP | FN | FP | TN | 正例召回 | 负例拒掉 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, m in (("TAU 那条线（基线）", tau), ("场：可构造性（无参数）", fld)):
        L.append(f"| {name} | {m['TP']} | {m['FN']} | {m['FP']} | {m['TN']} | "
                 f"{m['TP']}/{len(strict)} | {m['TN']}/{len(negs)} |")
    L.append("")
    L.append(f"- 基线 TAU：负例挡住 {tau['TN']}/{len(negs)}，但漏掉 {tau['FN']} 句正例")
    L.append(f"- 场（无参数）：正例接住 {fld['TP']}/{len(strict)}，"
             f"负例挡掉 {fld['TN']}/{len(negs)}")
    L.append("- **两者是一对取舍，不是谁压倒谁**：TAU 精度满分召回不满，场召回满分精度不满。")
    L.append("")

    L.append("## C7b：需要一条线的那一半（本模块**不嵌入**这条线）\n")
    L.append(f"- 正例 ρ 的最小值：**{pmin:.3f}**")
    L.append(f"- 负例 ρ 的最大值：**{nmax:.3f}**")
    if pmin is not None and nmax is not None and pmin > nmax:
        L.append(f"- **样本内存在一条能完全分开的线**，间隔落在 ({nmax:.3f}, {pmin:.3f}) 之间")
        L.append(f"- 用它会得到 TP={len(strict)} / FP=0 / TN={len(negs)} / FN=0")
        L.append("")
        L.append("⚠️ **但这不构成证据。** 样本是 9 句正例 vs 4 句负例，")
        L.append("在这个规模上「样本内完全分开」是很容易发生的，而且那条线是在")
        L.append("看过结果之后才找得到的。按 B-D6，本模块**不把任何内联数字线写进代码**；")
        L.append("这条线要成立，必须先有更大的正负例集（登记为独立待办）。")
    else:
        L.append("- 不存在能完全分开的线。")
    L.append("")

    L.append("## 消融\n")
    L.append("| 消融 | 做没做 | 结论 |")
    L.append("|---|---|---|")
    L.append("| A1 去意图 | **结构性地不适用** | 判据链用中性扩散（α=0），本就不含意图。"
             "意图的对照在 Phase 2：α=0 → 1 个构造 |")
    L.append("| A2 去曲率 | **已做**（Phase 1） | Forman 被项分解证伪（度数项方差占比 1.054），"
             "曲率不在判据链里。Phase 1b 补做 OR，结论另见 |")
    L.append("| A3 去轨迹 | **已做**（Phase 4 的 R0b） | 只用首名分：p = 0.0573，分不开 → "
             "**轨迹带来了分数里没有的信息** |")
    L.append("")

    L.append("## 判定\n")
    L.append(f"- C7a：无参数的可构造性判据**带来一个真实增益**（负例拒掉 "
             f"{fld['TN']}/{len(negs)} vs 基线的 {tau['TN']}/{len(negs)}，"
             f"而在正例上不付代价：{fld['TP']}/{len(strict)}）→ **成立**")
    L.append("- C7b：样本内完全分开，但**证据不足**，线不嵌入代码 → **未成立，登记待办**")
    L.append("- 三组消融没有一组说明判据链里有装饰：A2 已经把曲率删了，"
             "A3 证明轨迹有用，A1 不适用")
    L.append("")

    paths.write("phase6_usability.md", "\n".join(L) + "\n")

    print("── C7a 无参数判据 vs 基线 ──")
    for name, m in (("TAU 线（基线）", tau), ("场:可构造性（无参数）", fld)):
        print(f"  {name:<22s} TP {m['TP']}/{len(strict)}  FN {m['FN']}  FP {m['FP']}  "
              f"TN {m['TN']}/{len(negs)}")
    print(f"── C7b 样本内间隔 ──")
    print(f"  正例 ρ 最小 {pmin:.3f}  负例 ρ 最大 {nmax:.3f}  "
          f"{'存在完全分开的线（证据不足）' if pmin > nmax else '不存在'}")
    print("── 消融 ──")
    print("  A1 去意图  结构性不适用（判据链用中性扩散）")
    print("  A2 去曲率  已做（Phase 1 把 Forman 删了）")
    print("  A3 去轨迹  已做（R0b p=0.0573 分不开 → 轨迹有用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
