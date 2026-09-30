"""Phase 4 跑腿：可判据性（C5）—— go / no-go。

问题：**不靠内容真值，能不能判断一个构造有没有根据。**

---
ρ 的定义：先登记，再看结果
--------------------------

设计稿 §3 Phase 4 把 ρ 定义成「共识区步数 / 总步数」，而"共识区"原本定义为 `κ > 0`。
**Phase 1 已经把这条证伪**（Forman 在本图上是度数代理，方差占比 1.054；36/47 条边为负）。

所以 ρ 必须改用存活的量。三条候选全报，主判据只取一条：

    R1（主判据·共振）  从分数最高的 k 个入口各自构造，看这些构造彼此**同不同意**。
                       ρ_R1 = k 个读数两两 Jaccard 的平均。

                       —— 这正是设计稿 §三 激活第 3 条写的那件事：
                          「多张局部地图同时激活，发生共振干涉」。

    R0a（对照·定义存疑） ρ_R0a = 第一名分数 / 前 k 名分数之和。
    R0b（对照·更强）     ρ_R0b = 第一名分数本身。

    R2（次要·桥）        读数里与入口同落在一个 2-边连通分量里的比例。

---
⚠️ 修正一：构不出来 ≠ 共振低，但两者的 ρ 都是"没有共振"
--------------------------------------------------------

首次跑完发现：4 条负例里只有 **2 条**算得出 ρ_R1——
「公司不给我批算力」只排得出 1 个候选（凑不出两两比较），
「帮我写一首诗」一个候选都没有（与全库零重合）。

把这两种情况**从统计里剔掉**等于说"它们不算数"，那是错的：
**构不出构造就是最彻底的不共振**。所以主判据把"构不出来"记为 ρ_R1 = 0
（这不是为凑 p 值而定的值，是定义：没有两个构造就没有一致性）。
剔掉的那种算法（严格法）同时报出来。

---
⚠️ 修正二：R0a 是个稻草人，所以补了 R0b
----------------------------------------

首次跑完发现负例的 R0a 反而更高（0.572 vs 0.477）。查下来是定义 artifact：
分数普遍接近零时，`s1/Σs` 反而接近 1。所以 R0a 不是一个公平的对手。
补 R0b = 第一名分数本身，这是最直接的"只用分数"判据。

---
判决（提前写死，一个字不改）
----------------------------

    C5 成立  <=>  ρ_R1(正例) 显著高于 ρ_R1(负例)  **且**  R1 不劣于 R0b

    · 显著：精确置换检验（枚举全部分组，不抽样）
    · 正例只用**真正改写的 9 句**（B-D7 命中的 8 句 cue 原文单独报，不混入）
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import corpus as C           # noqa: E402

# 语料位置显式化（默认 ../rl-scaffold，可用 FIELD_CORPUS 覆盖）。见 corpus.py。
CORPUS, fp = C.attach()

import graph as G            # noqa: E402
import spectral as S         # noqa: E402
import field as F            # noqa: E402
import connectivity as CX    # noqa: E402
import probe_gaps as pg      # noqa: E402

# 显著性水平。**给它起名，不给它豁免**（B-D6 的规矩）。
# 0.05 是统计惯例，不是为本图调出来的线——换任何一张图都还是 0.05。
ALPHA = 0.05

K_MAIN = 3
T_ARC = 2.0


def mean_jaccard(sets: list[frozenset]):
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return None
    return sum(1.0 - F.jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def rho_for_entries(entries, vals, vecs, didx, node_ids):
    sets = [F.readout(F.trajectory(vals, vecs, didx[e], [0.0] * len(node_ids), 0.0, T_ARC),
                      node_ids) for e in entries if e in didx]
    return mean_jaccard(sets), sets


def measure(queries, match_by_id, vals, vecs, didx, node_ids, k=K_MAIN):
    """每句返回 (R1严格, R1把构不出记为0, R0a, R0b, 入口数)。"""
    out = []
    for q in queries:
        rows = fp.rank(q, match_by_id)
        entries = [r[0] for r in rows[:k]]
        match_values = [r[1] for r in rows[:k]]
        r1, _sets = rho_for_entries(entries, vals, vecs, didx, node_ids)
        r0b = match_values[0] if match_values else 0.0
        tot = sum(match_values)
        r0a = (match_values[0] / tot) if tot else None
        out.append((r1, 0.0 if r1 is None else r1, r0a, r0b, len(entries)))
    return out


def exact_perm_p(pos: list[float], neg: list[float]):
    pool = pos + neg
    n1 = len(pos)
    if not pos or not neg:
        return None, None
    obs = sum(pos) / len(pos) - sum(neg) / len(neg)
    ge = total = 0
    for idx in combinations(range(len(pool)), n1):
        s = set(idx)
        b = [pool[i] for i in range(len(pool)) if i not in s]
        if not b:
            continue
        total += 1
        a = [pool[i] for i in idx]
        d = sum(a) / len(a) - sum(b) / len(b)
        if abs(d) >= abs(obs):
            ge += 1
    return obs, ((ge + 1) / (total + 1) if total else 1.0)


def main() -> int:
    raw, match_by_id = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    zeros = [0.0] * len(node_ids)

    br = CX.bridges(adj)
    comps = CX.edge_connected_components(adj, br)
    pend = [c[0] for c in comps if len(c) == 1]

    cue = lambda w: [str(c).strip() for c in (raw[w].get("cues") or [])]   # noqa: E731
    strict = [(w, q) for w, q in pg.HELD_OUT if q not in cue(w)]
    verbatim = [(w, q) for w, q in pg.HELD_OUT if q in cue(w)]
    negs = [q for cat, q in pg.CORPUS if "越界" in cat]

    pos_q = [q for _, q in strict]
    pos_v = [q for _, q in verbatim]
    M = {
        "正例（改写 9）": measure(pos_q, match_by_id, vals, vecs, didx, node_ids),
        "正例（原文 8）": measure(pos_v, match_by_id, vals, vecs, didx, node_ids),
        "负例（越界 4）": measure(negs, match_by_id, vals, vecs, didx, node_ids),
    }

    L: list[str] = []
    L.append("# Phase 4 · 可判据性（C5）\n")
    L.append(f"- 弧长 t = {T_ARC}（中性扩散 α=0）；主判据 k = {K_MAIN}")
    L.append(f"- 正例 = **真正改写的 {len(strict)} 句**；另 {len(verbatim)} 句 cue 原文单独报")
    L.append(f"- 负例 = 越界 {len(negs)} 句")
    L.append("- 置换检验：**精确枚举全部分组**，不抽样\n")

    L.append("## 图的精确连通性（R2 的依据，先说破它先天没有区分力）\n")
    L.append(f"- 边 {sum(len(a) for a in adj.values())//2} 条，其中**桥 {len(br)} 条**")
    L.append(f"- 2-边连通分量 {len(comps)} 个，其中**单点分量 {len(pend)} 个**")
    L.append(f"    单点：{'、'.join(pend)}")
    L.append(f"- 正例目标 judge-0010 / judge-0011 是单点；负例 argmax judge-0008 也是单点")
    L.append("  → **R2 在本测试床上先天分不开**，这一句必须写在结果前面。\n")

    L.append("## 逐句读数\n")
    L.append("| 组 | 语句 | 入口数 | R1 严格 | **R1 构不出记0** | R0a 分数占比 | R0b 首名分 |")
    L.append("|---|---|---|---|---|---|---|")
    for label in ("正例（改写 9）", "正例（原文 8）", "负例（越界 4）"):
        qs = pos_q if label.startswith("正例（改写") else pos_v if label.startswith("正例（原文") else negs
        for q, (r1s, r1z, r0a, r0b, ne) in zip(qs, M[label]):
            L.append(f"| {label} | {q} | {ne} | "
                     f"{'—' if r1s is None else f'{r1s:.3f}'} | **{r1z:.3f}** | "
                     f"{'—' if r0a is None else f'{r0a:.3f}'} | {r0b:.3f} |")
    L.append("")

    # ── 判决 ──────────────────────────────────────────────────────────
    def col(group, i):
        return [x[i] for x in M[group] if x[i] is not None]

    L.append("## 判决\n")
    L.append("| 判据 | 正例均值 | 负例均值 | 差值 | 精确 p | 结论 |")
    L.append("|---|---|---|---|---|---|")
    V = {}
    for name, i in (("R1 严格（剔掉构不出的）", 0), ("**R1 构不出记0（主判据）**", 1),
                    ("R0a 分数占比（稻草人）", 2), ("R0b 首名分（强对照）", 3)):
        p, n = col("正例（改写 9）", i), col("负例（越界 4）", i)
        if not p or not n:
            L.append(f"| {name} | — | — | — | — | 数据不足 |")
            V[name] = None
            continue
        d, pv = exact_perm_p(p, n)
        V[name] = (len(p), len(n), sum(p) / len(p), sum(n) / len(n), d, pv)
        L.append(f"| {name} | {sum(p)/len(p):.3f} | {sum(n)/len(n):.3f} | {d:+.3f} | {pv:.4f} | "
                 f"{'分开' if pv < ALPHA else '**分不开**'} |")
    L.append("")

    v1 = V.get("**R1 构不出记0（主判据）**")
    v0 = V.get("R0b 首名分（强对照）")
    sep = bool(v1 and v1[5] < ALPHA)
    beat = bool(v1 and v0 and v0[5] is not None and v1[5] <= v0[5])
    L.append(f"- 正例 {v1[0]} vs 负例 {v1[1]}（样本很小，p 的分辨率上限是 "
             f"1/{len(list(combinations(range(v1[0]+v1[1]), v1[0])))+1:.0f}）")
    L.append(f"- R1 显著分开：**{sep}**（p = {v1[5]:.4f}）")
    L.append(f"- R1 不劣于 R0b：**{beat}**（R1 p = {v1[5]:.4f} vs R0b p = "
             f"{'—' if v0[5] is None else format(v0[5],'.4f')}）")
    L.append(f"\n**C5 {'成立' if (sep and beat) else '不成立'}**\n")

    # ── 针对性比较：分数分不开的那两句 ────────────────────────────────
    L.append("## 针对性比较：**分数分不开的那两句**，共振分不分得开\n")
    L.append("Phase 3 已量出：没过 TAU 的 2 句正例分数是 0.138，而 4 条负例最高 0.150。")
    L.append("**光看分数这两个区间重叠且负例更高。** 现在只看这 6 句：\n")
    L.append("| 语句 | 组 | 首名分 R0b | R1 共振（构不出记0） |")
    L.append("|---|---|---|---|")
    hard = [(q, m) for q, m in zip(pos_q, M["正例（改写 9）"]) if m[3] < fp.TAU]
    hard += [(q, m) for q, m in zip(negs, M["负例（越界 4）"])]
    for q, m in hard:
        grp = "负例" if q in negs else "正例"
        L.append(f"| {q} | {grp} | {m[3]:.3f} | **{m[1]:.3f}** |")
    hp = [m[1] for q, m in hard if q not in negs]
    hn = [m[1] for q, m in hard if q in negs]
    L.append("")
    L.append(f"- 分数落在 TAU 以下的**正例**（{len(hp)} 句）：R1 = {[f'{x:.3f}' for x in hp]}")
    L.append(f"- **负例**（{len(hn)} 句）：R1 = {[f'{x:.3f}' for x in hn]}")
    if hp and hn:
        d, pv = exact_perm_p(hp, hn)
        L.append(f"- 差值 {d:+.3f}，精确 p = {pv:.4f}")
        L.append(f"- **这是全场最要命的一格：分数说不了话的地方，"
                 f"共振{'说得了' if pv < ALPHA else '也说不了'}。**")
    L.append("")

    # ── 损伤实验（修掉入口集混淆）──────────────────────────────────────
    L.append("## 损伤实验：删掉正确节点，共振应该掉\n")
    L.append("⚠️ 首次跑完后发现旧设计有混淆：删掉目标节点后，前 k 名入口本身也变了，")
    L.append("所以 ρ 的变化分不清是「结构被破坏」还是「换了一批入口」。")
    L.append("**修法：入口集在损伤前后固定**——取原排名里前 k 个**不等于目标**的节点。")
    L.append("`nodes/` 一个字不动。\n")
    L.append("| 目标 | 度数 | 固定入口 | 原 ρ_R1 | 损伤后 | 变化 |")
    L.append("|---|---|---|---|---|---|")
    lesion = []
    for target in ["judge-0010", "judge-0011", "judge-0012", "judge-0013"]:
        qs = [q for w, q in strict if w == target] + [q for w, q in verbatim if w == target]
        if not qs:
            continue
        # 固定入口：每句取原排名里前 k 个 != target
        fixed = []
        for q in qs:
            rows = fp.rank(q, match_by_id)
            fixed.append([r[0] for r in rows if r[0] != target][:K_MAIN])
        b = [rho_for_entries(e, vals, vecs, didx, node_ids)[0] for e in fixed]
        b = [x for x in b if x is not None]
        keep = [v for v in node_ids if v != target]
        adj2 = G.subgraph(G.build(raw)["adj"], keep)
        vals2, vecs2 = S.jacobi(S.laplacian(adj2, keep))
        didx2 = {v: i for i, v in enumerate(keep)}
        a = [rho_for_entries([e for e in ent if e in didx2], vals2, vecs2, didx2, keep)[0]
             for ent in fixed]
        a = [x for x in a if x is not None]
        mb = sum(b) / len(b) if b else None
        ma = sum(a) / len(a) if a else None
        L.append(f"| {target} | {len(adj[target])} | {'、'.join(fixed[0])} | "
                 f"{'—' if mb is None else f'{mb:.3f}'} | {'—' if ma is None else f'{ma:.3f}'} | "
                 f"{'—' if (mb is None or ma is None) else f'{ma-mb:+.3f}'} |")
        lesion.append((target, len(adj[target]), mb, ma))
    L.append("")
    drops = [1 for _, _, mb, ma in lesion if mb is not None and ma is not None and ma < mb]
    L.append(f"- 预测：删掉正确节点后 ρ_R1 下降。**{len(drops)}/{len(lesion)} 个目标符合预测。**")
    L.append("- 幅度很小（0.0x），所以这个实验**只能算弱证据**，不能算确认。\n")

    paths.report("phase4_grounding.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── 终端 ──────────────────────────────────────────────────────────
    print("══ 图结构 ══")
    print(f"桥 {len(br)}/47；2-边连通分量 {len(comps)}；单点分量 {len(pend)}")
    print("══ 判决 ══")
    for name, i in (("R1 严格", 0), ("R1 构不出记0（主）", 1), ("R0a 分数占比", 2), ("R0b 首名分", 3)):
        p, n = col("正例（改写 9）", i), col("负例（越界 4）", i)
        if p and n:
            d, pv = exact_perm_p(p, n)
            print(f"  {name:<18s} 正 {sum(p)/len(p):.3f}(n={len(p)})  负 {sum(n)/len(n):.3f}(n={len(n)})  "
                  f"差 {d:+.3f}  p={pv:.4f}  {'分开' if pv < ALPHA else '分不开'}")
    print(f"══ C5 {'成立' if (sep and beat) else '不成立'} ══")
    print("══ 分数分不开的那几句 ══")
    for q, m in hard:
        print(f"  {'负例' if q in negs else '正例'}  分数 {m[3]:.3f}  共振 {m[1]:.3f}  {q}")
    if hp and hn:
        d, pv = exact_perm_p(hp, hn)
        print(f"  这 6 句上：共振差 {d:+.3f}  p={pv:.4f}")
    print("══ 损伤实验（固定入口）══")
    for target, dg, mb, ma in lesion:
        print(f"  {target} 度数 {dg}  ρ {mb:.3f} -> {ma:.3f}  变化 {ma-mb:+.3f}")
    print(f"  符合预测 {len(drops)}/{len(lesion)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
