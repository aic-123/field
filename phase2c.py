"""Phase 2 重做：把意图换成设计稿本来写的那种（有符号种子向量）。

---
为什么重做，以及为什么这不是 p-hacking
--------------------------------------

`phase2.py` 报 C2 不成立。`phase2b_readout.py` 一诊断，发现**检验本身是坏的**：

    意图取的是正交归一的特征方向 S_d = λ_d·u_d，于是 L⁺S_d = u_d。
    而 u_d 两两正交且都是单位向量，所以**任意两个意图之间的平衡场距离都是 √2**。

    结论：意图空间的"距离"是常数矩阵，里面没有任何排序信息。
    `phase2.py` 里那个 Mantel r 是在**特征向量正交性的浮点残差**上算出来的，
    值从 +0.03 到 +0.48 乱跳、p 从 0.03 到 0.85——那全是噪声，不是信号。

    （这正是上游 `nested/DECLARATION.md` §22.8 记的那一类
     "failure-looks-like-success" bug：数出来了，但是假的。）

还发现了第二个、更实质的退化：

    对特征向量做意图时，场距离满足 d(i,j)² = a_i + a_j（a 只依赖各自的 λ）。
    这是**加性**距离——所有意图的最近邻是同一个，意图之间没有互相的结构。

所以必须换意图。**而设计稿本来写的就是另一种**（§1.3）：

    「意图 = 有符号的种子向量：正分量 = 要往那边走，负分量 = 要避开」

本模块就按这句实现，取最简单且不退化的一族：

    S_A = e_A − mean        「把构造拉向节点 A，从其余节点平均地推开」

这一族有一个干净的性质：

    L⁺S_A = L⁺e_A（因为常数向量在 L 的零空间里）
    |L⁺S_A − L⁺S_B|² = R(A,B) = **有效电阻距离**

也就是说：**意图空间的参考几何，正好就是 Phase 1 算出来的那个度量。**
不是另造一把尺子，是把尺子接上。

于是 C2 的问题变成一句可证伪的话：

    **构造的几何，能不能恢复有效电阻的几何？**

⚠️ 判据与 `phase2.py` 完全一致（Mantel r + 置换 p、以及三层恢复率），
一个字都没放宽。换的只是意图族——而且换成的是设计稿原本指定的那一种。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402
from phase2b_readout import dmats, recovery, spread, nn   # noqa: E402

VAR_TOL = 1e-300
PERMUTATIONS = 2000
N_ANCHORS = 16


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= VAR_TOL or vb <= VAR_TOL:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va ** 0.5 * vb ** 0.5)


def upper(mat: list[list[float]]) -> list[float]:
    return [mat[i][j] for i in range(len(mat)) for j in range(i + 1, len(mat))]


def mantel(ref_d: list[list[float]], con_d: list[list[float]], seed: int = 424242):
    a, b = upper(ref_d), upper(con_d)
    r = pearson(a, b)
    k = len(con_d)
    ge = 0
    for s in range(PERMUTATIONS):
        p = list(range(k))
        st = seed + s * 7919
        for i in range(k - 1, 0, -1):
            st = _xorshift(st)
            j = st % (i + 1)
            p[i], p[j] = p[j], p[i]
        pb = [con_d[p[i]][p[j]] for i in range(k) for j in range(i + 1, k)]
        if abs(pearson(a, pb)) >= abs(r):
            ge += 1
    return r, (ge + 1) / (PERMUTATIONS + 1)


def main() -> int:
    raw, _m = G.load_nodes()
    types = {k: str(raw[k].get("type") or "?") for k in raw}
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    deg = G.degrees(adj)
    order_deg = sorted(node_ids, key=lambda v: (-deg[v], v))
    entries = [order_deg[0], order_deg[-1], order_deg[len(order_deg) // 2]]

    # 锚点：按 id 均匀取，保证跨类型铺开。确定性。
    stride = max(1, n // N_ANCHORS)
    anchors = sorted(node_ids)[::stride][:N_ANCHORS]

    # 意图 S_A = e_A − mean
    intents = []
    for a in anchors:
        v = [1.0 if x == a else 0.0 for x in node_ids]
        m = sum(v) / n
        intents.append([x - m for x in v])

    # 参考几何 = 有效电阻距离（Phase 1 的度量）
    eqs = [F.pinv_apply(vals, vecs, s) for s in intents]
    R = S.effective_resistance(vals, vecs, node_ids)
    refR = [[R["R"][didx[a]][didx[b]] for b in anchors] for a in anchors]
    ref_eq = dmats(eqs, F.l2)

    L: list[str] = []
    L.append("# Phase 2（重做）· 可控性 C2 / C3\n")
    L.append("意图族：`S_A = e_A − mean`（把构造拉向 A）。")
    L.append("参考几何：**有效电阻距离 R(A,B)**——意图空间的度量就是 Phase 1 的度量。\n")
    L.append(f"- 锚点 {len(anchors)} 个：{'、'.join(anchors)}")
    L.append(f"- 锚点类型：{'、'.join(types[a] for a in anchors)}")
    L.append(f"- 置换次数 {PERMUTATIONS}\n")

    # 意图空间的退化检查（上一轮就死在这里）
    off = [refR[i][j] for i in range(len(anchors)) for j in range(i + 1, len(anchors))]
    L.append("## 先检查意图空间是否退化（上一轮的死因）\n")
    L.append(f"- 有效电阻距离：min {min(off):.4f} / 中位 {sorted(off)[len(off)//2]:.4f} / max {max(off):.4f}")
    L.append(f"- 标准差 {spread(refR)['sd']:.4f}")
    L.append(f"- 不同取值个数 {len(set(round(x, 6) for x in off))} / {len(off)} 对")
    L.append(f"- 平衡场距离的 sd {spread(ref_eq)['sd']:.4f}")
    L.append("\n（上一轮这两个 sd 都接近 0，因为特征方向两两正交、距离恒为 √2。）\n")

    # ── C3 ────────────────────────────────────────────────────────────
    e0 = entries[0]
    s0 = intents[0]
    a1 = F.trajectory(vals, vecs, didx[e0], s0, 1.0, 1.0)
    a2 = F.trajectory(vals, vecs, didx[e0], s0, 1.0, 1.0)
    L.append("## C3 确定性\n")
    L.append(f"- 同一 (入口, 意图, α=1, t=1) 两次：逐元素相同 = **{a1 == a2}**")
    L.append(f"- 读数相同 = **{F.readout(a1,node_ids) == F.readout(a2,node_ids)}**\n")

    # ── α=0 对照 ─────────────────────────────────────────────────────
    ctrl = {F.readout(F.trajectory(vals, vecs, didx[e0], s, 0.0, 1.0), node_ids) for s in intents}
    L.append("## 对照：α=0（意图不进入）\n")
    L.append(f"- {len(intents)} 个意图 → **{len(ctrl)}** 个不同构造（必须是 1）\n")

    # ── C2 ────────────────────────────────────────────────────────────
    L.append("## C2 · 三层恢复率 + Mantel（对硬读数）\n")
    L.append("| 入口 | α | t | 不同构造 | 场恢复 | 软恢复 | 硬恢复 | 硬 Mantel r | p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for e in entries:
        for alpha in (1.0, 2.0):
            for t in (0.5, 2.0, 20.0):
                fields = [F.trajectory(vals, vecs, didx[e], s, alpha, t) for s in intents]
                sets = [F.readout(f, node_ids) for f in fields]
                soft = [[f[j] if node_ids[j] in sets[i] else 0.0 for j in range(n)]
                        for i, f in enumerate(fields)]
                fd = dmats(fields, F.l2)
                sd = dmats(soft, F.l2)
                hd = dmats(sets, F.jaccard)
                a0, a1r, a2r = recovery(fd, ref_eq), recovery(sd, ref_eq), recovery(hd, ref_eq)
                r, p = mantel(refR, hd)
                nd = len(set(sets))
                L.append(f"| {e} | {alpha} | {t} | {nd}/{len(intents)} | {a0:.3f} | {a1r:.3f} "
                         f"| **{a2r:.3f}** | {r:+.3f} | {p:.4f} |")
                rows.append((e, alpha, t, nd, a0, a1r, a2r, r, p))
    L.append("")

    # ── 旋钮 ──────────────────────────────────────────────────────────
    L.append("## 弧长 t（入口=%s, α=1）\n" % e0)
    L.append("| t | 读数规模 | 不同构造 |")
    L.append("|---|---|---|")
    for t in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
        ss, cc = [], set()
        for s in intents:
            rd = F.readout(F.trajectory(vals, vecs, didx[e0], s, 1.0, t), node_ids)
            ss.append(len(rd)); cc.add(rd)
        L.append(f"| {t} | {min(ss)}–{max(ss)} | {len(cc)}/{len(intents)} |")
    L.append("")
    L.append("## 强度 α（入口=%s, t=2）\n" % e0)
    L.append("| α | 读数规模 | 不同构造 |")
    L.append("|---|---|---|")
    for alpha in (0.0, 0.5, 1.0, 2.0, 5.0):
        ss, cc = [], set()
        for s in intents:
            rd = F.readout(F.trajectory(vals, vecs, didx[e0], s, alpha, 2.0), node_ids)
            ss.append(len(rd)); cc.add(rd)
        L.append(f"| {alpha} | {min(ss)}–{max(ss)} | {len(cc)}/{len(intents)} |")
    L.append("")

    paths.report("phase2c_controllability.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"锚点 {len(anchors)}：{'、'.join(anchors)}")
    print(f"意图空间（有效电阻）sd = {spread(refR)['sd']:.4f}   "
          f"不同取值 {len(set(round(x,6) for x in off))}/{len(off)}")
    print(f"C3 确定性 = {a1 == a2}")
    print(f"α=0 → {len(ctrl)} 个不同构造（必须 1）")
    print("── 三层恢复率 + Mantel ──")
    for e, alpha, t, nd, a0, a1r, a2r, r, p in rows:
        print(f"  {e:<10s} α={alpha:<4} t={t:<5} 构造{nd:2d}/{len(intents)}  "
              f"场{a0:.3f} 软{a1r:.3f} 硬{a2r:.3f}  r={r:+.3f} p={p:.4f}")
    print(f"随机基线 1/{len(intents)} = {1/len(intents):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
