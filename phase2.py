"""Phase 2 跑腿（**这一版作废，留着是为了可审计**）：可控性（C2 / C3）。

设计稿 §3 Phase 2 要验三件事：

    C3  确定性：同一 (入口, 意图, 弧长) 两次 → 同一构造
    C2  可控性：不同意图 → 不同且**可预测**的构造
    旋钮 弧长 t 与强度 α 是否真的起作用

⚠️ C2 有两个层次，必须分开报，否则会自己骗自己：

    C2-恒真层  不同意图给出不同的**场**          —— 线性 ODE，恒真，测了不证明什么
    C2-实质层  不同意图给出不同的**读数（构造）** —— 读数带选择，会丢信息，这一层才有内容

本模块两层都算，并且**用 α=0 做对照**：α=0 时意图完全不进入，构造必须全同。
如果 α=0 时构造还不全同，说明扰动来自别处，C2 的读数作废。

---
⚠️⚠️ 事后记录：本模块的 C2 结论**作废**，因为检验本身是坏的
----------------------------------------------------------------

诊断见 `phase2b_readout.py`。两个问题：

一、意图取的是正交归一的特征方向 `S_d = λ_d·u_d`，于是 `L⁺S_d = u_d`，
    而 `u_d` 两两正交且都是单位向量 —— **任意两个意图之间的平衡场距离恒为 √2**。
    所以意图空间距离是**常数矩阵**，本模块那个 Mantel r 是在特征向量正交性的
    **浮点残差**上算出来的：r 从 +0.03 跳到 +0.48，p 从 0.03 到 0.85，全是噪声。

二、场距离满足 `d(i,j)² = a_i + a_j`（a 只依赖各自的 λ），是**加性**的：
    所有意图的最近邻是同一个，意图之间没有互相的结构。

这正是上游 `nested/DECLARATION.md` §22.8 记的那类 **failure-looks-like-success**。

**有效的那一版在 `phase2c.py`**：意图换成设计稿 §1.3 本来指定的有符号种子向量。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402

PERMUTATIONS = 2000
T_GRID = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
A_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
N_INTENTS = 12


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / (va ** 0.5 * vb ** 0.5)


def _xorshift(state: int) -> int:
    """确定性伪随机：不依赖 random 模块的版本行为。"""
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def _perm(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    st = seed or 88172645463325252
    for i in range(n - 1, 0, -1):
        st = _xorshift(st)
        j = st % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def upper(mat: list[list[float]]) -> list[float]:
    out = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            out.append(mat[i][j])
    return out


def mantel(intent_d: list[list[float]], con_d: list[list[float]], seed: int = 12345):
    a = upper(intent_d)
    b = upper(con_d)
    r = pearson(a, b)
    n = len(con_d)
    ge = 0
    for s in range(PERMUTATIONS):
        p = _perm(n, seed + s * 7919)
        perm_b = []
        for i in range(n):
            for j in range(i + 1, n):
                perm_b.append(con_d[p[i]][p[j]])
        rp = pearson(a, perm_b)
        if abs(rp) >= abs(r):
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

    # 意图 = 前 N_INTENTS 个非平凡扩散方向，归一到"等射程"（|L⁺S| = 1）。
    # L⁺u_d = u_d/λ_d，范数 1/λ_d，所以乘 λ_d 即归一。
    nontrivial = [k for k in range(n) if vals[k] > S.ZERO_TOL]
    intents: list[tuple[int, list[float]]] = []
    for k in nontrivial[:N_INTENTS]:
        u = [vecs[i][k] for i in range(n)]
        intents.append((k, [x * vals[k] for x in u]))

    # 入口：最高度、最低度、一个中等的
    deg = G.degrees(adj)
    order_deg = sorted(node_ids, key=lambda v: (-deg[v], v))
    entries = [order_deg[0], order_deg[-1], order_deg[len(order_deg) // 2]]

    L: list[str] = []
    L.append("# Phase 2 · 可控性（C2 / C3）—— ⚠️ **本版作废**\n")
    L.append("意图取的是正交归一的特征方向，导致意图空间距离是**常数矩阵**，")
    L.append("Mantel r 测的是浮点残差。有效版本见 `phase2c_controllability.md`。\n")
    L.append(f"- 意图数 {len(intents)}（前 {N_INTENTS} 个非平凡扩散方向，等射程归一）")
    L.append(f"- 入口：{', '.join(f'{e}({types.get(e,'?')},d={deg[e]})' for e in entries)}")
    L.append(f"- 置换次数 {PERMUTATIONS}（确定性伪随机）\n")

    # ── C3 确定性 ─────────────────────────────────────────────────────
    e0 = entries[0]
    s0 = intents[0][1]
    a1 = F.trajectory(vals, vecs, didx[e0], s0, 1.0, 1.0)
    a2 = F.trajectory(vals, vecs, didx[e0], s0, 1.0, 1.0)
    det_exact = a1 == a2
    r1 = F.readout(a1, node_ids)
    r2 = F.readout(a2, node_ids)
    L.append("## C3 确定性\n")
    L.append(f"- 同一 (入口={e0}, 意图=d{intents[0][0]}, α=1, t=1) 跑两次：逐元素相同 = **{det_exact}**")
    L.append(f"- 读数相同 = **{r1 == r2}**（{len(r1)} 个节点）\n")

    # ── α=0 对照 ─────────────────────────────────────────────────────
    L.append("## 对照：α=0（意图不进入）\n")
    ctrl = set()
    for k, s in intents:
        kt = F.trajectory(vals, vecs, didx[e0], s, 0.0, 1.0)
        ctrl.add(F.readout(kt, node_ids))
    L.append(f"- α=0 时，{len(intents)} 个意图给出 **{len(ctrl)} 个不同的构造**")
    L.append(f"- 必须是 1。否则扰动不来自意图，C2 的读数作废。\n")

    # ── 意图空间的退化检查（这是本版死掉的地方）────────────────────────
    eqs = [F.pinv_apply(vals, vecs, s) for _k, s in intents]
    idm = [[F.l2(eqs[i], eqs[j]) for j in range(len(eqs))] for i in range(len(eqs))]
    flat = upper(idm)
    L.append("## 意图空间是否退化（本版的死因）\n")
    L.append(f"- 平衡场两两距离：最小 {min(flat):.6f}，最大 {max(flat):.6f}")
    L.append(f"- 都是 √2 ≈ 1.414214（因为特征向量两两正交且为单位向量）")
    L.append(f"- **所以意图空间距离是常数矩阵，下面所有 Mantel r 都是噪声。**\n")

    # ── C2 实质层：Mantel 检验 ────────────────────────────────────────
    L.append("## C2 实质层：意图 → 构造（Mantel 置换检验）—— 全部作废\n")
    L.append("| 入口 | α | t | 不同构造数 | Mantel r | p |")
    L.append("|---|---|---|---|---|---|")
    summary = []
    for e in entries:
        for alpha in (1.0, 2.0):
            for t in (0.5, 2.0, 20.0):
                con = []
                for k, s in intents:
                    kt = F.trajectory(vals, vecs, didx[e], s, alpha, t)
                    con.append(F.readout(kt, node_ids))
                cdm = [[F.jaccard(con[i], con[j]) for j in range(len(con))] for i in range(len(con))]
                r, p = mantel(idm, cdm)
                ndist = len(set(con))
                L.append(f"| {e} | {alpha} | {t} | {ndist}/{len(intents)} | {r:+.3f} | {p:.4f} |")
                summary.append((e, alpha, t, ndist, r, p))
    L.append("")

    # ── 弧长旋钮 ──────────────────────────────────────────────────────
    L.append("## 弧长 t 旋钮（入口=%s, α=1）\n" % e0)
    L.append("| t | 读数规模 | 不同构造数 |")
    L.append("|---|---|---|")
    for t in T_GRID:
        sizes = []
        cons = set()
        for k, s in intents:
            kt = F.trajectory(vals, vecs, didx[e0], s, 1.0, t)
            rd = F.readout(kt, node_ids)
            sizes.append(len(rd))
            cons.add(rd)
        L.append(f"| {t} | {min(sizes)}–{max(sizes)} | {len(cons)}/{len(intents)} |")
    L.append("")

    # ── 强度旋钮 ──────────────────────────────────────────────────────
    L.append("## 强度 α 旋钮（入口=%s, t=2）\n" % e0)
    L.append("| α | 读数规模 | 不同构造数 |")
    L.append("|---|---|---|")
    for alpha in A_GRID:
        sizes = []
        cons = set()
        for k, s in intents:
            kt = F.trajectory(vals, vecs, didx[e0], s, alpha, 2.0)
            rd = F.readout(kt, node_ids)
            sizes.append(len(rd))
            cons.add(rd)
        L.append(f"| {alpha} | {min(sizes)}–{max(sizes)} | {len(cons)}/{len(intents)} |")
    L.append("")

    # ── C2-恒真层（对照用，说明它不证明什么）───────────────────────────
    L.append("## C2-恒真层（仅供参考，不作为证据）\n")
    fields = [F.trajectory(vals, vecs, didx[e0], s, 1.0, 2.0) for k, s in intents]
    dmin = min(F.l2(fields[i], fields[j]) for i in range(len(fields)) for j in range(i + 1, len(fields)))
    L.append(f"- {len(fields)} 个意图给出的**场**之间，最小距离 {dmin:.6f}")
    L.append("- 这一层恒真（线性 ODE + L⁺ 在常数补上可逆），所以它**不作为 C2 的证据**。\n")

    paths.report("phase2_controllability.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── 终端摘要 ──────────────────────────────────────────────────────
    print("── C3 确定性 ──")
    print(f"逐元素相同 = {det_exact}；读数相同 = {r1 == r2}")
    print("── α=0 对照 ──")
    print(f"{len(intents)} 个意图 → {len(ctrl)} 个不同构造（必须是 1）")
    print("── 意图空间退化（本版死因）──")
    print(f"平衡场两两距离 min {min(flat):.6f} max {max(flat):.6f} → 常数矩阵")
    print("── C2 实质层（全部作废）──")
    for e, alpha, t, ndist, r, p in summary:
        print(f"  {e:<10s} α={alpha:<4} t={t:<5} 不同构造 {ndist:2d}/{len(intents)}  r={r:+.3f}  p={p:.4f}")
    print("=> 本版作废，有效版本见 phase2c.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
