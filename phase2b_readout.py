"""Phase 2 诊断：C2 失败到底是"控制坏了"还是"读数坏了"。

---
为什么必须做这一步，而不是直接换读数
------------------------------------

`phase2.py` 是**预先写死**的检验，它报 C2 不成立。如果这时直接换一个读数
再跑，跑出一个好看的数就宣布通过，那是 p-hacking。**不能那样做。**

正确做法是先定位失败在哪一层。构造的定义是一条复合链：

    意图 S ──线性ODE──▶ 场 K ──选择──▶ 软读数 ──丢幅度──▶ 硬读数（集合）

三层的信息损失完全不同，逐层量：

    第 0 层  场          K_i − K_j = α(I − e^{−tL}) L⁺(S_i − S_j)
                        注意**基础项在差里消掉了**，所以这一层的差别完全由意图决定。
    第 1 层  软读数      只在场自己的均值以上保留原值，其余置零
    第 2 层  硬读数      只看集合，丢掉全部幅度（Jaccard 比较）

测量方法：**意图恢复率**。给每个 i 找它在某一层的最近邻，
看这个最近邻是否等于它在意图空间里的最近邻。三层用同一把尺子（意图空间）。

    · 若第 0 层高、第 2 层接近随机  => 失败在**读数**，控制本身没坏
    · 若第 0 层就不行              => 失败在**控制**，C2 真挂
    · 若第 1 层好、第 2 层不行      => 损失在**集合比较**，不在选择

零线：1/12 ≈ 0.083。另跑一组同范数的随机零和意图，给出"随机也能对"的底。

⚠️ 本模块**不引入新的读数**（不新增第四层），也不改判据。它只测量已有链条的损失。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402

N_INTENTS = 12


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def random_zero_sum(n: int, count: int, target_norm: float, seed: int = 20260930):
    out = []
    st = seed
    for _ in range(count):
        v = []
        while len(v) < n:
            st = _xorshift(st)
            v.append((st % 20000) / 10000.0 - 1.0)
        m = sum(v) / n
        v = [x - m for x in v]
        cur = (sum(x * x for x in v)) ** 0.5
        if cur > 0 and target_norm > 0:
            v = [x * target_norm / cur for x in v]
        out.append(v)
    return out


def nn(dist: list[list[float]], i: int) -> int:
    js = sorted((j for j in range(len(dist)) if j != i), key=lambda j: (dist[i][j], j))
    return js[0] if js else -1


def recovery(dist: list[list[float]], ref: list[list[float]]) -> float:
    k = len(dist)
    if k == 0:
        return 0.0
    hit = sum(1 for i in range(k) if nn(dist, i) == nn(ref, i))
    return hit / k


def dmats(vectors: list[list[float]], metric) -> list[list[float]]:
    k = len(vectors)
    return [[metric(vectors[i], vectors[j]) for j in range(k)] for i in range(k)]


def spread(mat: list[list[float]]) -> dict:
    vals = [mat[i][j] for i in range(len(mat)) for j in range(i + 1, len(mat))]
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "sd": 0.0}
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return {"min": min(vals), "max": max(vals), "mean": m, "sd": var ** 0.5}


def main() -> int:
    raw, _m = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    deg = G.degrees(adj)
    order_deg = sorted(node_ids, key=lambda v: (-deg[v], v))
    entries = [order_deg[0], order_deg[-1], order_deg[len(order_deg) // 2]]

    nontrivial = [k for k in range(n) if vals[k] > S.ZERO_TOL]
    intents = []
    for k in nontrivial[:N_INTENTS]:
        u = [vecs[i][k] for i in range(n)]
        intents.append([x * vals[k] for x in u])   # 等射程：|L⁺S| = 1

    eqs = [F.pinv_apply(vals, vecs, s) for s in intents]
    ref = dmats(eqs, F.l2)                          # 意图空间距离，三层共用的尺子

    # 随机零和意图：同范数
    s_norm = (sum(x * x for x in intents[0])) ** 0.5
    rnd_s = random_zero_sum(n, N_INTENTS, s_norm)
    rnd_eqs = [F.pinv_apply(vals, vecs, s) for s in rnd_s]
    rnd_ref = dmats(rnd_eqs, F.l2)

    L: list[str] = []
    L.append("# Phase 2 诊断 · 失败在控制层还是读数层\n")
    L.append("链条：`意图 --线性ODE--> 场 --选择--> 软读数 --丢幅度--> 硬读数(集合)`\n")
    L.append("三层用同一把尺子：意图空间的平衡场距离 `|L⁺Sᵢ − L⁺Sⱼ|`。")
    L.append(f"意图数 {N_INTENTS}，零线 1/{N_INTENTS} ≈ {1/N_INTENTS:.3f}。\n")
    L.append("| 入口 | α | t | 第0层 场 | 第1层 软读数 | 第2层 硬读数 | 场sd | 软sd | 硬sd |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    rows = []
    for e in entries:
        for alpha in (1.0, 2.0):
            for t in (0.5, 2.0, 20.0):
                fields = [F.trajectory(vals, vecs, didx[e], s, alpha, t) for s in intents]
                sets = [F.readout(f, node_ids) for f in fields]
                soft = []
                for i, f in enumerate(fields):
                    mask = sets[i]
                    soft.append([f[j] if node_ids[j] in mask else 0.0 for j in range(n)])

                fd = dmats(fields, F.l2)
                sd = dmats(soft, F.l2)
                hd = dmats(sets, F.jaccard)

                a0, a1, a2 = recovery(fd, ref), recovery(sd, ref), recovery(hd, ref)
                L.append(f"| {e} | {alpha} | {t} | **{a0:.3f}** | **{a1:.3f}** | **{a2:.3f}** "
                         f"| {spread(fd)['sd']:.4f} | {spread(sd)['sd']:.4f} | {spread(hd)['sd']:.4f} |")
                rows.append((e, alpha, t, a0, a1, a2, spread(fd), spread(sd), spread(hd)))

    # ── 随机零和对照 ──────────────────────────────────────────────────
    rnd_fields = [F.trajectory(vals, vecs, didx[entries[0]], s, 1.0, 2.0) for s in rnd_s]
    rnd_sets = [F.readout(f, node_ids) for f in rnd_fields]
    rnd_soft = []
    for i, f in enumerate(rnd_fields):
        mask = rnd_sets[i]
        rnd_soft.append([f[j] if node_ids[j] in mask else 0.0 for j in range(n)])
    r0 = recovery(dmats(rnd_fields, F.l2), rnd_ref)
    r1 = recovery(dmats(rnd_soft, F.l2), rnd_ref)
    r2 = recovery(dmats(rnd_sets, F.jaccard), rnd_ref)

    L.append("\n## 随机零和对照（同范数的随机意图）\n")
    L.append(f"- 第 0 层 场 {r0:.3f} · 第 1 层 软读数 {r1:.3f} · 第 2 层 硬读数 {r2:.3f}")
    L.append(f"- 硬读数距离 sd = {spread(dmats(rnd_sets, F.jaccard))['sd']:.4f}")
    L.append("\n（随机意图的场本来就没有结构，所以它的恢复率是「随机也能对」的噪声底。）")
    L.append("三层都要跟这个底比，不是跟 0 比。）\n")

    paths.report("phase2b_readout.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 三层意图恢复率（尺子 = 意图空间距离，零线 1/12=0.083）──")
    for e, alpha, t, a0, a1, a2, sf, ss, sh in rows:
        print(f"  {e:<10s} α={alpha:<4} t={t:<5} 场 {a0:.3f} | 软 {a1:.3f} | 硬 {a2:.3f}"
              f"   sd {sf['sd']:.4f}/{ss['sd']:.4f}/{sh['sd']:.4f}")
    print("── 随机零和对照 ──")
    print(f"  场 {r0:.3f} | 软 {r1:.3f} | 硬 {r2:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
