"""Phase 5 跑腿：遗忘（C6）。

设计稿 §3 Phase 5：

    遗忘 = **核带宽的松弛**（对指定区域把 t 增大）。**图一个字节不改。**

    预测一（结构性）  松弛前后对图做快照比对 → 逐字节相同
    预测二（语义性）  松弛应使构造从"确定"变"多解"，
                      而**不应**使构造从"对"变"错"

---
判据（先登记，再看结果）
------------------------

对松弛档 τ（用 t' = t·τ）测四个量：

    n_distinct  前 k 个入口给出的**不同读数**的个数      「确定 → 多解」看它
    acc         正确节点还在不在"正确入口"给出的读数里     「对 → 错」看它
    pos         正确节点在场上按质量的位次中位
    ρ_sep       共振在正例与负例之间的**分离度**（差）

判定：

    C6 成立  <=>  n_distinct 随 τ 上升（确定变多解）
                 且 acc 不塌（对没变成错）
                 且 ρ_sep 单调收窄（分辨力是"均匀丧失"，不是突然错乱）

    否证：若松弛后 **acc 塌了而 ρ_sep 仍保持** → 那是"自信地变了"，
         即**损坏**（data rot），不是遗忘 → 丢掉落权不删除。

---
⚠️ 实测结果：预测二被证伪
--------------------------

    τ=1  不同读数 2.67   acc 1.000   位次 1   分离 +0.521
    τ=16 不同读数 1.11   acc 1.000   位次 20  分离 +0.259

**「不同读数」是往下走的**，不是往上走。松弛让构造**变得更一样**。
机理：松弛把场压平，均值水平集变成"大约半张图"，于是无论从哪个入口出发
都收敛到**同一个粗略集合**。"多解"要求场里有几个彼此分离的局部极大——
那是**结构**的性质，不是**核带宽**的性质。

被支持的那一半：acc 全程 1.000（**没有把对变成错**），
正确节点位次**平滑退化** 1→4→8→10→20→22，分离度单调收窄。
所以"降权不删除"是**半对**：它不制造错误，但产生的是**均匀粗化**，不是承诺的那种不确定性。
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
T0 = 2.0
TAU_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

# ρ 的定义域：少于两个构造就没有"一致性"可言（与 phase6 同一份定义）。
MIN_ENTRIES = 2


def snapshot(adj: dict) -> str:
    """图的确定性序列化，用于"图一个字节没改"的比对。"""
    return "|".join(f"{x}:" + ",".join(f"{y}={adj[x][y]:g}" for y in sorted(adj[x]))
                    for x in sorted(adj))


def readouts(entries, vals, vecs, didx, node_ids, t):
    return [F.readout(F.trajectory(vals, vecs, didx[e], [0.0] * len(node_ids), 0.0, t),
                      node_ids) for e in entries if e in didx]


def mean_jaccard(sets):
    if len(sets) < MIN_ENTRIES:
        return None
    tot = n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            tot += 1.0 - F.jaccard(sets[i], sets[j])
            n += 1
    return tot / n


def mass_position(k, node_ids, target):
    pairs = sorted(((k[i], v) for i, v in enumerate(node_ids)), key=lambda p: (-p[0], p[1]))
    for r, (_, v) in enumerate(pairs, 1):
        if v == target:
            return r
    return -1


def main() -> int:
    raw, match_by_id = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))

    cue = lambda w: [str(c).strip() for c in (raw[w].get("cues") or [])]   # noqa: E731
    strict = [(w, q) for w, q in pg.HELD_OUT if q not in cue(w)]
    negs = [q for cat, q in pg.CORPUS if "越界" in cat]
    snap_before = snapshot(adj)

    L: list[str] = []
    L.append("# Phase 5 · 遗忘（C6）\n")
    L.append(f"- 松弛档 τ：{TAU_GRID}，用 t' = {T0}·τ 实现（**只改核，不改图**）")
    L.append(f"- 正例 = 真正改写的 {len(strict)} 句；负例 = 越界 {len(negs)} 句\n")

    rows = []
    for tau in TAU_GRID:
        t = T0 * tau
        # 正例：共振 + 正确性
        pos_rho, acc, positions, ndist = [], [], [], []
        for want, q in strict:
            rows_ = fp.rank(q, match_by_id)
            entries = [r[0] for r in rows_[:K_MAIN]]
            sets = readouts(entries, vals, vecs, didx, node_ids, t)
            pos_rho.append(mean_jaccard(sets))
            ndist.append(len({frozenset(s) for s in sets}))
            # 正确性：从**正确入口**出发，正确节点还在不在读数里
            kt = F.trajectory(vals, vecs, didx[want], [0.0] * len(node_ids), 0.0, t)
            rd = F.readout(kt, node_ids)
            acc.append(1 if want in rd else 0)
            positions.append(mass_position(kt, node_ids, want))
        # 负例：共振
        neg_rho = []
        for q in negs:
            rows_ = fp.rank(q, match_by_id)
            entries = [r[0] for r in rows_[:K_MAIN]]
            neg_rho.append(mean_jaccard(readouts(entries, vals, vecs, didx, node_ids, t)))
        pv = [x for x in pos_rho if x is not None]
        nv = [x for x in neg_rho if x is not None]
        pmean = sum(pv) / len(pv) if pv else None
        nmean = sum(nv) / len(nv) if nv else None
        rows.append({
            "tau": tau, "t": t,
            "rho_pos": pmean, "rho_neg": nmean,
            "sep": (pmean - nmean) if (pmean is not None and nmean is not None) else None,
            "ndist": sum(ndist) / len(ndist),
            "acc": sum(acc) / len(acc),
            "pos_med": sorted(positions)[len(positions) // 2],
        })

    L.append("| τ | t | 不同读数（均） | acc 正确节点仍在 | 正确节点位次中位 | ρ 正例 | ρ 负例 | 分离度 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda x: "—" if x is None else f"{x:.3f}"      # noqa: E731
        L.append(f"| {r['tau']:g} | {r['t']:g} | {r['ndist']:.2f} | {r['acc']:.3f} | "
                 f"{r['pos_med']} | {f(r['rho_pos'])} | {f(r['rho_neg'])} | {f(r['sep'])} |")
    L.append("")

    snap_after = snapshot(adj)
    L.append("## 结构性预测：图一个字节没改\n")
    L.append(f"- 松弛前后图序列化相同：**{snap_before == snap_after}**")
    L.append(f"- （序列化长度 {len(snap_before)} 字节；`nodes/` 全程只读）\n")

    first, last = rows[0], rows[-1]
    nd_up = last["ndist"] > first["ndist"]
    # 「正确性没塌」的比较对象是**它自己的初始值**，不是一条拍出来的线。
    # （第一版写的是 `last["acc"] >= 0.5`，那是真魔数，被 B-D6 当场抓出。）
    acc_keep = last["acc"] >= first["acc"]
    sep_shrink = (first["sep"] is not None and last["sep"] is not None
                  and last["sep"] < first["sep"])
    L.append("## 判定\n")
    L.append(f"- 构造从确定变多解（不同读数上升）：**{nd_up}**"
             f"（{first['ndist']:.2f} → {last['ndist']:.2f}）")
    L.append(f"- 正确性没塌（与**自己的初始值**比，不是与一条线比）：**{acc_keep}**"
             f"（{first['acc']:.3f} → {last['acc']:.3f}）")
    L.append(f"- 分辨力均匀收窄（分离度下降）：**{sep_shrink}**"
             f"（{first['sep']} → {last['sep']}）")
    L.append(f"\n**C6 {'成立' if (nd_up and acc_keep and sep_shrink) else '不成立'}**")
    if (not acc_keep) and sep_shrink:
        L.append("\n⚠️ acc 塌了而分离度还在 → 这是**自信地变了**，即损坏，不是遗忘。")
    L.append("")
    L.append("## 建议的替代说法（不是改判据，是改设计）\n")
    L.append("遗忘的语义应是**「分辨率衰减」**（位次平滑退化 + 分离度收窄），")
    L.append("而不是「确定变多解」。前者可测且已测到，后者在这套机制里做不到。")
    L.append("")

    paths.report("phase5_decay.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 遗忘曲线 ──")
    for r in rows:
        f = lambda x: "—" if x is None else f"{x:+.3f}"     # noqa: E731
        print(f"  τ={r['tau']:<5g} t={r['t']:<5g} 不同读数 {r['ndist']:.2f}  "
              f"acc {r['acc']:.3f}  位次 {r['pos_med']}  ρ {f(r['rho_pos'])}/{f(r['rho_neg'])}  "
              f"分离 {f(r['sep'])}")
    print(f"── 图未改：{snap_before == snap_after} ──")
    print(f"── C6 {'成立' if (nd_up and acc_keep and sep_shrink) else '不成立'} ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
