"""Phase 7c：有向谱与有向 Ollivier-Ricci（方法推荐 ⑧）。

---
先说清一件事：方向语义在上游文档里是矛盾的
------------------------------------------

⚠️ **先分清这些文件名各自住在哪个仓库**（这一节原来写错过，见下）：

    SPEC.md               上游 `aic-123/Scaffold` 的文件，**不在 rl-scaffold 里**
    index/by-situation.md rl-scaffold 的生成物

    SPEC.md 的散文规则    「A 出现在 B 的 relations 里，表示「A 指向 B」」
    SPEC.md 自己的方向表  「**论据 → 立场**、**立场 → 议题**……反向关系写在正文里」

而实测数据：`arg-0001` 的 `relations` 列的是 `stance-0001`。
两者只能对一个：

    SPEC 散文规则那条 ⟹ stance-0001 → arg-0001
    SPEC 方向表那条   ⟹ arg-0001 → stance-0001   ← 与 find_path 的行走方向一致

**冲突在上游 `SPEC.md` 自己内部**（散文规则与它自己的方向表相反），
不是「SPEC.md 与 by-situation.md 互相矛盾」——原先那种写法把引用范围
伸到了本仓库实际读不到的地方。详见 `DECLARATION.md` §五·三。

**本模块采用方向表那一侧（实践约定）**，理由：`find_path` 是沿着 `relations` 正向走的，
而 `SPEC.md` 自己的方向表与它一致；`README.md:47` 也明说方向「查不了，也不打算查」。
所以这是一处**上游文档冲突**，登记在案，不擅自改数据。

---
量什么
------

    平稳分布 π         有向游走的平稳分布一般不是均匀的，它把被大量指向的位置抬高
    Chung 有向拉普拉斯   用 π 对称化，谱是实的，可与 Phase 1 的无向谱直接比
    有向 OR            惰性测度换到出邻居上，传输代价仍用无向距离

**判定标准：有向版与无向版的相关系数。** 若接近 1，说明方向什么都没加。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import ollivier as O       # noqa: E402
import directed as DR      # noqa: E402
import metrics as M        # noqa: E402

# 符号零容差（与 ollivier 同一份约定）与"加了多少"的判据线。
SIGN_TOL = O.TOL
R2_ADDS = 0.9          # 判据先写死：有向 vs 无向 OR 的 R² < 0.9 视为方向加了东西
MAX_LIST = 8           # 报告里每层最多列几个节点（显示用，不是判据）


def main() -> int:
    raw, _ = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj_u = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj_u, node_ids))
    n = len(node_ids)

    adj_d = DR.directed_adj(full["rel_pairs"], node_ids)
    pi, it, delta, cap = DR.stationary(adj_d, node_ids)
    outd = DR.out_degree(adj_d)
    dist = O.all_pairs_dist(adj_u)

    L: list[str] = []
    L.append("# Phase 7c · 有向谱与有向 Ollivier-Ricci\n")
    L.append("## 方向语义的登记（上游文档冲突）\n")
    L.append("- **上游 `SPEC.md` 自己内部矛盾**（散文规则 vs 它自己的方向表）：")
    L.append("  - 散文规则：A 出现在 B 的 relations 里 ⟹ A 指向 B")
    L.append("  - 方向表：论据 → 立场 → 议题（**与上一条推出相反方向**）")
    L.append("- `by-situation.md:298`（rl-scaffold 的生成物）与方向表一侧一致。")
    L.append("- ⚠️ `SPEC.md` 属于上游 `aic-123/Scaffold`，**不在 rl-scaffold 里**。")
    L.append("- 实测：`arg-0001` 的 relations 列出 `stance-0001`")
    L.append("- **本模块采用读法约定**（= `find_path` 的行走方向）：")
    L.append("  「X 的 relations 列出 Y ⟹ X → Y」")
    L.append("- `README.md:47` 明说方向「查不了，也不打算查」，所以校验器拦不住这个冲突\n")

    L.append("## 平稳分布 π\n")
    L.append(f"- 幂迭代 {it} 次，收敛量 {delta:.3e}，撞上限 = {cap}")
    L.append(f"- π 范围 {min(pi):.6f} .. {max(pi):.6f}，"
             f"均匀分布是 {1.0/n:.6f}")
    spread = (max(pi) / min(pi)) if min(pi) > 0 else float("inf")
    L.append(f"- 最大/最小 = **{spread:.1f} 倍** —— 偏离均匀的程度")
    L.append(f"- 出度为 0 的节点数：{sum(1 for v in node_ids if outd[v] == 0)}/{n}")
    L.append("- π 最高的 5 个节点："
             + "、".join(f"{node_ids[i]}({pi[i]:.4f})"
                         for i in sorted(range(n), key=lambda i: -pi[i])[:5]) + "\n")

    L.append("## Chung 有向拉普拉斯：**前提不成立，不能用**\n")
    L.append("我第一次直接算，得到 λ₁ = −1.585 —— 而对称化后的拉普拉斯不该有负特征值。")
    L.append("查过之后原因清楚了：那个构造的定理前提是**有向图强连通**。\n")
    app = DR.laplacian_applicable(adj_d, node_ids)
    L.append(f"- 强连通分量数 = **{app['n_components']}**，"
             f"最大分量 {app['largest']} 个节点（共 {n} 个）")
    L.append(f"- 强连通 = **{app['strongly_connected']}**")
    L.append(f"- 结论：{app['reason']}")
    L.append("")
    L.append("**这不是意外，是设计：** `R3` 禁止成环，")
    L.append("`by-situation.md:298` 也明说「反向关系写在正文里，不写进 relations")
    L.append("（否则构成环，被 R3 判错）」。所以 `relations` **按设计就是一张 DAG**。")
    L.append("")
    L.append("把那个对称化对象照样算出来、看它坏成什么样（作为证据）：\n")
    Ld = DR.chung_laplacian(adj_d, node_ids, pi)
    vd, _vd = S.jacobi(Ld)
    neg = [x for x in vd if x < -SIGN_TOL]
    L.append(f"- λ 范围 {vd[0]:+.6f} .. {vd[-1]:+.6f}，**负特征值 {len(neg)} 个**，"
             f"最小 {vd[0]:+.6f}")
    L.append(f"- 所以它**不是拉普拉斯**，它的谱不能读成连通性谱。")
    L.append(f"- 无向谱（Phase 1）作对照：λ₁={vals[1]:.6f}，λ_max={vals[-1]:.6f}\n")

    L.append("## 换成 DAG 上对的工具\n")
    depth = DR.topological_depth(adj_d, node_ids)
    reach = DR.reach_count(adj_d, node_ids)
    L.append(f"- **拓扑层级**（到该点的最长路径）：可算 = {depth['ok']}")
    if depth["ok"]:
        byd: dict[int, list[str]] = {}
        for v, d in depth["depth"].items():
            byd.setdefault(d, []).append(v)
        for d in sorted(byd):
            L.append(f"    - 第 {d} 层（{len(byd[d])} 个）："
                     f"{'、'.join(sorted(byd[d])[:MAX_LIST])}"
                     f"{'…' if len(byd[d]) > MAX_LIST else ''}")
    L.append(f"- **可达数**（向下能到达几个节点）："
             f"范围 {min(reach.values())} .. {max(reach.values())}")
    top_reach = sorted(node_ids, key=lambda v: (-reach[v], v))[:5]
    L.append(f"    - 最高的 5 个："
             + "、".join(f"{v}({reach[v]})" for v in top_reach))
    L.append("- ⚠️ 可达数是**结构量**（完全由图决定），不是热度量，")
    L.append("  所以它不违反 `§C7.1 ①`。这一点必须写清楚，否则容易被误当成「被引用次数」。\n")

    L.append("## 有向 OR vs 无向 OR\n")
    rows_u = {r["edge"]: r for r in O.curvature(adj_u, dist)}
    rows_d = {r["edge"]: r for r in DR.directed_curvature(adj_d, dist)}
    common = sorted(set(rows_u) & set(rows_d))
    ku = [rows_u[e]["kappa"] for e in common]
    kd = [rows_d[e]["kappa"] for e in common]
    r2 = M.pearson_sq(ku, kd)
    L.append(f"- 可比边数 {len(common)}（有向图只含有向边，比无向少）")
    L.append(f"- 两者 κ 的 **R² = {r2:.4f}**"
             f"（接近 1 说明方向没加东西；接近 0 说明它是另一个量）")
    L.append(f"- 无向：正 {sum(1 for x in ku if x>SIGN_TOL)} / 负 "
             f"{sum(1 for x in ku if x<-SIGN_TOL)}")
    L.append(f"- 有向：正 {sum(1 for x in kd if x>SIGN_TOL)} / 负 "
             f"{sum(1 for x in kd if x<-SIGN_TOL)}")
    if common:
        mu = min(common, key=lambda e: rows_u[e]["kappa"])
        md = min(common, key=lambda e: rows_d[e]["kappa"])
        L.append(f"- 无向最负的边：`{mu[0]} — {mu[1]}` κ={rows_u[mu]['kappa']:+.4f}")
        L.append(f"- 有向最负的边：`{md[0]} — {md[1]}` κ={rows_d[md]['kappa']:+.4f}")
        same = mu == md
        L.append(f"- **是否同一条边：{same}**")
    L.append("")
    L.append("**OR 不需要强连通**，所以它在 DAG 上照样有定义 —— ")
    L.append("方向这一条路里，坏掉的只有拉普拉斯那一个工具，不是整条路。\n")
    L.append("## 判定\n")
    L.append(f"- Chung 有向拉普拉斯：**不适用**（{app['n_components']} 个强连通分量，"
             f"{len(neg)} 个负特征值）")
    L.append(f"- 方向加了多少（有向 OR vs 无向 OR）：R² = {r2:.4f}")
    adds = r2 is not None and r2 < R2_ADDS
    L.append(f"- **{'加了一些' if adds else '几乎没加'}**"
             f"（判据先写死：R² < 0.9 视为加了东西）")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 没有把有向性并进 Phase 1–6 的任何判据。变量一次只动一个。")
    L.append("- 没有分解「方向冲突」对它有多大影响（那需要知道正确方向，")
    L.append("  而上游 `SPEC.md` 的散文规则与它自己的方向表互相矛盾）。")
    paths.write("phase7c_directed.md", "\n".join(L) + "\n")

    print("── 有向谱与有向 OR ──")
    print(f"  π 最大/最小 = {spread:.1f} 倍（均匀是 1.0）")
    print(f"  强连通分量数 = {app['n_components']}，最大 {app['largest']} 个节点"
          f"  → 强连通 = {app['strongly_connected']}")
    print(f"  → **Chung 有向拉普拉斯不适用**：{len(neg)} 个负特征值，"
          f"最小 {vd[0]:+.4f}（它不是拉普拉斯）")
    if depth["ok"]:
        dd = depth["depth"]
        print(f"  改用 DAG 工具：拓扑层级范围 {min(dd.values())} .. {max(dd.values())}；"
              f"可达数范围 {min(reach.values())} .. {max(reach.values())}")
    print(f"  有向 vs 无向 OR 的 R² = {r2:.4f}  → {'加了一些' if adds else '几乎没加'}")
    if common:
        print(f"  最负的边：无向 {mu[0]}—{mu[1]}（{rows_u[mu]['kappa']:+.4f}）  "
              f"有向 {md[0]}—{md[1]}（{rows_d[md]['kappa']:+.4f}）  同一条={same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
