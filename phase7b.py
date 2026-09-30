"""Phase 7b：合成图上的界面桥机制验证（方法推荐 ⑥）。

---
要验的那一条，以及为什么它需要合成图
------------------------------------

Phase 1b 在真实图上量到：

    悬边（一端是叶子）        OR 均值 +0.1256    n = 15
    界面桥（两端都有结构）    OR        −0.5000    **n = 1**
    核内边                    OR 均值 −0.0966    n = 31

方向对得上：**最负的曲率正好落在唯一那条界面桥上**。
但 n = 1 是轶事，不是证据。

合成图能把 n 抬起来：k 个团连成环就有 k 条界面桥，
每个团再挂若干叶子就有大量悬边。于是"OR 能不能把界面与悬边分开"
可以在几十条边上验。

---
判据（先登记）
--------------

    **AUC(界面 vs 悬边)：随机取一条界面桥、一条悬边，
    前者 κ 更低的概率。** 预测接近 1。

    AUC = 0.5 表示两种桥在 OR 眼里没有区别 —— 那么"最负 = 界面"
    就只是"最负 = 桥"的一个别名，本模块如实记这个结果。

同时也报 OR 的度数 R²，看度数代理是不是真实图特有的。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import ollivier as O        # noqa: E402
import synthetic as SYN     # noqa: E402
import metrics as M         # noqa: E402

# 算度数 R² 至少要几条边（少于两条就无法定义相关）。
MIN_EDGES_FOR_R2 = 2
# 判断曲率符号的零容差：与 ollivier 用同一个（同一份数值约定）。
SIGN_TOL = O.TOL


def analyse(g: dict) -> dict:
    adj = g["adj"]
    rows = O.curvature(adj)
    kappa = {r["edge"]: r["kappa"] for r in rows}
    deg = {r["edge"]: r["dx"] + r["dy"] for r in rows}
    grp = SYN.edge_groups(g)

    def vals(key):
        return [kappa[e] for e in grp[key] if e in kappa]

    iface, pend, core, other = vals("interface"), vals("pendant"), vals("core"), vals("other_bridge")
    degs = [deg[e] for e in kappa]
    ks = [kappa[e] for e in kappa]
    return {
        "name": g["name"], "n_nodes": len(adj), "n_edges": len(kappa),
        "n_interface": len(iface), "n_pendant": len(pend),
        "n_other_bridge": len(other), "n_core": len(core),
        "mean_interface": (sum(iface) / len(iface)) if iface else None,
        "mean_pendant": (sum(pend) / len(pend)) if pend else None,
        "mean_other_bridge": (sum(other) / len(other)) if other else None,
        "mean_core": (sum(core) / len(core)) if core else None,
        "auc_interface_vs_pendant": M.auc([-x for x in iface], [-x for x in pend])
        if iface and pend else None,
        "auc_interface_vs_core": M.auc([-x for x in iface], [-x for x in core])
        if iface and core else None,
        "deg_r2": M.pearson_sq(degs, ks) if len(degs) > MIN_EDGES_FOR_R2 else None,
        "pos": sum(1 for x in ks if x > SIGN_TOL), "neg": sum(1 for x in ks if x < -SIGN_TOL),
    }


def main() -> int:
    graphs = [
        SYN.two_cliques(6, 6),
        SYN.clique_ring(5, 4),
        SYN.clique_ring_with_pendants(4, 4, 3),
        SYN.clique_ring_with_pendants(6, 3, 2),
        SYN.random_tree(24, seed=7),
    ]

    L: list[str] = []
    L.append("# Phase 7b · 合成图上的界面桥机制验证\n")
    L.append("**这是机制验证，不是评测。** 埋点位置事先已知，")
    L.append("问的是「OR 有没有那个性质」，不是「系统准不准」。\n")
    L.append("## 逐图\n")
    L.append("| 图 | 节点 | 边 | 界面桥 | 悬边 | 其它桥 | 核内边 | "
             "OR 界面均值 | OR 悬边均值 | OR 其它桥均值 | OR 核内均值 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")

    res = []
    for g in graphs:
        r = analyse(g)
        res.append(r)
        f = lambda x: "—" if x is None else f"{x:+.4f}"   # noqa: E731
        L.append(f"| {r['name']} | {r['n_nodes']} | {r['n_edges']} | **{r['n_interface']}** "
                 f"| {r['n_pendant']} | {r['n_other_bridge']} | {r['n_core']} "
                 f"| {f(r['mean_interface'])} | {f(r['mean_pendant'])} "
                 f"| {f(r['mean_other_bridge'])} | {f(r['mean_core'])} |")
    L.append("")

    L.append("## 核心判据：OR 分不分得开「界面桥」与「悬边」\n")
    L.append("AUC = 随机取一条界面桥、一条悬边，前者 κ 更低的概率。0.5 = 分不开。\n")
    L.append("| 图 | 界面 n | 悬边 n | AUC(界面 vs 悬边) | AUC(界面 vs 核内) | OR 的度数 R² |")
    L.append("|---|---|---|---|---|---|")
    for r in res:
        a1 = "—" if r["auc_interface_vs_pendant"] is None else f"{r['auc_interface_vs_pendant']:.3f}"
        a2 = "—" if r["auc_interface_vs_core"] is None else f"{r['auc_interface_vs_core']:.3f}"
        d2 = "—" if r["deg_r2"] is None else f"{r['deg_r2']:.3f}"
        L.append(f"| {r['name']} | {r['n_interface']} | {r['n_pendant']} | **{a1}** | {a2} | {d2} |")
    L.append("")

    withpair = [r for r in res if r["auc_interface_vs_pendant"] is not None]
    tot_if = sum(r["n_interface"] for r in withpair)
    allperfect = all(r["auc_interface_vs_pendant"] >= 1.0 for r in withpair)
    L.append("## 判定\n")
    L.append(f"- 参与判定的图 {len(withpair)} 张，**埋进去的界面桥合计 n = {tot_if}**"
             f"（真实图上只有 1 条）")
    if withpair:
        for r in withpair:
            L.append(f"    - {r['name']}：AUC = {r['auc_interface_vs_pendant']:.3f}"
                     f"（界面 {r['n_interface']} 条 / 悬边 {r['n_pendant']} 条）")
    L.append(f"- 是否每张图都完全分开（AUC = 1.000）：**{allperfect}**")
    L.append("")
    L.append("### 与真实图的对照\n")
    L.append("```")
    L.append("真实图      界面 1 条 κ=−0.5000   悬边 15 条 均值 +0.1256   （n=1，轶事）")
    L.append(f"合成图     界面 {tot_if} 条           见上表                    （n≥5，可判）")
    L.append("```")
    L.append("")
    L.append("### 树作为反面对照\n")
    tree = next((r for r in res if r["name"].startswith("随机树")), None)
    if tree:
        L.append(f"- 随机树 {tree['n_edges']} 条边**全是桥**，但没有界面。")
        L.append(f"- 它的 OR：正 {tree['pos']} / 负 {tree['neg']}，"
                 f"核内边均值 {tree['mean_core']}（这里的「核内边」其实是其它桥）")
        L.append("- **读法**：如果 OR 只是「见桥就给负」，树上的负值应当遍地；")
        L.append("  实测见上。这一格是「界面」与「桥」可不可分的直接检验。")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 合成图只用来做机制验证，**不用来报任何性能数字**。")
    L.append("- 没有调 α（惰性参数），固定 1/2。")
    paths.report("phase7b_synthetic_interfaces.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 合成图上的界面桥验证 ──")
    for r in res:
        a1 = "—" if r["auc_interface_vs_pendant"] is None else f"{r['auc_interface_vs_pendant']:.3f}"
        f = lambda x: "—" if x is None else f"{x:+.4f}"   # noqa: E731
        print(f"  {r['name']:<28s} 界面 {r['n_interface']:2d}  悬边 {r['n_pendant']:2d}"
              f"  OR界面 {f(r['mean_interface'])}  OR悬边 {f(r['mean_pendant'])}"
              f"  AUC {a1}")
    print(f"  埋进去的界面合计 n = {tot_if}；每张图都完全分开 = {allperfect}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
