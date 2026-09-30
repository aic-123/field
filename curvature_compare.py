"""Phase 1 补做：Ollivier-Ricci 与 Forman 的正面对比。

设计稿 §1.1 原本指定的是 Ollivier-Ricci。Phase 1 时用 Forman 顶替，
结果被项分解证伪（度数项方差占比 1.054）。本模块把 OR 真做出来，
回答两个问题，**两个都不预设答案**：

    Q1  OR 是不是也被度数吞掉？（Forman 的 R² = 0.684）
    Q2  OR 能不能指出"结构薄"的地方？符号对不对？

Q2 要分组看，因为本图 16 条桥里大部分是**通向单点叶子的悬边**——
悬边"薄"但没有"两边相摩"，和设计稿说的"分歧焦点"不是一回事：

    a 悬边        一端是单点分量（叶子）
    b 界面桥      两端分量都 ≥ 2  —— 这才是"两块结构相摩"的地方
    c 核内边      非桥

⚠️ 本模块不判定，只报数。判定写在 verdict 文件里。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G            # noqa: E402
import curvature as C        # noqa: E402
import connectivity as CX    # noqa: E402
import ollivier as O         # noqa: E402

# 符号判定用的数值零容差，取自 ollivier（同一份定义，不另立一个数）。
TOL = O.TOL

GROUP_PENDANT = "a 悬边（一端是单点分量）"
GROUP_INTERFACE = "b 界面桥（两端分量都 >=2）"
GROUP_CORE = "c 核内边（非桥）"


def r_squared(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return (sxy * sxy) / (sxx * syy) if sxx and syy else 0.0


def describe(vals: list[float]) -> dict:
    v = sorted(vals)
    n = len(v)
    return {"n": n, "mean": sum(v) / n, "median": v[n // 2],
            "min": v[0], "max": v[-1],
            "pos": sum(1 for x in v if x > TOL),
            "neg": sum(1 for x in v if x < -TOL)}


def main() -> int:
    raw, _ = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)

    orows = {r["edge"]: r for r in O.curvature(adj)}
    frows = {r["edge"]: r for r in C.forman(adj, node_ids)}
    br = CX.bridges(adj)
    comps = CX.edge_connected_components(adj, br)
    size = {}
    for c in comps:
        for v in c:
            size[v] = len(c)

    def group(e):
        x, y = e
        if e not in br:
            return GROUP_CORE
        if size[x] == 1 or size[y] == 1:
            return GROUP_PENDANT
        return GROUP_INTERFACE

    edges = sorted(orows)
    deg = [orows[e]["dx"] + orows[e]["dy"] for e in edges]
    kappa = [orows[e]["kappa"] for e in edges]
    forman = [frows[e]["F"] for e in edges]

    groups: dict[str, list[tuple]] = {}
    for e in edges:
        groups.setdefault(group(e), []).append(e)

    L: list[str] = []
    L.append("# Ollivier-Ricci vs Forman\n")
    L.append("## 自检（三张已知答案的小图）\n")
    for name, ok, why in O.self_test():
        L.append(f"- {'过' if ok else '**挂**'}  {name}：{why}")
    L.append("")

    L.append("## Q1：度数代理程度\n")
    L.append(f"- Ollivier-Ricci 对 (d(x)+d(y)) 的 R² = **{r_squared(deg, kappa):.4f}**")
    L.append(f"- Forman 对 (d(x)+d(y)) 的 R² = **{r_squared(deg, forman):.4f}**")
    L.append("- （Forman 的项分解已知度数项方差占比 1.054，与此处 R² 方向一致）\n")

    L.append("## Q2：符号分布与分组\n")
    d0 = describe(kappa)
    L.append(f"- OR 全体：正 {d0['pos']} / 负 {d0['neg']} / 零 {d0['n']-d0['pos']-d0['neg']}，"
             f"范围 {d0['min']:+.4f} .. {d0['max']:+.4f}，"
             f"不同取值 {len(set(round(x,6) for x in kappa))}")
    df = describe(forman)
    L.append(f"- Forman 全体：正 {df['pos']} / 负 {df['neg']} / 零 "
             f"{df['n']-df['pos']-df['neg']}\n")

    L.append("| 分组 | n | OR 均值 | OR 中位 | OR 范围 | Forman 均值 |")
    L.append("|---|---|---|---|---|---|")
    for g in (GROUP_PENDANT, GROUP_INTERFACE, GROUP_CORE):
        if g not in groups:
            continue
        es = groups[g]
        kv = [orows[e]["kappa"] for e in es]
        fv = [frows[e]["F"] for e in es]
        s = describe(kv)
        L.append(f"| {g} | {s['n']} | {s['mean']:+.4f} | {s['median']:+.4f} | "
                 f"{s['min']:+.3f} .. {s['max']:+.3f} | {sum(fv)/len(fv):+.3f} |")
    L.append("")

    L.append("### 界面桥逐条（「两块结构相摩」的地方）\n")
    L.append("| 边 | d(x) | d(y) | OR κ | Forman F | 两端分量大小 |")
    L.append("|---|---|---|---|---|---|")
    for e in groups.get(GROUP_INTERFACE, []):
        x, y = e
        L.append(f"| {x} — {y} | {orows[e]['dx']} | {orows[e]['dy']} | "
                 f"{orows[e]['kappa']:+.4f} | {frows[e]['F']:+d} | {size[x]} / {size[y]} |")
    L.append("")

    L.append("## 没做的事\n")
    L.append("- 没有把 OR 用到 Phase 4 的 ρ 里。Phase 4 已按预先登记的 R1（共振）判决，")
    L.append("  事后把 OR 塞进去就是改判据。OR 的用途只能登记到 Phase 7。")
    L.append("- 没有扫 α（惰性参数）。本模块固定 α = 1/2。")
    paths.report("phase1b_curvature_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 自检 ──")
    for name, ok, why in O.self_test():
        print(f"  {'过' if ok else '挂'}  {name}：{why}")
    print("── Q1 度数代理 ──")
    print(f"  OR     R² = {r_squared(deg, kappa):.4f}")
    print(f"  Forman R² = {r_squared(deg, forman):.4f}")
    print("── Q2 分组 ──")
    for g in (GROUP_PENDANT, GROUP_INTERFACE, GROUP_CORE):
        if g not in groups:
            continue
        es = groups[g]
        kv = [orows[e]["kappa"] for e in es]
        s = describe(kv)
        print(f"  {g:<28s} n={s['n']:2d}  OR 均值 {s['mean']:+.4f}  中位 {s['median']:+.4f}")
    print("── 界面桥 ──")
    for e in groups.get(GROUP_INTERFACE, []):
        x, y = e
        print(f"  {x} — {y}  κ {orows[e]['kappa']:+.4f}  F {frows[e]['F']:+d}  "
              f"分量 {size[x]}/{size[y]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
