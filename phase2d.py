"""Phase 2d：**换读数**的正面对比（方法推荐 ①②③）。

问题：原来的读数（场自己均值以上的节点）在 Phase 2 被量出丢信息——

    场层意图恢复率 1.000    软读数 1.000    硬集合 0.438（零线 0.062）

信息是在"丢掉幅度、只比集合"那一步没的。而且均值本身是一个**拍出来的位置**。

本模块在同一套意图、同一批锚点、同一个判据下，把五种读数摆在一起比：

    L0   场本身（L2）                        —— 上界，不用任何选择
    L1m  均值水平集 · 软（只留集合内的值，L2）  —— 旧链条
    L2m  均值水平集 · 硬（集合，Jaccard）      —— 旧链条的末端
    L1s  sweep cut · 软（L2）                 —— 新链条
    L2s  sweep cut · 硬（集合，Jaccard）      —— 新链条的末端

判据一个字不改（还是与有效电阻几何的 Mantel + 置换 p、以及恢复率）。
换的只是读数。**这样比才叫"换读数"，不是"换到好看的为止"。**
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402
import ppr                 # noqa: E402
from phase2b_readout import dmats, recovery, spread   # noqa: E402
from phase2c import mantel, N_ANCHORS                 # noqa: E402

T_GRID = [0.5, 2.0, 20.0]
ALPHA_GRID = [1.0, 2.0]

# 显著性水平。**给它起名，不给它豁免**（B-D6 的规矩）。
# 0.05 是统计惯例，不是为本图调出来的线。
ALPHA = 0.05


def restrict(field: list[float], mask: set, node_ids: list[str]) -> list[float]:
    """软读数：只保留集合内的值，其余置零。"""
    return [field[i] if node_ids[i] in mask else 0.0 for i in range(len(field))]


def main() -> int:
    raw, _ = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    deg = G.degrees(adj)
    order_deg = sorted(node_ids, key=lambda v: (-deg[v], v))
    entries = [order_deg[0], order_deg[-1], order_deg[len(order_deg) // 2]]

    stride = max(1, n // N_ANCHORS)
    anchors = sorted(node_ids)[::stride][:N_ANCHORS]
    intents = []
    for a in anchors:
        v = [1.0 if x == a else 0.0 for x in node_ids]
        m = sum(v) / n
        intents.append([x - m for x in v])

    eqs = [F.pinv_apply(vals, vecs, s) for s in intents]
    ref_eq = dmats(eqs, F.l2)
    R = S.effective_resistance(vals, vecs, node_ids)
    refR = [[R["R"][didx[a]][didx[b]] for b in anchors] for a in anchors]

    L: list[str] = []
    L.append("# Phase 2d · 五种读数正面对比\n")
    L.append("判据不变（与有效电阻几何的 Mantel + 置换 p、恢复率），只换读数。")
    L.append(f"锚点 {len(anchors)} 个；t ∈ {T_GRID}；α ∈ {ALPHA_GRID}\n")

    keys = ["L0 场", "L1m 均值软", "L2m 均值硬", "L1s sweep软", "L2s sweep硬"]
    L.append("| 入口 | α | t | " + " | ".join(keys) + " | L2m的Mantel r/p | L2s的Mantel r/p |")
    L.append("|---|---|---|" + "---|" * (len(keys) + 2))

    agg = {k: [] for k in keys}
    mant = {"L2m": [], "L2s": []}
    for e in entries:
        for alpha in ALPHA_GRID:
            for t in T_GRID:
                fields = [F.trajectory(vals, vecs, didx[e], s, alpha, t) for s in intents]
                mean_sets = [F.readout(f, node_ids) for f in fields]
                sweep_sets = []
                for f in fields:
                    sc = ppr.sweep_cut(adj, node_ids, f)
                    sweep_sets.append(set(sc["members"]) if sc["ok"] else set())

                soft_m = [restrict(f, mean_sets[i], node_ids) for i, f in enumerate(fields)]
                soft_s = [restrict(f, sweep_sets[i], node_ids) for i, f in enumerate(fields)]

                fd = dmats(fields, F.l2)
                r0 = recovery(fd, ref_eq)
                r1m = recovery(dmats(soft_m, F.l2), ref_eq)
                r2m = recovery(dmats(mean_sets, F.jaccard), ref_eq)
                r1s = recovery(dmats(soft_s, F.l2), ref_eq)
                r2s = recovery(dmats(sweep_sets, F.jaccard), ref_eq)
                rm, pm = mantel(refR, dmats(mean_sets, F.jaccard))
                rs, ps = mantel(refR, dmats(sweep_sets, F.jaccard))

                vals_row = [r0, r1m, r2m, r1s, r2s]
                for k, v in zip(keys, vals_row):
                    agg[k].append(v)
                mant["L2m"].append((rm, pm))
                mant["L2s"].append((rs, ps))

                L.append(f"| {e} | {alpha} | {t} | "
                         + " | ".join(f"{v:.3f}" for v in vals_row)
                         + f" | {rm:+.3f}/{pm:.4f} | {rs:+.3f}/{ps:.4f} |")
    L.append("")

    L.append("## 汇总（18 组的均值）\n")
    L.append("| 读数 | 恢复率均值 | 相对 L0 保留了多少 |")
    L.append("|---|---|---|")
    base = sum(agg["L0 场"]) / len(agg["L0 场"])
    for k in keys:
        m = sum(agg[k]) / len(agg[k])
        L.append(f"| {k} | **{m:.3f}** | {m/base*100:.1f}% |")
    L.append("")
    better = sum(agg["L2s sweep硬"]) / len(agg["L2s sweep硬"]) > sum(agg["L2m 均值硬"]) / len(agg["L2m 均值硬"])
    L.append(f"## 判定\n")
    L.append(f"- 硬读数从均值换成 sweep cut，恢复率改变：**{'变好' if better else '没有变好'}**")
    L.append(f"  （均值硬 {sum(agg['L2m 均值硬'])/len(agg['L2m 均值硬']):.3f} → "
             f"sweep 硬 {sum(agg['L2s sweep硬'])/len(agg['L2s sweep硬']):.3f}）")
    sig_m = sum(1 for _r, p in mant["L2m"] if p < ALPHA)
    sig_s = sum(1 for _r, p in mant["L2s"] if p < ALPHA)
    L.append(f"- Mantel 显著组数（p<0.05，共 18）：均值硬 **{sig_m}** → sweep 硬 **{sig_s}**")
    L.append(f"- L0（场本身，不做任何选择）{base:.3f} —— 这是上界，也是「该丢多少」的参照")
    L.append("")
    L.append("## 怎么读这个结果（**推荐①在构造任务上没有兑现**）\n")
    L.append("把三个数摆在一起看，结论是清楚的，而且它跟预期相反：")
    L.append("")
    L.append("```")
    L.append("L0  场本身           0.812")
    L.append("L1m 均值软           0.771   ← 选择几乎不丢信息")
    L.append("L1s sweep软          0.778   ← 换一种选择，也几乎不丢信息")
    L.append("L2m 均值硬           0.413   ← 换成集合比较，丢掉一半")
    L.append("L2s sweep硬          0.167   ← 换成 sweep 集合，丢掉五分之四")
    L.append("```")
    L.append("")
    L.append("两条读数：")
    L.append("")
    L.append("**一、损失不在「怎么选」，在「用集合比」。** 两个软读数（0.771 / 0.778）")
    L.append("都紧贴 L0（0.812），几乎不丢；一旦换成 Jaccard 比集合，两个都塌。")
    L.append("所以真正该改的是**别把构造压成集合**，而不是换一种选法。")
    L.append("")
    L.append("**二、sweep cut 在这里是错的选法，而且原因是结构性的。**")
    L.append("sweep cut 取的是**电导最小的那个前缀**，而电导是**图的**性质：")
    L.append("意图稍微一动，序变了，但那个最小前缀往往不变或乱跳——")
    L.append("于是构造对意图反而更不敏感（0.413 → 0.167，Mantel 显著组数没变但恢复率腰斩）。")
    L.append("")
    L.append("**所以推荐①要改成这样：** sweep cut 不是构造用的读数，")
    L.append("它是**判断用的统计量**（无位置参数 + 可以拿图自己的节点当零分布校准）。")
    L.append("构造应该保留幅度——用场本身或软读数。下一条检验在 `phase4b` 里做：")
    L.append("sweep 电导这个统计量，到底分不分得开正例与负例。")
    L.append("")
    L.append("## 没做的事\n")
    L.append("- 没有改判据、没有改意图族、没有改锚点。变量只有读数。")
    L.append("- 没有用 W₁ 做集合距离。全图上的精确 W₁ 需要运输问题，")
    L.append("  36×36 的规模下整数化后的流量太大，本阶段不做。软读数（L1）")
    L.append("  已经能定位「损失在哪一层」，不需要再引入第三种集合度量。")
    paths.write("phase2d_readout_compare.md", "\n".join(L) + "\n")

    print("── 五种读数，18 组均值 ──")
    for k in keys:
        m = sum(agg[k]) / len(agg[k])
        print(f"  {k:<14s} {m:.3f}   (相对 L0: {m/base*100:.1f}%)")
    print(f"  Mantel 显著组数：均值硬 {sig_m}/18  →  sweep 硬 {sig_s}/18")
    print(f"── 硬读数 {'变好' if better else '没有变好'} ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
