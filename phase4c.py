"""Phase 4c：谱模式敲除（方法推荐 ⑤）。

替代 Phase 4 那个失败的删节点损伤实验。

    Phase 4 的旧设计：删掉正确节点 → **入口集也跟着变了** → 混淆
    本模块的设计：  **图一个字节不动**，只把场的某一个模式敲掉

预测（先登记）：

    Fiedler 模（k = 1）是主切方向，敲掉它的**构造变化应当最大**。

对照：如果效应是随机的，35 个模式里每一个"夺魁"的机会均等，
     即 1/35 ≈ 0.029。观测比例要跟这个底比。

两个效应都报，但判定只看**构造变化**（非线性泛函）；
场的变化量只是投影系数大小，不含新信息。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402
import knockout as KO      # noqa: E402

T_ARC = 2.0
UNIFORM = 1.0 / 35.0        # 35 个非平凡模式的均等机会（n=36 时）


def main() -> int:
    raw, _ = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}

    L: list[str] = []
    L.append("# Phase 4c · 谱模式敲除（替代删节点的损伤实验）\n")
    L.append("**图一个字节不动**：只把场的某一个模式 `c_k·u_k` 减掉。")
    L.append("所以入口集、度分布、桥、连通性全部原样，混淆从设计上不存在。")
    L.append(f"弧长 t = {T_ARC}（中性扩散）。种子 = 全部 {n} 个节点，逐个数。\n")
    L.append("| 种子 | 效应最大的模式 | 构造变化 | 该模式系数 | 模式1的构造变化 | 模式1的位次 |")
    L.append("|---|---|---|---|---|---|")

    wins = {1: 0}
    mode1_position = []
    n_seed = 0
    rows_out = []
    for seed in node_ids:
        field_vec = F.heat(vals, vecs, [1.0 if i == didx[seed] else 0.0 for i in range(n)], T_ARC)
        rows = KO.sweep_all_modes(adj, node_ids, vecs, field_vec)
        if not rows:
            continue
        n_seed += 1
        best = KO.largest_effect_mode(rows)
        # 模式 1 的位次（按构造变化从大到小，1 起）
        usable = sorted([r for r in rows if r["jaccard_change"] is not None],
                        key=lambda r: (-r["jaccard_change"], r["mode"]))
        r1 = next((i for i, r in enumerate(usable, 1) if r["mode"] == 1), -1)
        mode1_position.append(r1)
        if best == 1:
            wins[1] += 1
        m1 = next((r for r in rows if r["mode"] == 1), None)
        bb = next((r for r in rows if r["mode"] == best), None)
        L.append(f"| {seed} | **{best}** | "
                 f"{'—' if bb is None else format(bb['jaccard_change'], '.3f')} | "
                 f"{'—' if bb is None else format(bb['coef'], '+.3f')} | "
                 f"{'—' if m1 is None else format(m1['jaccard_change'], '.3f')} | {r1} |")
        rows_out.append((seed, best, r1, m1))

    L.append("")
    L.append("## 读数\n")
    L.append(f"- 种子数 {n_seed}")
    L.append(f"- **效应最大的模式是 k=1 的次数：{wins[1]}/{n_seed} "
             f"= {wins[1]/n_seed:.3f}**")
    L.append(f"- 若效应随机，期望比例 = 1/35 ≈ {UNIFORM:.3f}")
    ratio = (wins[1] / n_seed) / UNIFORM if UNIFORM else None
    L.append(f"- 相对随机底：**{ratio:.1f} 倍**")
    med = sorted(mode1_position)[len(mode1_position) // 2]
    L.append(f"- 模式 1 的位次中位 = {med}（1 = 每次都是它夺魁）")
    L.append("")
    L.append("## 判定（判据先写死：k=1 夺魁的比例显著高于 1/35）\n")
    ok = wins[1] / n_seed > UNIFORM * 3
    L.append(f"- 预测「Fiedler 模的构造效应最大」：**{'成立' if ok else '不成立'}**"
             f"（比例 {wins[1]/n_seed:.3f} vs 底 {UNIFORM:.3f}）")
    L.append("")
    L.append("### 与旧损伤实验的对比\n")
    L.append("```")
    L.append("旧（删节点）   效应 0.08–0.12 → 修掉入口集混淆后掉到 0.004–0.031，基本消失")
    L.append("新（敲模式）   图不动，混淆不存在；效应量见上表")
    L.append("```")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 只敲单个模式，没做多模式联合敲除。")
    L.append("- 没有把敲除结果与「随机方向」作对照（敲一个非特征方向）。")
    L.append("  理由：特征基是场的自然分解，随机方向不对应任何结构含义；")
    L.append("  但这也意味着「效应大」只能解释为「这个模式在这里重要」，不能解释为因果。")
    paths.report("phase4c_knockout.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 谱模式敲除（36 个种子）──")
    print(f"  k=1 夺魁 {wins[1]}/{n_seed} = {wins[1]/n_seed:.3f}   随机底 {UNIFORM:.3f}"
          f"   倍数 {ratio:.1f}")
    print(f"  模式1位次中位 {med}")
    print(f"  预测{'成立' if ok else '不成立'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
