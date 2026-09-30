"""Phase 1 跑腿：建图 → 谱 → 有效电阻 → 曲率 → 空间，然后写报告。

三个候选量，对应设计稿 §3 Phase 1 的两个 kill test 加一个替补：

    kill test 1（谱）     谱是否退化？退化 => 只有一个方向可拧 => C2 无从谈起
    kill test 2（曲率）   曲率符号能不能区分"紧"与"薄"？
    替补（Fiedler 切）    曲率不行的话，场还有没有别的量能指出结构在哪里变薄？

⚠️ 本模块**不判定**。数字摆出来，判定写在报告正文，且必须写清
哪些数字支持、哪些数字反对。判定里不许出现拍出来的线。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G                    # noqa: E402
import spectral as S                 # noqa: E402
import curvature as C                # noqa: E402
import curvature_split as CS         # noqa: E402
import geometry as GEO               # noqa: E402


def line(s: str = "") -> None:
    print(s)


def main() -> int:
    raw, _match = G.load_nodes()
    full = G.build(raw)
    types = {nid: str(raw[nid].get("type") or "?") for nid in raw}

    node_ids = full["node_ids"]
    node_adj = G.subgraph(full["adj"], node_ids)

    st_full = G.stats(full["adj"])
    st_node = G.stats(node_adj)

    # ── 图 ────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("# Phase 1 · 图结构\n")
    L.append("## 整张图（节点 + cues）\n")
    L.append(f"- 节点 {st_full['n_nodes']}，cues {st_full['n_cues']}，顶点合计 {st_full['n_vertices']}")
    L.append(f"- 边 {st_full['n_edges']}，边质量合计（volume）{st_full['volume']:.1f}")
    L.append(f"- 连通分量 {st_full['n_components']}")
    L.append(f"- cue 的度取值集合：{st_full['cue_degrees']}")
    L.append(f"- **cue 全是叶子：{st_full['all_cues_are_leaves']}**")
    L.append(f"- 度分布：{st_full['degree_hist']}")
    L.append("\n## 只留节点（36 个）的诱导子图 —— 几何只长在这里\n")
    L.append(f"- 顶点 {st_node['n_nodes']}，边 {st_node['n_edges']}，volume {st_node['volume']:.1f}")
    L.append(f"- 连通分量 {st_node['n_components']}")
    L.append(f"- 度分布：{st_node['degree_hist']}")
    L.append("\n### 连通分量明细\n")
    for i, comp in enumerate(st_node["components"], 1):
        L.append(f"- 分量 {i}（{len(comp)} 个）：{'、'.join(comp)}")
    if full["dangling"]:
        L.append("\n### 悬挂 relations 引用（指向不存在的 id）\n")
        for a, b in full["dangling"]:
            L.append(f"- {a} → {b}")
    paths.report("phase1_graph.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── 谱 + 有效电阻 ─────────────────────────────────────────────────
    lap = S.laplacian(node_adj, node_ids)
    vals, vecs = S.jacobi(lap)
    desc = S.describe(vals)
    R = S.effective_resistance(vals, vecs, node_ids)
    rst = S.resistance_stats(R)

    Sl: list[str] = []
    Sl.append("# Phase 1 · 拉普拉斯谱与有效电阻\n")
    Sl.append("顶点序（读 R 矩阵与坐标时用）：\n")
    Sl.append("```")
    for i, v in enumerate(node_ids):
        Sl.append(f"  [{i:2d}] {v:<10s} {types.get(v,'?')}")
    Sl.append("```\n")
    Sl.append("## 谱（升序，组合拉普拉斯 L = D − A）\n")
    Sl.append("```")
    for i, x in enumerate(vals):
        Sl.append(f"  λ{i:<3d} = {x: .8f}")
    Sl.append("```\n")
    Sl.append("## 谱的读数\n")
    Sl.append(f"- 顶点数 n = {desc['n']}")
    Sl.append(f"- 零特征值个数（= 连通分量数）= {desc['zeros']}")
    Sl.append(f"- 最小正特征值 λ₁（Fiedler）= {desc['smallest_positive']}")
    Sl.append(f"- 次小正特征值 λ₂ = {desc['next_positive']}")
    if "ratio_l2_over_l1" in desc:
        Sl.append(f"- λ₂/λ₁ = {desc['ratio_l2_over_l1']:.4f}")
    if "ratio_l3_over_l1" in desc:
        Sl.append(f"- λ₃/λ₁ = {desc['ratio_l3_over_l1']:.4f}")
    Sl.append(f"- λ_max = {desc['lambda_max']:.6f}")
    Sl.append(f"- λ_max / λ₁ = {desc['lambda_max'] / desc['smallest_positive']:.2f}")
    Sl.append(f"- 头部 8 个：{['%.6f' % x for x in desc['head']]}")
    Sl.append(f"- 尾部 5 个：{['%.6f' % x for x in desc['tail']]}")
    Sl.append("\n## 有效电阻距离 R(i,j) 的分布\n")
    for k, v in rst.items():
        Sl.append(f"- {k}: {v}")
    paths.report("phase1_spectrum.md").write_text("\n".join(Sl) + "\n", encoding="utf-8")

    # ── 曲率 + 分解 ───────────────────────────────────────────────────
    rows = C.forman(node_adj, node_ids)
    sc = C.sign_counts(rows)
    split = CS.decompose(rows)

    Cl: list[str] = []
    Cl.append("# Phase 1 · 曲率（Forman）与它的项分解\n")
    Cl.append("## 逐边（按边 id 排，**不是名次**）\n")
    Cl.append("| 边 | 类型 | d(x) | d(y) | 度数项 | 三角形 | 三角形项 | F(e) |")
    Cl.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        x, y = r["edge"]
        Cl.append(f"| {x} — {y} | {types.get(x,'?')}/{types.get(y,'?')} "
                  f"| {r['dx']} | {r['dy']} | {r['dx'] + r['dy']} "
                  f"| {r['triangles']} | {3 * r['triangles']} | **{r['F']}** |")
    Cl.append("\n## 符号读数\n")
    for k, v in sc.items():
        Cl.append(f"- {k}: {v}")
    Cl.append("\n## 项分解：F = 4 − (d(x)+d(y)) + 3·t(e)\n")
    for k in ("degree_term", "triangle_term", "F"):
        d = split[k]
        Cl.append(f"- {k}: 范围 {d['min']:g} .. {d['max']:g}（跨度 {d['span']:g}），"
                  f"方差 {d['var']:.3f}")
    Cl.append(f"- Cov(度数项, 三角形项) = {split['cov_degree_triangle']:.3f}")
    Cl.append(f"- Var(F) = {split['var_F']:.3f}")
    Cl.append(f"- 度数项方差占比 = {split['share_degree']:.3f}")
    Cl.append(f"- 三角形项方差占比 = {split['share_triangle']:.3f}")
    Cl.append("\n## 没做的事\n")
    Cl.append("- **没有实现 Ollivier-Ricci 曲率**（在 `ollivier.py` 里补做了，")
    Cl.append("  对比见 `phase1b_curvature_compare.md`）。本阶段的 kill test 2 用的是 Forman。")
    paths.report("phase1_curvature.md").write_text("\n".join(Cl) + "\n", encoding="utf-8")

    # ── 空间 ──────────────────────────────────────────────────────────
    emb = GEO.embed_coords(vals, vecs, node_ids, dim=3)
    cut = GEO.fiedler_cut(vals, vecs, node_ids)
    ce = GEO.cut_edges(node_adj, cut["side_a"]) if cut["ok"] else []
    cond = GEO.conductance(node_adj, cut["side_a"]) if cut["ok"] else {}

    Gl: list[str] = []
    Gl.append("# Phase 1 · 空间（扩散坐标）与 Fiedler 切\n")
    Gl.append("## 扩散坐标 Ψ_k(i) = u_k(i)/√λ_k\n")
    Gl.append("`R(i,j) = ‖Ψ(i) − Ψ(j)‖²`，所以这就是有效电阻距离的欧氏实现。\n")
    Gl.append(f"- 用了几维：{emb['dims']}")
    Gl.append(f"- 用到的 λ：{['%.6f' % x for x in emb['lambda_used']]}\n")
    Gl.append("| 节点 | 类型 | Ψ₁ | Ψ₂ | Ψ₃ |")
    Gl.append("|---|---|---|---|---|")
    for v in node_ids:
        c = emb["coords"][v]
        Gl.append(f"| {v} | {types.get(v,'?')} | " + " | ".join(f"{x: .5f}" for x in c) + " |")
    if cut["ok"]:
        Gl.append(f"\n## Fiedler 切（λ₁ = {cut['lambda']:.6f}）\n")
        Gl.append(f"- A 侧（{len(cut['side_a'])} 个）：{'、'.join(cut['side_a'])}")
        Gl.append(f"- B 侧（{len(cut['side_b'])} 个）：{'、'.join(cut['side_b'])}")
        Gl.append(f"\n### 跨切的边（{len(ce)} 条）—— 结构最薄的地方\n")
        for x, y in ce:
            Gl.append(f"- {x}（{types.get(x,'?')}） — {y}（{types.get(y,'?')}）")
        Gl.append("\n### 切的读数\n")
        for k, v in cond.items():
            Gl.append(f"- {k}: {v if not isinstance(v, float) else round(v, 6)}")
    Gl.append("\n## 没做的事\n")
    Gl.append("- 不判定哪一侧更重要。两半等价，切本身才是信息。")
    Gl.append("- 这是**一个**切，不是全部切。更高的特征向量给出更细的切，本阶段先只看主切。")
    paths.report("phase1_geometry.md").write_text("\n".join(Gl) + "\n", encoding="utf-8")

    # ── 终端摘要 ──────────────────────────────────────────────────────
    line("── 图 ──")
    line(f"节点 {st_full['n_nodes']}  cues {st_full['n_cues']}  边 {st_full['n_edges']}")
    line(f"cue 的度集合 {st_full['cue_degrees']}  => cue 全是叶子: {st_full['all_cues_are_leaves']}")
    line(f"节点子图：顶点 {st_node['n_nodes']}  边 {st_node['n_edges']}  分量 {st_node['n_components']}")
    line(f"节点子图度分布 {st_node['degree_hist']}")
    line("── kill test 1：谱 ──")
    line(f"零特征值 {desc['zeros']} 个   λ₁={desc['smallest_positive']:.6f}   "
         f"λ₂={desc['next_positive']:.6f}   λ₂/λ₁={desc.get('ratio_l2_over_l1'):.4f}")
    line(f"λ_max={desc['lambda_max']:.6f}   λ_max/λ₁={desc['lambda_max']/desc['smallest_positive']:.2f}")
    line(f"头部 8：{['%.5f' % x for x in desc['head']]}")
    line(f"尾部 5：{['%.5f' % x for x in desc['tail']]}")
    line(f"有效维数 = {desc['effective_dim']:.3f} / {desc['n_positive']}")
    line("── kill test 2：曲率 ──")
    line(f"{sc}")
    line(f"度数项 跨度 {split['degree_term']['span']:g}  方差占比 {split['share_degree']:.3f}")
    line(f"三角形项 跨度 {split['triangle_term']['span']:g}  方差占比 {split['share_triangle']:.3f}")
    line(f"Cov(度,三角) = {split['cov_degree_triangle']:.3f}")
    line("── 替补：Fiedler 切 ──")
    line(f"A 侧 {len(cut['side_a'])} 个 / B 侧 {len(cut['side_b'])} 个，跨切边 {len(ce)} 条")
    line(f"切电导 {cond.get('conductance')}  跨切质量 {cond.get('cut_mass')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
