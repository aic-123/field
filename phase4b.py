"""Phase 4b：把 sweep 电导当**判断用的统计量**来检验（方法推荐 ②③④）。

Phase 2d 的结论把推荐①重新定位了：

    sweep cut **不是构造用的读数**（它当构造时 0.167，比均值水平集的 0.413 还差）
    它是**判断用的统计量**：无位置参数，而且可以拿**这张图自己的节点**当零分布校准。

于是本模块回答一个具体的问题：**这个统计量分不分得开正例与负例，**
而且这一次用 AUC（优超概率）报，不是用 p 值——因为 9 vs 4 的精确置换
p 的分辨率下限是 1/715，那个尺度在这点样本上几乎是哑的。

---
四个候选统计量，同一批查询，一起比
----------------------------------

    1−u      sweep 电导的**校准读数**：φ* 在图自身零分布里的分位，取 top-k 里最好的那个
    ρ_均值   旧的共振（均值水平集，Jaccard，top-k 两两平均）
    ρ_sweep  共振换成 sweep 集合
    R0b      只用匹配器首名分（**强对照**，Phase 4 里它 p=0.0573 没分开）

校准怎么算（**没有拍出来的线**）：

    零分布 = 36 个节点各当一次种子，各得一个 φ*
    u_i    = 零分布里 φ* ≤ φ*_i 的比例        （越小 = 这个种子越不寻常）
    读数   = 1 − min_i u_i                    （越大 = 越有根据）
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import corpus as C           # noqa: E402

# 语料位置显式化（默认 ../rl-scaffold，可用 FIELD_CORPUS 覆盖）。见 corpus.py。
CORPUS, fp = C.attach()

import graph as G            # noqa: E402
import spectral as S         # noqa: E402
import field as F            # noqa: E402
import ppr                   # noqa: E402
import metrics as M          # noqa: E402
import conformal as CF       # noqa: E402
import probe_gaps as pg      # noqa: E402

K_MAIN = 3
C_GRID = [0.05, 0.15, 0.5]
T_ARC = 2.0     # Phase 4 原版共振用的热核弧长

# ρ 的定义域：少于两个构造就没有"一致性"可言（与 phase5/6 同一份定义）。
MIN_ENTRIES = 2
# 配对检验至少要几条配对样本才值得做。
MIN_PAIRED = 3
# 细看用的传送参数：取网格中间那一档，不是挑出来的最优值。
C_DETAIL = 0.15


def top_entries(q, match_by_id, k=K_MAIN):
    rows = fp.rank(q, match_by_id)
    return [(r[0], r[1]) for r in rows[:k]]


def mean_jaccard(sets):
    if len(sets) < MIN_ENTRIES:
        return None
    tot = n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            tot += 1.0 - F.jaccard(sets[i], sets[j])
            n += 1
    return tot / n


def main() -> int:
    raw, match_by_id = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    didx = {v: i for i, v in enumerate(node_ids)}

    cue = lambda w: [str(c).strip() for c in (raw[w].get("cues") or [])]   # noqa: E731
    strict = [q for w, q in pg.HELD_OUT if q not in cue(w)]
    negs = [q for cat, q in pg.CORPUS if "越界" in cat]

    L: list[str] = []
    L.append("# Phase 4b · sweep 电导作为判断统计量\n")
    L.append("四个候选一起比；统一用 **AUC（优超概率）** 报，另附精确置换 p 与其分辨率下限。")
    L.append("正例 = 9 句真改写；负例 = 4 条越界。\n")

    rows_all = []
    for c in C_GRID:
        null = ppr.node_seed_null(adj, node_ids, c)
        nsorted = null["sorted"]

        def reading_from_phi(phi):
            if phi is None:
                return None
            below = sum(1 for x in nsorted if x < phi)
            equal = sum(1 for x in nsorted if x == phi)
            u = (below + 0.5 * equal) / len(nsorted)
            return 1.0 - u

        def for_query(q):
            ents = top_entries(q, match_by_id)
            if not ents:
                # 零候选 = 最彻底的不共振，两条统计量都记 0（定义，不是补值）
                return None, None, None, 0.0, 0.0
            reads, msets, ssets, hsets = [], [], [], []
            for eid, mv in ents:
                mass, _it, _d, _cap = ppr.personalized_pr(adj, node_ids, eid, c)
                sc = ppr.sweep_cut(adj, node_ids, mass)
                reads.append(reading_from_phi(sc["conductance"]))
                ssets.append(set(sc["members"]) if sc["ok"] else set())
                msets.append(F.readout(mass, node_ids))
                # Phase 4 原版：热核中性扩散 + 均值水平集，构不出记 0
                kt = F.trajectory(vals, vecs, didx[eid], [0.0] * len(node_ids), 0.0, T_ARC)
                hsets.append(F.readout(kt, node_ids))
            best = max([r for r in reads if r is not None], default=None)
            r_heat = mean_jaccard(hsets)
            return (best, mean_jaccard(msets), mean_jaccard(ssets), ents[0][1],
                    0.0 if r_heat is None else r_heat)

        P = [for_query(q) for q in strict]
        N = [for_query(q) for q in negs]

        L.append(f"## c = {c}\n")
        L.append(f"- 零分布：φ* 范围 {null['min']:.4f} .. {null['max']:.4f}，"
                 f"中位 {nsorted[len(nsorted)//2]:.4f}\n")
        L.append("| 统计量 | AUC | 95% 区间 | 均值差 | 精确 p | p 下限 |")
        L.append("|---|---|---|---|---|---|")
        got = {}
        for idx, name in ((4, "R1 热核水平集（Phase 4 原版）"),
                          (2, "ρ sweep 集合"),
                          (1, "ρ 均值水平集（PPR）"),
                          (0, "1−u sweep 校准读数"),
                          (3, "R0b 首名分（强对照）")):
            p = [t[idx] for t in P if t[idx] is not None]
            n = [t[idx] for t in N if t[idx] is not None]
            s = M.summarize(p, n, name)
            got[name] = s
            L.append(f"| {name} | **{s['auc']:.3f}** | "
                     f"[{s['auc_lo']:.3f}, {s['auc_hi']:.3f}] | {s['diff']:+.3f} | "
                     f"{s['p_exact']:.4f} | {s['p_floor']:.4f} |")
            rows_all.append((c, name, s))
        L.append("")

        # ── 配对检验：同一批查询上比两个统计量（方法来自成熟的配对置换做法）──
        #
        # ⚠️ `for_query` 返回的元组布局是固定的，**用显式下标，不靠名字对齐**：
        #     0 = 1−u  1 = ρ均值  2 = ρsweep  3 = R0b  4 = R1热核
        # 上一版用名字表建索引而名字表按另一个顺序写，导致配对检验被喂了错误的列
        # （比出来的其实是 1−u 与热核），差值 −0.593 就是那样来的。
        COL_1U, COL_RMEAN, COL_RSWEEP, COL_R0B, COL_R1 = 0, 1, 2, 3, 4
        allq = list(zip(strict, P)) + list(zip(negs, N))
        labels = [1] * len(strict) + [0] * len(negs)
        L.append("### 配对检验：这些 AUC 的差别显不显著\n")
        L.append("两个统计量是在**同一批 13 条查询**上算的，所以必须配对比较。")
        L.append("本模块用**精确符号翻转检验**（枚举全部 2¹³ = 8192 种交换）：")
        L.append("对每条查询独立交换两个统计量的取值，看 AUC 差的分布。")
        L.append("这是配对 ROC 比较的成熟做法之一（另一个是 DeLong 检验，")
        L.append("但它基于渐近正态，在 n=13 上不可靠，有文献因此改用置换）。")
        L.append("同时对多个比较施加 Holm–Bonferroni 校正。\n")
        L.append("| 对比 | AUC 差 | 精确 p | Holm 校正后 | 显著 |")
        L.append("|---|---|---|---|---|")
        pairs = [("R1 热核（原版）", COL_R1, "R0b 首名分（强对照）", COL_R0B),
                 ("R1 热核（原版）", COL_R1, "ρ sweep 集合", COL_RSWEEP),
                 ("R1 热核（原版）", COL_R1, "ρ 均值水平集（PPR）", COL_RMEAN),
                 ("R1 热核（原版）", COL_R1, "1−u sweep 校准读数", COL_1U)]
        ptests = []
        for na, ia, nb, ib in pairs:
            keep = [(t[ia], t[ib], l) for (q, t), l in zip(allq, labels)
                    if t[ia] is not None and t[ib] is not None]
            if len(keep) < MIN_PAIRED:
                continue
            r = M.paired_auc_test([x[0] for x in keep], [x[1] for x in keep],
                                  [x[2] for x in keep])
            ptests.append((f"{na} vs {nb}", r))
        if ptests:
            holm = M.holm([r["p_exact"] for _n, r in ptests], 0.05)
            for (nm, r), (p0, padj, sig) in zip(ptests, holm):
                L.append(f"| {nm} | {r['diff']:+.3f} | {p0:.4f} | {padj:.4f} | "
                         f"{'是' if sig else '**否**'} |")
        L.append("")

        if c == C_DETAIL:
            # 针对性比较：分数分不开的那 6 句
            L.append("### 分数失效的那 6 句（c = 0.15）\n")
            L.append("| 语句 | 组 | 首名分 | R1热核 | ρsweep | 1−u |")
            L.append("|---|---|---|---|---|---|")
            hard = [(q, t) for q, t in zip(strict, P) if t[3] < fp.TAU]
            hard += [(q, t) for q, t in zip(negs, N)]
            for q, t in hard:
                g = lambda x: "—" if x is None else f"{x:.3f}"      # noqa: E731
                L.append(f"| {q} | {'负例' if q in negs else '正例'} | {t[3]:.3f} | "
                         f"**{g(t[4])}** | {g(t[2])} | {g(t[0])} |")
            L.append("")
            hp = [t[4] for q, t in hard if q not in negs and t[4] is not None]
            hn = [t[4] for q, t in hard if q in negs and t[4] is not None]
            if hp and hn:
                s = M.summarize(hp, hn, "这 6 句上的 R1 热核")
                L.append(f"- 这 6 句上 R1 热核：{M.fmt(s)}")
                L.append("")

            # conformal：拿正例当校准集
            pv = [t[0] for t in P if t[0] is not None]
            nv = [t[0] for t in N if t[0] is not None]
            L.append("### conformal 应用（连同它的缺陷一起报）\n")
            for alpha in (0.1, 0.2):
                ap = CF.apply_with_caveat([-x for x in pv], [-x for x in nv], [], alpha)
                cut, k, ok = CF.conformal_cut([-x for x in pv], alpha)
                n_pos = sum(1 for x in pv if -x <= cut) if ok else None
                n_neg = sum(1 for x in nv if -x <= cut) if ok else None
                L.append(f"- α={alpha}：校准集 = 9 条真改写（**与测试集同分布**这件事做不到），"
                         f"下标 {k}，门 {cut}，"
                         f"可给保证 = {ok}；正例落在门内 {n_pos}/{len(pv)}，"
                         f"负例落在门内 {n_neg}/{len(nv)}")
            L.append("")
            L.append("⚠️ **这不是一个合格的 conformal 应用。** 校准集与测试集必须同分布，")
            L.append("而这里校准集就是测试集自己（9 条里取出 9 条），覆盖率保证无从谈起。")
            L.append("conformal 的**机制**已经在 `conformal.py` 里用同分布数据验过")
            L.append("（名义 0.90 → 实测 0.9077），这里只是把它接到真实数据上，")
            L.append("并如实指出**样本量不够**才是真正的障碍。")
            L.append("")

    L.append("## 判定\n")
    for c in C_GRID:
        for name in ("R1 热核水平集（Phase 4 原版）", "ρ sweep 集合", "R0b 首名分（强对照）"):
            s = next(s for cc, nm, s in rows_all if cc == c and nm == name)
            L.append(f"- c={c} {name}：AUC {s['auc']:.3f}，p={s['p_exact']:.4f}")
    r_heat = next(s for cc, nm, s in rows_all if cc == C_DETAIL
                  and nm == "R1 热核水平集（Phase 4 原版）")
    r_sweep = next(s for cc, nm, s in rows_all if cc == C_DETAIL
                   and nm == "ρ sweep 集合")
    r_base = next(s for cc, nm, s in rows_all if cc == C_DETAIL
                  and nm == "R0b 首名分（强对照）")
    L.append("")
    L.append("### 这一格是本次最重要的读数\n")
    L.append("```")
    L.append("                          AUC      均值差     精确 p     p 下限")
    L.append(f"R1 热核（Phase 4 原版）    {r_heat['auc']:.3f}    {r_heat['diff']:+.3f}    "
             f"{r_heat['p_exact']:.4f}     {r_heat['p_floor']:.4f}")
    L.append(f"ρ sweep 集合              {r_sweep['auc']:.3f}    {r_sweep['diff']:+.3f}    "
             f"{r_sweep['p_exact']:.4f}     {r_sweep['p_floor']:.4f}")
    L.append(f"R0b 首名分（强对照）       {r_base['auc']:.3f}    {r_base['diff']:+.3f}    "
             f"{r_base['p_exact']:.4f}     {r_base['p_floor']:.4f}")
    L.append("```")
    L.append("")
    perfect = r_heat["auc"] >= 1.0
    L.append(f"- **R1 热核的 AUC = {r_heat['auc']:.3f}"
             f"{'（完全分开，36/36 对全部排对）' if perfect else ''}**，"
             f"强对照 R0b = {r_base['auc']:.3f}")
    L.append(f"- 均值差 {r_heat['diff']:+.3f} vs {r_base['diff']:+.3f}；"
             f"精确 p {r_heat['p_exact']:.4f} vs {r_base['p_exact']:.4f}"
             f"（下限 {r_heat['p_floor']:.4f}）")
    L.append("")
    L.append("### 三条结论（第三条是在配对检验之后才成立的）\n")
    L.append("**一、场统计量自身能分开正负例，这一条仍然成立。**")
    L.append(f"R1 热核单看：AUC {r_heat['auc']:.3f}（36/36 对全排对），"
             f"均值差 {r_heat['diff']:+.3f}，精确 p {r_heat['p_exact']:.4f}")
    L.append(f"（下限 {r_heat['p_floor']:.4f}）。**这是关于 R1 自己的结论，与对比无关。**")
    L.append("")
    L.append("**二、推荐①（sweep cut）在两个角色上都没兑现。**")
    L.append("```")
    L.append("作为构造读数（phase2d）  均值水平集 0.413  >  sweep 集合 0.167")
    L.append("作为判断统计量（本模块）  R1 热核 AUC 1.000  >  ρsweep 0.944  >  1−u 0.407")
    L.append("```")
    L.append("1−u 甚至是**反的**（0.407 < 0.5）：校准读数越低（种子越「不寻常」），")
    L.append("越可能是负例。原因是校准拿的是**图的节点**当零分布，")
    L.append("而正例与负例的种子在图上的位置并没有系统差异——")
    L.append("它量的是「这个节点在图里特不特别」，不是「这条查询有没有根据」。")
    L.append("")
    L.append("**三、「场比纯分数更好」这一条，配对检验之后不成立。**")
    L.append("")
    L.append("AUC 差只有 +0.056 —— 36 个配对里 R1 全对、R0b 错 2 对。")
    L.append("对同一批 13 条查询做**精确符号翻转检验**：")
    L.append("")
    L.append("```")
    L.append("R1 热核 vs R0b 首名分      AUC 差 +0.056   精确 p = 0.7852   Holm 后 1.0000")
    L.append("```")
    L.append("")
    L.append("**不显著。** 也就是说：这个 2 对的差距，在 13 条查询的规模上")
    L.append("与「纯属偶然」完全相容。")
    L.append("")
    L.append("所以 Phase 4 那句「**场做到了纯分数做不到的事**」要被撤掉，")
    L.append("换成：**两者都能分开，但没有任何证据表明场的那一个更好。**")
    L.append("")
    L.append("### 这一条是怎么被发现的（必须记下来）\n")
    L.append("我**先写了「这是两个不同量级的结果」**——依据是 AUC 1.000 对 0.944，")
    L.append("以及 p 值 0.0028 对 0.0573。看起来差别很大，其实没有检验这个差别本身。")
    L.append("补上配对检验（成熟的配对 ROC 比较做法）之后，它当场被推翻。")
    L.append("")
    L.append("**这是本项目第三次「先下结论、后被测掉」**，前两次是：")
    L.append("Phase 2 那个常数距离矩阵、Phase 4 那个被混淆放大的损伤实验。")
    L.append("三次的共同点：**两个数看起来不一样，就当成它们真的不一样。**")
    L.append("看两个统计量的距离，和检验这两个统计量是否有差异，是两件事。")
    L.append("")
    L.append("### 还有一处同类错误（同一轮内）\n")
    L.append("配对检验的第一版我按**名字表**建索引，而 `for_query` 返回的元组")
    L.append("是按另一个顺序排的，于是检验被喂了错误的列（比出来是 1−u 对热核，")
    L.append("差值 −0.593）。已改成**显式下标 + 在注释里写明元组布局**。")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 没有扫 sweep cut 的其它族（二部图上的 sweep、按度数归一的前缀等）。")
    L.append("- 没有把任何两个统计量合并成一个。上游 R-8：两个指标永不合并。")
    L.append("- 没有做 DeLong 检验做交叉验证。它在 n=13 上不可靠，")
    L.append("  而精确符号翻转已经把这个规模上的检验做尽了。")
    (HERE / "phase4b_sweep_statistic.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 五个统计量的 AUC（正例 9 / 负例 4）──")
    for c in C_GRID:
        print(f"  c={c}")
        for name in ("R1 热核水平集（Phase 4 原版）", "ρ sweep 集合",
                     "ρ 均值水平集（PPR）", "1−u sweep 校准读数", "R0b 首名分（强对照）"):
            s = next(s for cc, nm, s in rows_all if cc == c and nm == name)
            print(f"    {name:<26s} AUC {s['auc']:.3f} [{s['auc_lo']:.3f},{s['auc_hi']:.3f}]"
                  f"  均值差 {s['diff']:+.3f}  p={s['p_exact']:.4f}（下限 {s['p_floor']:.4f}）")
    print(f"── 判定：R1热核 AUC {r_heat['auc']:.3f} vs 强对照 {r_base['auc']:.3f}"
          f" → {'完全分开' if perfect else '未完全分开'} ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
