"""Phase 3 跑腿：入口鲁棒性（C4）。

设计稿 §3 Phase 3 的两件事：

    A. 拿 17 条同义改写留出句当入口，看轨迹能不能到正确节点
    B. 把入口故意挪到相邻节点，看构造的定性内容是否保持

⚠️ 本模块**不碰匹配器**。设计稿写死了这一条的变量必须是"轨迹"，
否则轨迹与匹配器两个变量一起动，归因不清。

---
A 部分为什么必须先查一件事
--------------------------

上游 `GAPS.md:131-133` 的结论是：

    「检索的瓶颈**不在内容量，在匹配算法**——中文走的是字符二元组重合，
      需要字面共享；真人换一套词说同一件事，就落不下去。」

这个结论如果成立，那么 15/17 里那 2 条"没匹配上"应该是**排序错了**。
但如果它们只是**卡在 TAU 那条线下面**、排序其实是对的，那结论就得改：
瓶颈既不在内容量、也不在匹配算法，而在**那条线**。

这是可以查的，而且必须查——它决定 Phase 3 到底是"改匹配器"还是"换判据"。

---
B 部分的对照
------------

"轨迹能不能把人拉回正确区域"这句话，必须跟**随机入口的底**比，
不能跟 0 比：读数是"场自己均值以上"，大约占一半节点，
随机入口本来就有约一半概率命中。

所以按图距离分桶报，桶越远越接近随机底。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import corpus as C           # noqa: E402

# 语料位置显式化（默认 ../rl-scaffold，可用 FIELD_CORPUS 覆盖）。见 corpus.py。
CORPUS, fp = C.attach()

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402
import probe_gaps as pg    # noqa: E402

TARGETS = ["judge-0010", "judge-0011", "judge-0012", "judge-0013"]
T_GRID = [0.2, 1.0, 2.0, 5.0, 20.0]

# 「远桶」的下界。**给它起名，不给它豁免**（B-D6 的规矩）。
# 为什么是 4：图直径 8，全图 630 对的距离分布是
#   1:94  2:232  3:310  4:290  5:216  6:78  7:28  8:12
# 众数在 3–4，所以 d>=4 已经是"离得很远"，可以当随机底用。
# 它是**报告分桶**，不是判据线；换一个分桶不改变结论。
FAR_MIN = 4


def bfs_dist(adj: dict, src: str) -> dict:
    d = {src: 0}
    q = [src]
    while q:
        nxt = []
        for x in q:
            for y in adj[x]:
                if y not in d:
                    d[y] = d[x] + 1
                    nxt.append(y)
        q = nxt
    return d


def mass_position(k: list[float], order: list[str], target: str) -> int:
    """target 在场上按质量从大到小的**位次**（1 起）。并列按 id 破。

    这是给报告看的诊断量，不是产物里的排名轴：它衡量的是场本身，
    不是给来源/主张排序（上游 B2 禁的那件事）。
    """
    pairs = sorted(((k[i], v) for i, v in enumerate(order)), key=lambda p: (-p[0], p[1]))
    for r, (_, v) in enumerate(pairs, 1):
        if v == target:
            return r
    return -1


def main() -> int:
    raw, match_by_id = G.load_nodes()
    types = {k: str(raw[k].get("type") or "?") for k in raw}
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}
    zeros = [0.0] * n

    negatives = [q for cat, q in pg.CORPUS if "越界" in cat]

    L: list[str] = []
    L.append("# Phase 3 · 入口鲁棒性（C4）\n")

    # ══ A. 先查瓶颈到底在哪 ══════════════════════════════════════════
    L.append("## A · 瓶颈在排序还是在判据线\n")
    L.append(f"- 匹配器判据线 TAU = {fp.TAU}")
    L.append(f"- 留出集 {len(pg.HELD_OUT)} 句；越界负例 {len(negatives)} 句\n")
    L.append("⚠️ 表里多一列 `cue_dice`：它等于 1.0 就说明这句话**本身就是某个节点的 cue**，")
    L.append("那么它根本不是留出句，是训练句。这一列必须报到，不能只报总分。\n")
    L.append("| 语句 | 期望 | argmax | argmax 分 | argmax 对不对 | 期望节点位次 | cue_dice | 过 TAU |")
    L.append("|---|---|---|---|---|---|---|---|")

    held_rows = []
    ax_ok = 0
    for want, q in pg.HELD_OUT:
        rows = fp.rank(q, match_by_id)
        top_id, top_sc = (rows[0][0], rows[0][1]) if rows else ("（无重合）", 0.0)
        pos = next((i + 1 for i, r in enumerate(rows) if r[0] == want), -1)
        cd = max((r[2].get("cue_dice", 0.0) for r in rows), default=0.0)
        # 「是不是 cue 原文」用**精确成员判定**，不用 dice 的阈值。
        # 上游 B-D6 禁内联数字线，而这里本来就有精确答案。
        is_verbatim = q in [str(c).strip() for c in (raw[want].get("cues") or [])]
        ok = top_id == want
        ax_ok += 1 if ok else 0
        L.append(f"| {q} | {want} | {top_id} | {top_sc:.3f} | {'✓' if ok else '✗'} "
                 f"| {pos} | {cd:.3f} | {'是' if top_sc >= fp.TAU else '**否**'} |")
        held_rows.append((want, q, top_id, top_sc, ok, pos, cd, is_verbatim))

    neg_rows = []
    for q in negatives:
        rows = fp.rank(q, match_by_id)
        top_id, top_sc = (rows[0][0], rows[0][1]) if rows else ("（无重合）", 0.0)
        L.append(f"| {q} | （越界） | {top_id} | {top_sc:.3f} | — | — "
                 f"| {'**是**' if top_sc >= fp.TAU else '否'} |")
        neg_rows.append((q, top_id, top_sc))

    L.append(f"\n### A 的读数\n")
    L.append(f"- 留出集 argmax 正确：**{ax_ok}/{len(held_rows)}**")
    passed = [r for r in held_rows if r[3] >= fp.TAU]
    L.append(f"- 过 TAU 的留出句：**{len(passed)}/{len(held_rows)}**（这就是 15/17 的来源）")
    verbatim = [r for r in held_rows if r[7]]
    L.append(f"- **本身是 cue 原文的句子（不是留出句）：{len(verbatim)}/{len(held_rows)}**")
    if verbatim:
        for r in verbatim:
            L.append(f"    - {r[1]}  （期望 {r[0]}）")
    strict = [r for r in held_rows if not r[7]]
    L.append(f"- 真正是改写的句子：{len(strict)} 句，其中 argmax 正确 "
             f"**{sum(1 for r in strict if r[4])}/{len(strict)}**，"
             f"过 TAU **{sum(1 for r in strict if r[3] >= fp.TAU)}/{len(strict)}**")
    under = [r for r in held_rows if r[3] < fp.TAU]
    L.append(f"- 没过 TAU 但 argmax 正确的：**{len(under)}** 句")
    for want, q, top_id, sc, ok, pos, cd, _vb in under:
        L.append(f"    - {q}  → {top_id} 分 {sc:.3f}，位次 {pos}，cue_dice {cd:.3f}，"
                 f"{'argmax 正确' if ok else 'argmax 错'}")
    neg_max = [r[2] for r in neg_rows]
    L.append(f"- 越界负例的 argmax 分：{[f'{x:.2f}' for x in neg_max]}")
    L.append(f"- 越界负例过 TAU 的：**{sum(1 for x in neg_max if x >= fp.TAU)}/{len(neg_max)}**")
    if under and neg_max:
        lo = min(r[3] for r in under)
        hi = max(neg_max)
        L.append(f"\n**关键读数：没过线的留出句最低分 {lo:.3f}，越界负例最高分 {hi:.3f}。**")
        L.append(f"两者{'重叠' if hi >= lo else '不重叠'}——"
                 f"所以 TAU 这一条线{'无法' if hi >= lo else '可以'}把两者分开。")
    L.append("")

    # ══ B. 入口扰动容忍度 ═══════════════════════════════════════════
    L.append("## B · 入口扰动容忍度（按图距离分桶）\n")
    L.append("中性扩散（α=0，无意图），读数 = 场自己均值以上。")
    L.append("对照不是 0，是**随机入口的底**——最远那几个桶就是底。\n")

    curves = {}
    for t in T_GRID:
        bucket: dict[int, list] = {}
        for target in TARGETS:
            dist = bfs_dist(adj, target)
            for entry in node_ids:
                d = dist.get(entry, 99)
                k = F.trajectory(vals, vecs, didx[entry], zeros, 0.0, t)
                rd = F.readout(k, node_ids)
                r = mass_position(k, node_ids, target)
                bucket.setdefault(d, []).append((target in rd, r))
        curves[t] = bucket

    for t, bucket in curves.items():
        L.append(f"### t = {t}\n")
        L.append("| 图距离 d | 入口数 | 正确节点在读数里 | 正确节点位次中位 |")
        L.append("|---|---|---|---|")
        for d in sorted(bucket):
            rows = bucket[d]
            hit = sum(1 for h, _ in rows if h) / len(rows)
            rs = sorted(r for _, r in rows)
            med = rs[len(rs) // 2]
            L.append(f"| {d} | {len(rows)} | {hit:.3f} | {med} |")
        L.append("")

    # 汇总：d=0/1/2 与"远桶"（底）对比
    L.append("### B 的读数（t = 2）\n")
    b = curves[2.0]
    def rate(ds):
        rows = [x for d in ds for x in b.get(d, [])]
        if not rows:
            return None, None
        return sum(1 for h, _ in rows if h) / len(rows), sorted(r for _, r in rows)[len(rows) // 2]
    far = [d for d in b if d >= FAR_MIN]
    for label, ds in (("d=0（精确入口）", [0]), ("d=1", [1]), ("d=2", [2]),
                      ("d=3", [3]), (f"d>={FAR_MIN}（随机底，桶 {sorted(far)}）", far)):
        r, m = rate(ds)
        if r is not None:
            L.append(f"- {label}：命中率 {r:.3f}，位次中位 {m}")
    L.append("")

    (HERE / "phase3_robustness.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── 终端 ──────────────────────────────────────────────────────────
    print("══ A · 瓶颈在排序还是在判据线 ══")
    print(f"留出集 argmax 正确 {ax_ok}/{len(held_rows)}；过 TAU {len(passed)}/{len(held_rows)}")
    print(f"其中本身是 cue 原文的：{len(verbatim)}/{len(held_rows)}")
    if strict:
        print(f"真正改写句 {len(strict)} 句：argmax 正确 {sum(1 for r in strict if r[4])}/{len(strict)}，"
              f"过 TAU {sum(1 for r in strict if r[3] >= fp.TAU)}/{len(strict)}")
    for want, q, top_id, sc, ok, pos, cd, _vb in under:
        print(f"  未过线但 argmax {'对' if ok else '错'}：{q} → {top_id} {sc:.3f}（位次 {pos}）")
    print(f"越界负例 argmax 分 {[f'{x:.2f}' for x in neg_max]}，过线 {sum(1 for x in neg_max if x >= fp.TAU)}/{len(neg_max)}")
    if under:
        print(f"未过线留出句最低 {min(r[3] for r in under):.3f} vs 负例最高 {max(neg_max):.3f}")
    print("══ B · 入口扰动容忍度（t=2）══")
    for label, ds in (("d=0", [0]), ("d=1", [1]), ("d=2", [2]), ("d=3", [3]), (f"d>={FAR_MIN} 底", far)):
        r, m = rate(ds)
        if r is not None:
            print(f"  {label:<8s} 命中率 {r:.3f}  位次中位 {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
