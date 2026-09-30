"""Phase 7a：方向一致性 consensus + 多数派免疫（方法推荐 ⑦）。

三个部分，**第二部分是这一轮被自己的测试推翻过一次的地方**：

    机制验证   埋一个已知共享方向，做**噪声扫描**看两种规则各在什么信噪比下捞得回来
    真实数据   用真图的热核当视图，看共享方向、各视图投影与残差
    免疫测试   **三种规则**各受复制多少影响：求和均值 / 第一主成分 / max-min

---
为什么免疫测试要三种规则
------------------------

第一版只测了第一主成分，结果 **|cos| 掉到 0.40–0.52**——它不免疫。
原因是结构性的：第一主成分是**求和**意义下最优，
复制一个视图就等于把它的权重乘 100。

而 `§C7.1 ①` 要的「跨群共识」是**合取**（"要求同时满足所有群，单群不成立"），
不是求和。合取的正确形式化是 **max-min**：

    d* = argmax_{‖d‖=1} min_i ⟨f_i, d⟩

min 对重复成员不敏感（重复项的值与原件相同），所以它才是那句
"结构上不可能违反 #5"的正确落地。**本模块把三种规则一起报，让分界可见。**
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import spectral as S       # noqa: E402
import field as F          # noqa: E402
import direction as D      # noqa: E402
import ppr as PPR          # noqa: E402

T_ARC = 2.0
NOISE_GRID = [0.05, 0.1, 0.2, 0.4, 0.8]

# 判据线（**先写死，再跑**，且都是惯例值不是为本图调的）：
#   捞回埋点方向：|cos| > 0.99 视为捞回来
RECOVER_COS = 0.99
#   多数派免疫：把视图复制若干遍后 |cos| > 0.999 才算免疫
#   （免疫是"逐位相同"级别的性质，所以线比上一条紧一个量级）
IMMUNITY_COS = 0.999


def main() -> int:
    raw, _ = G.load_nodes()
    full = G.build(raw)
    node_ids = full["node_ids"]
    adj = G.subgraph(full["adj"], node_ids)
    vals, vecs = S.jacobi(S.laplacian(adj, node_ids))
    n = len(node_ids)
    didx = {v: i for i, v in enumerate(node_ids)}

    L: list[str] = []
    L.append("# Phase 7a · 方向一致性 consensus\n")
    L.append("替代「数节点出现在几个视图里」——那是热度信号，`arena §C7.1 ①` 明禁。")
    L.append("改为看**方向**：共享方向是共识，各视图的残差是分歧。\n")

    # ── 机制验证：噪声扫描 ────────────────────────────────────────────
    planted = [vecs[i][1] for i in range(n)]      # Fiedler 向量当埋点方向
    L.append("## 机制验证：埋一个已知共享方向，扫噪声\n")
    L.append("埋点方向 = Fiedler 向量 u₁（4 个视图 = 埋点 + 各自独立噪声）。")
    L.append("两种规则各捞回多少：\n")
    L.append("| 噪声幅度 | 第一主成分 \\|cos\\| | max-min \\|cos\\| | max-min 的最小投影 |")
    L.append("|---|---|---|---|")
    sweep = []
    for noise_amp in NOISE_GRID:
        fields_p = []
        for j in range(4):
            nz = [math.sin((i + 1) * (j + 2) * 0.7) * noise_amp for i in range(n)]
            fields_p.append([planted[i] + nz[i] for i in range(n)])
        c_pc = D.cos_sim(D.shared_direction(fields_p)["direction"], planted)
        ag = D.agreeing_direction(fields_p)
        c_ag = D.cos_sim(ag["direction"], planted)
        L.append(f"| {noise_amp} | {c_pc:.4f} | **{c_ag:.4f}** | {ag['min_projection']:+.4f} |")
        sweep.append((noise_amp, c_pc, c_ag))
    L.append("")
    L.append("- 判据（先写死）：\\|cos\\| > 0.99 视为捞回来。")
    ok_pc = [x for x in sweep if x[1] > RECOVER_COS]
    ok_ag = [x for x in sweep if x[2] > RECOVER_COS]
    L.append(f"- 第一主成分捞回来的档位：{len(ok_pc)}/{len(sweep)}"
             f"（噪声 {'、'.join(str(x[0]) for x in ok_pc) or '无'}）")
    L.append(f"- max-min 捞回来的档位：{len(ok_ag)}/{len(sweep)}"
             f"（噪声 {'、'.join(str(x[0]) for x in ok_ag) or '无'}）")
    L.append("")
    L.append("**读法**：噪声越大，第一主成分越容易被噪声的公共分量拖走；")
    L.append("max-min 优化的是**最差的那个视图**，所以它是防御性的。")
    L.append("两者在高信噪比下应当一致，在低信噪比下分道扬镳；")
    L.append("若两者处处一致，说明这个区分在本数据上没有意义，如实写。\n")

    # ── 真实数据 ──────────────────────────────────────────────────────
    L.append("## 真实数据：拿 4 个节点的热核当视图\n")
    seeds = ["con-0006", "judge-0009", "case-0003", "stance-0003"]
    fields = [F.heat(vals, vecs, [1.0 if i == didx[s] else 0.0 for i in range(n)], T_ARC)
              for s in seeds]
    vs_pc = D.variance_share(fields, D.shared_direction(fields)["direction"])
    ag = D.agreeing_direction(fields)
    res = D.residuals(fields, D.shared_direction(fields)["direction"])
    L.append(f"- 视图（种子）：{'、'.join(seeds)}")
    L.append(f"- 第一主成分占总能量 = **{vs_pc:.4f}**")
    L.append(f"- max-min 方向下，各视图的投影 = {[round(x, 4) for x in ag['projections']]}")
    L.append(f"- 最小投影（共识的「强度下界」）= **{ag['min_projection']:+.4f}**")
    L.append(f"- 各视图残差 = {[round(x, 4) for x in res]}")
    L.append("- 残差最大的是与其它三个最不一致的那个视图 —— **它是分歧所在，不是错误**\n")

    # ── 免疫测试（三种规则 × 四个被复制的视图）────────────────────────
    L.append("## 多数派免疫测试（把一个视图复制 100 遍）\n")
    views = []
    for s in seeds:
        pr, _it, _d, _cap = PPR.personalized_pr(adj, node_ids, s, 0.15)
        top = sorted(node_ids, key=lambda v: (-pr[didx[v]], v))[:8]
        views.append(set(top))
    L.append("| 被复制的视图 | 求和均值 \\|cos\\| | 第一主成分 \\|cos\\| | **max-min \\|cos\\|** |")
    L.append("|---|---|---|---|")
    imm = []
    for dup in range(len(seeds)):
        r = D.popularity_immunity_test(fields, views, duplicate_index=dup, copies=100)
        L.append(f"| {dup+1}（{seeds[dup]}） | {r['cos_mean']:.4f} | {r['cos_pc']:.4f} | "
                 f"**{r['cos_agree']:.4f}** |")
        imm.append((dup, r))
    L.append("")
    r0 = imm[0][1]
    pc_immune = all(x[1]["cos_pc"] > IMMUNITY_COS for x in imm)
    ag_immune = all(x[1]["cos_agree"] > IMMUNITY_COS for x in imm)
    mean_immune = all(x[1]["cos_mean"] > IMMUNITY_COS for x in imm)
    L.append("### 判定（判据先写死：所有复制档下 \\|cos\\| > 0.999 才算免疫）\n")
    L.append(f"- 求和均值方向免疫：**{mean_immune}**")
    L.append(f"- 第一主成分免疫：**{pc_immune}**")
    L.append(f"- **max-min 免疫：{ag_immune}**")
    L.append("")
    L.append(f"- 条数规则（对照）：复制前选出 `{r0['plurality_before']}`，"
             f"复制后 `{r0['plurality_after']}`，"
             f"{'变了' if r0['plurality_changed'] else '没变'}")
    L.append("  ⚠️ 这一格**在本数据上没有区分力**：四个视图的 top-8 有重叠，")
    L.append("  而被复制的那个视图的节点本来就是条数胜出者，所以复制不改变结果。")
    L.append("  它是个弱对照，不能拿它当「条数规则不免疫」的证据。")
    L.append("")
    L.append("### 这一轮被自己的测试推翻过一次\n")
    L.append("第一版只实现了第一主成分，并声称「看方向不看数量 ⇒ 多数派免疫」。")
    L.append("免疫测试当场给出 **|cos| = 0.40–0.52** —— **不免疫**，")
    L.append("于是那句声称被自己的检查打掉，换成了 max-min。")
    L.append("**这正是把「免疫」做成可测性质的意义**：它是个可能为假的命题，")
    L.append("而不是一句关于设计的形容词。")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 没有做真正的多来源视图比较。真实数据只有单来源（一张 Scaffold 图），")
    L.append("  这里的「视图」是用不同种子造出来的，属于机制演示，**不是多来源评测**。")
    L.append("- max-min 用投影次梯度上升求解，是局部方法，不保证全局最优；")
    L.append("  只报了它达到的最小投影，没有报与全局最优的差距。")
    paths.report("phase7a_direction.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── 方向一致性 ──")
    print("  机制验证（噪声扫描）：")
    for amp, c_pc, c_ag in sweep:
        print(f"    噪声 {amp:<5} 主成分 {c_pc:.4f}   max-min {c_ag:.4f}")
    print(f"  真实数据：主成分占比 {vs_pc:.4f}；max-min 最小投影 {ag['min_projection']:+.4f}")
    print("  免疫测试：")
    for dup, r in imm:
        print(f"    复制第 {dup+1} 个（{seeds[dup]:<12s}）"
              f" 均值 {r['cos_mean']:.4f}  主成分 {r['cos_pc']:.4f}  "
              f"max-min {r['cos_agree']:.4f}")
    print(f"  免疫：均值 {mean_immune} / 主成分 {pc_immune} / **max-min {ag_immune}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
