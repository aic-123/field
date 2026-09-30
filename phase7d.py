"""Phase 7d：攻击类型学（⑨）与随机图上的不变量测试（⑩）。

---
A · 三分攻击在本数据上到底能不能算
----------------------------------

这是本模块最要紧的一张表：**三种攻击各自需要什么数据**。

| 攻击 | 需要什么 | 本数据有没有 |
|---|---|---|
| rebut | 同一对端点上比较**关系种类** | ❌ `relations` 是 `list[str]`，不带种类 |
| undermine | 同一节点上比较**两个视图的状态** | ⚠️ 有 `evidence_status`，但**只有一个视图** |
| undercut | 一个可被指着的**规则对象** | ❌ 没有任何规则对象 |

**而且更靠前的一条结论是：单来源的任何一种都算不出来。**
分歧需要至少两个视图——这和 Phase 1 判决 §3 是同一条。

所以本模块分两半：
    如实报出真实数据上的计数（全是 0）与每个 0 的原因
    用**构造出来的两个视图**演示每一类都能触发，把"需要什么"钉成可执行的
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import paths               # noqa: E402

import graph as G          # noqa: E402
import attack as AT        # noqa: E402
import invariants as INV   # noqa: E402


def main() -> int:
    raw, _ = G.load_nodes()
    status = {nid: ("有争议" if (nid.startswith("issue")) else "未验证") for nid in raw}
    rel_pairs = G.build(raw)["rel_pairs"]

    L: list[str] = []
    L.append("# Phase 7d · 攻击类型学与不变量测试\n")
    L.append("# A · 三分攻击（ASPIC+ / ABA 的标准词汇）\n")
    L.append("## 真实数据上的可表达性\n")
    L.append("| 攻击 | 需要什么数据 | 本数据 | 计数 |")
    L.append("|---|---|---|---|")
    expr = AT.expressiveness(has_relation_kinds=False, has_rule_objects=False)
    L.append(f"| rebut | 同一对端点上比较关系种类 | "
             f"{'有' if expr['rebut']['expressible'] else '**无**'} | {expr['rebut']['count']} |")
    L.append(f"| undermine | 同一节点上的两个视图状态 | 有状态、无第二视图 | "
             f"{expr['undermine']['count']} |")
    L.append(f"| undercut | 一个可被指着的规则对象 | "
             f"{'有' if expr['undercut']['expressible'] else '**无**'} | "
             f"{expr['undercut']['count']} |")
    L.append("")
    L.append(f"- rebut 为什么是 0：{expr['rebut']['why']}")
    L.append(f"- undermine 为什么是 0：只有一个视图，**单来源没有分歧可言**")
    L.append(f"- undercut 为什么是 0：{expr['undercut']['why']}")
    L.append("")
    L.append("**三条都是 0，但三个 0 的含义完全不同**：")
    L.append("rebut 与 undercut 是**结构上表达不了**，undermine 是**缺第二个视图**。")
    L.append("把它们都记成「无分歧」是本模块要防的那种混淆。\n")

    L.append("## 把「需要什么」钉成可执行的：构造两个视图逐个触发\n")
    edges_a = {("x", "y"): "supports", ("p", "q"): "refutes"}
    edges_b = {("x", "y"): "contradicts", ("p", "q"): "refutes"}
    rb = AT.classify_rebut(edges_a, edges_b)
    L.append(f"- **rebut**：同一对端点给出互斥种类 → 触发 {len(rb)} 条 {[r['edge'] for r in rb]}")
    sa = {"n1": "已确立", "n2": "未验证"}
    sb = {"n1": "未验证", "n2": "未验证"}
    um = AT.classify_undermine(sa, sb)
    L.append(f"- **undermine**：一个视图当作成立、另一个标未确立 → 触发 {len(um)} 条 "
             f"{[r['node'] for r in um]}")
    r1 = AT.reify("rule-1", ["x"], "y", "human:alice")
    r2 = AT.reify("rule-1", ["x", "z"], "y", "human:bob")
    uc = AT.classify_undercut({"rule-1": r1}, {"rule-1": r2})
    L.append(f"- **undercut**（**必须先具体化**）：同一 rule_id 的前提集不同 "
             f"→ 触发 {len(uc)} 条 {[r['rule_id'] for r in uc]}")
    L.append("")
    L.append("### 具体化（reification）演示\n")
    L.append("```")
    L.append(f"具体化前：边 (x → y) 是一个二元组，**没有 id**，攻击指不到它")
    L.append(f"具体化后：{r1}")
    L.append(f"          {r2}")
    L.append(f"          → 两者前提集不同，undercut 才有靶子")
    L.append("```")
    L.append("")
    L.append("⚠️ **reify 是演示，不是推荐立刻上。** 它把图从「节点 + 边」扩成")
    L.append("「节点 + 边 + 规则」，规模与 Phase 7 的多来源同量级，")
    L.append("必须走一次结构变更流程，不能顺手加。\n")
    L.append("### 与 Phase 1 判决的呼应\n")
    L.append("Phase 1 §3 已经得出「单来源的图里不可能有分歧焦点」。")
    L.append("这里只是把同一件事从**类型学**这一侧再确认一次：")
    L.append("**三种攻击在单来源下全部为 0，而其中两种即使有了第二个视图也仍然为 0。**\n")

    # ── B 部分：不变量测试 ────────────────────────────────────────────
    L.append("# B · 随机图上的不变量测试\n")
    res = INV.run_all(trials=24, n=13, m=26)
    L.append("在 24 张随机图上各跑一遍。**不变量只有两种结局：成立，或被反例打掉。**\n")
    L.append("| 不变量 | 通过 | 反例 | 说明 |")
    L.append("|---|---|---|---|")
    for name, _fn in INV.CHECKS:
        r = res[name]
        note = ""
        if r["examples"]:
            note = f"反例：{r['examples'][0]}"
        elif name == "OR 落在 [−2,1]" and r["extra"]:
            lo = min(x[0] for x in r["extra"])
            hi = max(x[1] for x in r["extra"])
            note = f"实测范围 {lo:+.4f} .. {hi:+.4f}"
        L.append(f"| {name} | **{r['pass']}/{r['pass']+r['fail']}** | {r['fail']} | {note} |")
    L.append("")
    allok = all(res[n]["fail"] == 0 for n, _ in INV.CHECKS)
    L.append(f"### 判定\n")
    L.append(f"- 六条不变量全部通过：**{allok}**")
    if not allok:
        L.append("- **有反例，照实列在上面，不挑好看的。**")
    L.append("")
    L.append("### 没做的事\n")
    L.append("- 随机图只到 13 个节点（Jacobi 是 O(n³)、有效电阻三角不等式是 O(n³)，")
    L.append("  再大就跑不动了）。**所以「成立」只在中小规模上成立。**")
    L.append("- 没有把不变量测试接进 CI（本模块不是 CI 的一部分）。")
    paths.report("phase7d_attack_and_invariants.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("── A 三分攻击 ──")
    print(f"  真实数据：rebut 可表达={expr['rebut']['expressible']}  "
          f"undermine 可表达=True  undercut 可表达={expr['undercut']['expressible']}")
    print(f"  构造视图触发：rebut {len(rb)}  undermine {len(um)}  undercut {len(uc)}")
    print("── B 随机图不变量 ──")
    for name, _fn in INV.CHECKS:
        r = res[name]
        extra = ""
        if name == "OR 落在 [−2,1]" and r["extra"]:
            extra = (f"  范围 {min(x[0] for x in r['extra']):+.4f}"
                     f" .. {max(x[1] for x in r['extra']):+.4f}")
        print(f"  {name:<24s} 通过 {r['pass']}/{r['pass']+r['fail']}{extra}")
    print(f"  全部通过 = {allok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
