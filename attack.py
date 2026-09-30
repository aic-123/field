"""ASPIC+ / ABA 的三分攻击类型学，以及**边具体化**。

---
三分攻击，成熟词汇
------------------

计算论证里标准的攻击分类（ASPIC+ / ABA 一线）：

    rebut      攻击**结论**：对同一对节点给出互斥的主张
    undermine  攻击**前提**：说对方的某个前提本身不成立/未确立
    undercut   攻击**推理链**：不否认前提，否认"从这些前提能推出那个结论"

图尔敏的对应：rebut 打 claim，undercut 打 warrant，undermine 打 grounds。

---
为什么这一条对 DCE 是要紧的
---------------------------

DCE 只有一个 `contradiction`，会把这三件事混成一件事。而它们的可计算性**完全不同**：

    rebut      需要在同一对端点上比较**关系种类**                → 需要边带种类
    undermine  只需要节点级的状态（`evidence_status`）            → **单视图内部就有**
    undercut   需要一个**可以被指着的规则对象**（warrant）        → 需要边被具体化

于是本模块的第一件事不是分类，是**问数据里有没有这些东西**。
答案是：**两个都没有。**

    Scaffold 的 `relations` 是 `list[str]`，**不带关系种类**（`SPEC.md:128`）
    也没有任何"规则"对象

所以：**rebut 在本数据上不可表达，undercut 也不可表达，只有 undermine 可以。**
这不是分类不够细，是缺对象。

---
本模块做什么
------------

    `expressiveness(...)`  如实报出三种攻击在本数据上各自可不可表达
    `reify(...)`           演示：把一条边具体化成一个规则对象之后，undercut 才可表达

**reify 是演示，不是推荐立刻上——那是一次结构扩建，规模与 Phase 7 的多来源同量级。**
"""

from __future__ import annotations

REBUT = "rebut"
UNDERMINE = "undermine"
UNDERCUT = "undercut"

# 节点级"前提不成立"的判据：Scaffold 自己的状态词表
# （`schema/node.schema.yaml:71`，`user_writable` 里的五个）
UNSETTLED_STATUS = ("已淘汰", "有争议", "未验证")


def edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def reify(rule_id: str, premises: list[str], conclusion: str, asserted_by: str) -> dict:
    """把一条边**具体化**成一个可以被指着的规则对象（warrant）。

    这就是 ASPIC+ 里"规则是一等对象"的最小实现：
    它有了自己的 id，于是"攻击这条推理链"才有靶子。
    """
    return {"rule_id": rule_id, "premises": sorted(premises),
            "conclusion": conclusion, "asserted_by": asserted_by}


def classify_rebut(edges_a: dict, edges_b: dict) -> list[dict]:
    """rebut：同一对端点，两个视图给出**互斥的关系种类**。

    `edges` 的形状是 {(from, to): relation_kind}。
    ⚠️ 本数据里关系种类**不存在**，所以这个函数在真实数据上拿不到输入。
    """
    out = []
    for k in sorted(set(edges_a) & set(edges_b)):
        ka, kb = edges_a[k], edges_b[k]
        if ka != kb:
            out.append({"edge": k, "a": ka, "b": kb, "type": REBUT})
    return out


def classify_undermine(status_a: dict, status_b: dict) -> list[dict]:
    """undermine：一个视图把某个节点当作成立，另一个视图把它标成未确立。

    **这一条在真实数据上可以直接算**：`evidence_status` 就在节点里。
    """
    out = []
    for nid in sorted(set(status_a) | set(status_b)):
        sa = status_a.get(nid)
        sb = status_b.get(nid)
        if sa is None or sb is None:
            continue
        a_unsettled = sa in UNSETTLED_STATUS
        b_unsettled = sb in UNSETTLED_STATUS
        if a_unsettled != b_unsettled:
            out.append({"node": nid, "a": sa, "b": sb, "type": UNDERMINE})
    return out


def classify_undercut(rules_a: dict, rules_b: dict) -> list[dict]:
    """undercut：攻击的目标是一个**规则对象**（它有 rule_id）。

    没有规则对象时这个函数恒返回空 —— 那不是"没有 undercut"，
    是"**undercut 无从表达**"。两者的区别必须写清楚。
    """
    out = []
    for rid in sorted(set(rules_a) & set(rules_b)):
        ra, rb = rules_a[rid], rules_b[rid]
        if ra["conclusion"] != rb["conclusion"] or ra["premises"] != rb["premises"]:
            out.append({"rule_id": rid, "a": ra, "b": rb, "type": UNDERCUT})
    return out


def expressiveness(has_relation_kinds: bool, has_rule_objects: bool,
                   n_rebut: int = 0, n_undermine: int = 0, n_undercut: int = 0) -> dict:
    """如实报出三种攻击各自可不可表达。"""
    return {
        "rebut": {"expressible": has_relation_kinds, "count": n_rebut,
                  "why": ("边带关系种类，可以比较" if has_relation_kinds
                          else "**不可表达**：`relations` 是 list[str]，不带关系种类")},
        "undermine": {"expressible": True, "count": n_undermine,
                      "why": "`evidence_status` 在节点上，单视图内部即可算"},
        "undercut": {"expressible": has_rule_objects, "count": n_undercut,
                     "why": ("有规则对象，可以指着它攻击" if has_rule_objects
                             else "**不可表达**：没有任何规则对象（warrant）可被指向")},
    }
