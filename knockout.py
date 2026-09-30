"""谱模式敲除：**替代删节点的损伤实验**。

---
为什么换掉删节点
----------------

Phase 4 的损伤实验是「把正确节点从临时副本里删掉，看共振掉不掉」。
它失败了，而且失败的原因是设计缺陷：

    删掉目标节点之后，**前 k 名入口本身也变了**，
    所以 ρ 的变化分不清是「结构被破坏」还是「换了一批入口」。

我当时靠"把入口集在损伤前后固定"打补丁，效应随即从 0.08–0.12 掉到 0.004–0.031。
那说明先前的效应大半是混淆造成的。

---
谱模式敲除怎么就没有这个毛病
----------------------------

不动图，动**场**。场在特征基下可以按模式分解：

    K = Σ_k c_k·u_k ,   c_k = ⟨u_k, K⟩

敲掉第 k 个模式就是

    K⁽ᵏ⁾ = K − c_k·u_k

**图一个字节不改**，因此入口集、度分布、桥、连通性全部原样：
混淆从设计上不存在，而不是靠事后固定。

---
预测（先登记）
--------------

    Fiedler 模（k = 1）是主切方向，所以敲掉它，
    **构造的变化应当最大**。

判据：把 n−1 个模式的"构造变化量"排序，看 k=1 是不是第一。
若 k=1 不突出，说明"主切决定构造"这句话不成立，如实写。

⚠️ 注意区分两种"效应"：

    场的变化量  ‖c_k‖        —— 只是投影系数的大小，不含新信息
    构造的变化量  读数变了多少 —— 这是非线性泛函，**只有它才是证据**

本模块两个都报，但判定只看后者。
"""

from __future__ import annotations

import math

import field as F
import ppr


def mode_coefficients(vecs: list[list[float]], field_vec: list[float]) -> list[float]:
    """把场投影到特征基上：`c_k = ⟨u_k, K⟩`。"""
    n = len(field_vec)
    return [sum(vecs[i][k] * field_vec[i] for i in range(n)) for k in range(n)]


def knockout(field_vec: list[float], vecs: list[list[float]], k: int) -> list[float]:
    """敲掉第 k 个模式后的场。**图不动。**"""
    n = len(field_vec)
    c = sum(vecs[i][k] * field_vec[i] for i in range(n))
    return [field_vec[i] - c * vecs[i][k] for i in range(n)]


def construction_change(adj: dict, order: list[str],
                        before: list[float], after: list[float]) -> dict:
    """构造变化量：读数（sweep cut 集合）变了多少，以及电导变了多少。

    Jaccard 变化 1−|交|/|并|：0 = 构造完全没变，1 = 完全不重叠。
    """
    b = ppr.sweep_cut(adj, order, before)
    a = ppr.sweep_cut(adj, order, after)
    if not (b.get("ok") and a.get("ok")):
        return {"jaccard_change": None, "cond_before": None, "cond_after": None}
    sb, sa = set(b["members"]), set(a["members"])
    u = len(sb | sa)
    jc = 1.0 - (len(sb & sa) / u if u else 0.0)
    return {"jaccard_change": jc, "cond_before": b["conductance"],
            "cond_after": a["conductance"],
            "k_before": b["k"], "k_after": a["k"]}


def sweep_all_modes(adj: dict, order: list[str], vecs: list[list[float]],
                    field_vec: list[float], zero_tol: float = 1e-12) -> list[dict]:
    """逐模式敲除，返回每个模式的两种效应。**按模式号排，不是名次。**"""
    n = len(order)
    coefs = mode_coefficients(vecs, field_vec)
    rows = []
    for k in range(n):
        if abs(coefs[k]) <= zero_tol:
            continue
        after = knockout(field_vec, vecs, k)
        ch = construction_change(adj, order, field_vec, after)
        field_delta = math.sqrt(sum((after[i] - field_vec[i]) ** 2 for i in range(n)))
        rows.append({"mode": k, "coef": coefs[k], "field_delta": field_delta, **ch})
    return rows


def largest_effect_mode(rows: list[dict]) -> int | None:
    usable = [r for r in rows if r["jaccard_change"] is not None]
    if not usable:
        return None
    # 并列按模式号小者优先，确定性
    best = max(usable, key=lambda r: (r["jaccard_change"], -r["mode"]))
    return best["mode"]
