"""PPR + sweep cut：**替代水平集读数**。

---
为什么换掉读数
--------------

原来的读数是「场自己均值以上的节点」，Phase 2 已经量出它丢信息：

    场层意图恢复率 1.000    软读数 1.000    硬集合 0.438（零线 0.062）

**信息是在"丢掉幅度、只比集合"这一步没的。** 而水平集还有一个更根本的
毛病：它拿"均值"当水平，而均值是**一个拍出来的位置**——为什么是均值不是
中位数、不是某条分位线？换一个位置，读数就换一批节点。

---
sweep cut 是什么，为什么它没有这个毛病
--------------------------------------

先算一个**局部质量分布**（PPR 向量），再把节点按质量从大到小排，
然后**把每一个前缀的电导都算出来，取电导最小的那个前缀**：

    pr      = PPR(种子)                 局部质量分布
    φ(k)    = 电导(质量最高的 k 个)       k = 1 .. n−1
    读数    = argmin_k φ(k) 那个前缀

关键在于**取的是最小值**：最小值是这条曲线自己的性质，不是我选的位置。
没有分位线、没有均值、没有任何位置参数。

而且它有定论：sweep cut 是**局部切分**的标准做法，局部 Cheeger 不等式
保证它找到一个局部簇（Andersen–Chung–Lang 一类结果）。
本模块**不引用具体常数**，而是把曲线打出来让人看，并且：
**一条平坦的 sweep 曲线（所有前缀的电导都不小）= 种子周围没有可分离的
局部结构 = 场在这一带没话可说**，这就是 omission 的判据。

---
参数怎么定（τ 与"电导小不小"）
------------------------------

PPR 有一个传送参数 c。本模块**扫一整个网格**，不做单一选择，
并报告结果对该参数稳不稳。

"电导小不小"**不设线**，而是拿**这张图自己的节点当种子**算出零分布：
每个节点做一次种子，得到 n 个 φ* 值，构成 null。一条查询的 φ* 在这条
null 里的分位就是它的校准读数——**这就是 conformal 的做法在图上落地**。
没有拍出来的线，只有"它在图自己的分布里排在哪"。
"""

from __future__ import annotations

import math

import field as F


def personalized_pr(adj: dict, order: list[str], seed: str, c: float,
                    max_iter: int = 20000):
    """几何 PPR：`pr = (1−c)·s + c·Pᵀ·pr`，`P = D⁻¹A`。

    幂迭代到**浮点收敛**（更新量归零）或撞上 max_iter。确定性、无随机数。

    返回 (向量, 迭代次数, 最终更新量, 是否撞上限)。
    **撞上限必须报出来**：c 越接近 1 收敛越慢，撞上限时结果仍然可用
    （残差可能已在浮点分辨率上），但"靠上限停下"和"真的收敛"不是一回事。
    """
    n = len(order)
    idx = {v: i for i, v in enumerate(order)}
    pr = [0.0] * n
    pr[idx[seed]] = 1.0
    deg = [len(adj[v]) for v in order]
    it = 0
    delta = float("inf")
    hit_cap = True
    for it in range(1, max_iter + 1):
        new = [(1.0 - c) * (1.0 if i == idx[seed] else 0.0) for i in range(n)]
        for i, u in enumerate(order):
            d = deg[i]
            if not d:
                continue
            share = c * pr[i] / d
            for v in adj[u]:
                new[idx[v]] += share
        delta = sum(abs(new[i] - pr[i]) for i in range(n))
        pr = new
        if delta == 0.0:
            hit_cap = False
            break
    return pr, it, delta, hit_cap


def heat_pr(vals, vecs, order: list[str], seed_index: int, t: float) -> list[float]:
    """热核版本的局部质量分布：`exp(−tL)·δ_seed`。复用 Phase 1 的机器。"""
    n = len(order)
    delta = [1.0 if i == seed_index else 0.0 for i in range(n)]
    return F.heat(vals, vecs, delta, t)


def set_conductance(adj: dict, S: set) -> dict:
    """一个顶点集的电导：cut / min(vol(S), vol(V∖S))。无参数。"""
    allv = set(adj)
    inside = S
    cut = 0.0
    seen = set()
    for x in inside:
        for y, m in adj[x].items():
            if y in inside:
                continue
            key = (x, y) if x <= y else (y, x)
            if key in seen:
                continue
            seen.add(key)
            cut += m
    vol_s = sum(sum(adj[x].values()) for x in inside)
    vol_c = sum(sum(adj[x].values()) for x in allv - inside)
    denom = min(vol_s, vol_c)
    return {"cut": cut, "vol_s": vol_s, "vol_c": vol_c,
            "conductance": (cut / denom) if denom else None}


def sweep_curve(adj: dict, order: list[str], mass: list[float]) -> list[dict]:
    """按质量从大到小扫全部前缀，返回每个前缀的电导。

    并列按 id 破，确定性。前缀长度取 1 .. n−1（全图和空集不算切）。
    """
    ordered = sorted(range(len(order)), key=lambda i: (-mass[i], order[i]))
    curve = []
    S: set = set()
    for k in range(1, len(ordered)):
        S.add(order[ordered[k - 1]])
        c = set_conductance(adj, S)
        curve.append({"k": k, "conductance": c["conductance"],
                      "vol_s": c["vol_s"], "cut": c["cut"]})
    return curve


def sweep_cut(adj: dict, order: list[str], mass: list[float]) -> dict:
    """sweep cut：电导最小的那个前缀。**取最小值，所以没有位置参数。**"""
    curve = sweep_curve(adj, order, mass)
    if not curve:
        return {"ok": False}
    best = min(curve, key=lambda r: (r["conductance"] if r["conductance"] is not None else 1e18,
                                     r["k"]))
    ordered = sorted(range(len(order)), key=lambda i: (-mass[i], order[i]))
    members = sorted(order[i] for i in ordered[:best["k"]])
    return {"ok": True, "k": best["k"], "conductance": best["conductance"],
            "members": members, "curve": curve}


def node_seed_null(adj: dict, order: list[str], c: float) -> dict:
    """**这张图自己的零分布**：每个节点当一次种子，各得一个 φ*。

    它是"电导小不小"的参照，替代任何拍出来的线。
    返回 {节点: φ*} 与排序后的值列表。
    """
    vals = {}
    for v in order:
        pr, _it, _d, _cap = personalized_pr(adj, order, v, c)
        vals[v] = sweep_cut(adj, order, pr)["conductance"]
    ordered = sorted(x for x in vals.values() if x is not None)
    return {"phi": vals, "sorted": ordered,
            "min": ordered[0] if ordered else None,
            "max": ordered[-1] if ordered else None,
            "n": len(ordered)}


def calibrated_reading(phi: float, null: dict) -> float:
    """φ 在这张图的零分布里的分位（越低越"不寻常"）。

    无参数：只用 null 自己的排序位置。返回 0..1。
    """
    s = null["sorted"]
    if not s or phi is None:
        return None
    below = sum(1 for x in s if x < phi)
    equal = sum(1 for x in s if x == phi)
    return (below + 0.5 * equal) / len(s)


def null_by_size(null: dict) -> dict:
    """按前缀规模分桶的零分布。

    为什么要它：sweep cut 的规模本身会影响电导的可达范围（小前缀的电导
    天然受限）。拿整体 null 去比会偏。分桶之后再校准，是同一件事的细版。
    """
    buckets: dict[int, list[float]] = {}
    for v, phi in null["phi"].items():
        if phi is None:
            continue
        buckets.setdefault(v, []).append(phi)
    return buckets
