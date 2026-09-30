"""Phase 1 第四步：把谱变成一个**空间**。

---
为什么这一步是关键
------------------

设计稿要的是"意图 = 方向"。方向必须有空间才谈得上。谱直接给出这个空间：

    扩散坐标   Ψ_k(i) = u_k(i) / √λ_k        （k = 1, 2, 3, …）

这不是随手选的 embedding。它有一个精确性质：

    R(i,j) = Σ_k (u_k(i) − u_k(j))² / λ_k = ‖Ψ(i) − Ψ(j)‖²

也就是说 **√R_eff 是一个欧氏距离，而 Ψ 就是它的坐标**。
有效电阻距离（Phase 1 第二步算的那个"真正的度量"）在这个坐标里
就是普通的欧氏距离。

于是：

    节点 → R^{n−1} 中的一个点
    意图 → 这个空间里的一个方向
    滑行 → 沿方向的位移

全部落在已有对象上，没有引入任何新的假设，也没有引入内容。

---
Fiedler 切：曲率的替代量
------------------------

Forman 曲率在本图上是度数代理（见 `curvature_split.py` 的分解），
符号不能用来区分"紧"与"薄"。退一步问：**场自己能不能指出结构在哪里变薄？**

候选量是 Fiedler 向量 u₁ 给出的主切：

    按 u₁ 的符号把节点分成两半
    跨过这个切的边 = 结构最薄的地方

这是有定论的：Fiedler 向量给出近似最优的稀疏切（Cheeger）。
本模块把切、跨切的边、以及切的电导（conductance）都算出来。

⚠️ 本模块不判定哪一半"更重要"。两半是等价的，切本身才是信息。
"""

from __future__ import annotations

import math


def embed_coords(vals: list[float], vecs: list[list[float]], order: list[str],
                 dim: int = 3, zero_eps: float = 1e-9) -> dict:
    """扩散坐标。跳过零特征值那一维（常数向量，不含结构信息）。"""
    n = len(order)
    keep = [k for k in range(n) if vals[k] > zero_eps][:dim]
    coords: dict[str, tuple[float, ...]] = {}
    for i, v in enumerate(order):
        coords[v] = tuple(vecs[i][k] / math.sqrt(vals[k]) for k in keep)
    return {"dims": len(keep), "lambda_used": [vals[k] for k in keep], "coords": coords}


def fiedler_cut(vals: list[float], vecs: list[list[float]], order: list[str],
                zero_eps: float = 1e-9) -> dict:
    """按 Fiedler 向量（第一个非平凡特征向量）的符号切两半。"""
    n = len(order)
    keep = [k for k in range(n) if vals[k] > zero_eps]
    if not keep:
        return {"ok": False, "why": "没有非平凡特征向量"}
    k = keep[0]
    vec = {order[i]: vecs[i][k] for i in range(n)}
    a = sorted([v for v in order if vec[v] > 0])
    b = sorted([v for v in order if vec[v] <= 0])
    return {"ok": True, "lambda": vals[k], "vector": vec,
            "side_a": a, "side_b": b}


def cut_edges(adj: dict, side_a: list[str]) -> list[tuple[str, str]]:
    A = set(side_a)
    out = []
    for x in sorted(adj):
        for y in adj[x]:
            if (x in A) != (y in A):
                key = (x, y) if x <= y else (y, x)
                if key not in out:
                    out.append(key)
    return sorted(out)


def node_volume(adj: dict, nodes: list[str]) -> float:
    return sum(sum(adj[x].values()) for x in nodes)


def conductance(adj: dict, side_a: list[str]) -> dict:
    """切电导 = 跨切边质量 / 两侧质量里较小的那个。"""
    A = set(side_a)
    all_nodes = list(adj)
    cross = 0.0
    seen = set()
    for x in all_nodes:
        for y, m in adj[x].items():
            if (x in A) != (y in A):
                key = (x, y) if x <= y else (y, x)
                if key in seen:
                    continue
                seen.add(key)
                cross += m
    vol_a = node_volume(adj, [x for x in all_nodes if x in A])
    vol_b = node_volume(adj, [x for x in all_nodes if x not in A])
    denom = min(vol_a, vol_b)
    return {"cut_mass": cross, "vol_a": vol_a, "vol_b": vol_b,
            "conductance": (cross / denom) if denom else None}
