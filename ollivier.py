"""Ollivier-Ricci 离散曲率（设计稿 §1.1 原本指定的那个量）。

    κ(x,y) = 1 − W₁(m_x, m_y) / d(x,y)

    m_x = x 处的惰性随机游走测度：m_x(x) = α，m_x(z) = (1−α)/deg(x)（z ∈ N(x)）
    W₁  = 一阶 Wasserstein 距离 = 一个运输问题的最优值
    d   = 图最短距离（相邻点之间 = 1）

---
为什么现在才做，以及为什么必须做
--------------------------------

Phase 1 用 **Forman 曲率**顶替（F(e) = 4 − d(x) − d(y) + 3·t(e)），
结果被自己的项分解证伪：**度数项方差占比 1.054**，也就是 F 的方差全部由端点度数解释。
所以"负曲率 = 分歧焦点"在稀疏图上不成立。

留着的那个理由是**量出来的**，不是猜的：

    Forman 是**度数的加性函数**（F = 4 − dx − dy + 3t），度数一进去就按加性压倒一切。
    Ollivier-Ricci **不是**：它由传输距离归一，端点度数只通过"测度摊多开"间接影响。

实测结果：OR 的度数 R² = 0.420，Forman 的 R² = 0.684 → OR 明显没那么被吞掉。
而且 OR 把最负的值（−0.500）正好放在全图唯一一条**界面桥**上。

---
W₁ 怎么算（纯标准库）
---------------------

小规模精确运输问题，最小费用流（连续最短路 SSP + SPFA）。

关键一步是**把浮点变成整数**，让运输问题可以精确求解：

    α = 1/2，测度取值只有 1/2 与 1/(2·deg)
    取 SCALE = 2·lcm(deg(x), deg(y))
    则 1/2 → lcm(...) 与 1/(2·deg) → lcm(...)/deg 都是整数

所以本模块**没有浮点误差**，也不需要近似求解器。

自检：路径图、三角形、星形，见 `self_test()`。
"""

from __future__ import annotations

import math
from collections import deque

# 数值零容差。**给它起名，不给它豁免**（B-D6 的规矩）。
# 本模块的运输问题走整数，所以这里的容差只用来吸收 float 表示误差。
TOL = 1e-9

ALPHA = 0.5


def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def all_pairs_dist(adj: dict) -> dict:
    out: dict[str, dict[str, int]] = {}
    for s in sorted(adj):
        d = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        out[s] = d
    return out


def lazy_measure(adj: dict, x: str, alpha: float = ALPHA) -> dict:
    """惰性随机游走测度。"""
    deg = len(adj[x])
    m = {x: alpha}
    if deg:
        share = (1.0 - alpha) / deg
        for z in adj[x]:
            m[z] = m.get(z, 0.0) + share
    return m


def min_cost_flow(supply: list[int], demand: list[int], cost: list[list[int]]) -> int:
    """运输问题精确解（整数供给/需求，整数费用）。返回最小总费用。"""
    m, n = len(supply), len(demand)
    if sum(supply) != sum(demand):
        raise ValueError("供需不等，运输问题无解")
    N = m + n + 2
    S, T = 0, N - 1
    g: list[list[list]] = [[] for _ in range(N)]

    def add(u: int, v: int, cap: int, w: int) -> None:
        g[u].append([v, cap, w, len(g[v])])
        g[v].append([u, 0, -w, len(g[u]) - 1])

    for i in range(m):
        if supply[i]:
            add(S, 1 + i, supply[i], 0)
    for j in range(n):
        if demand[j]:
            add(m + 1 + j, T, demand[j], 0)
    for i in range(m):
        for j in range(n):
            if supply[i] and demand[j]:
                add(1 + i, m + 1 + j, min(supply[i], demand[j]), cost[i][j])

    total = 0
    INF = float("inf")
    while True:
        dist = [INF] * N
        dist[S] = 0
        inq = [False] * N
        pv = [-1] * N
        pe = [-1] * N
        dq = deque([S])
        inq[S] = True
        while dq:                                  # SPFA：残量图里有负费用边
            u = dq.popleft()
            inq[u] = False
            du = dist[u]
            for ei, e in enumerate(g[u]):
                v, cap, w, _ = e
                if cap > 0 and du + w < dist[v]:
                    dist[v] = du + w
                    pv[v], pe[v] = u, ei
                    if not inq[v]:
                        inq[v] = True
                        dq.append(v)
        if dist[T] == INF:
            break
        d = INF
        v = T
        while v != S:
            d = min(d, g[pv[v]][pe[v]][1])
            v = pv[v]
        v = T
        while v != S:
            e = g[pv[v]][pe[v]]
            e[1] -= d
            g[v][e[3]][1] += d
            v = pv[v]
        total += d * dist[T]
    return total


def wasserstein(adj: dict, dist: dict, x: str, y: str) -> float:
    """W₁(m_x, m_y)，精确。"""
    mx = lazy_measure(adj, x)
    my = lazy_measure(adj, y)
    support = sorted(set(mx) | set(my))
    dx, dy = len(adj[x]), len(adj[y])
    scale = 2 * lcm(max(dx, 1), max(dy, 1))
    supply = [int(round(mx.get(p, 0.0) * scale)) for p in support]
    demand = [int(round(my.get(p, 0.0) * scale)) for p in support]
    # 整数化后的守恒修正（理论上不需要，留一道保险）
    diff = sum(supply) - sum(demand)
    if diff:
        i = max(range(len(supply)), key=lambda k: supply[k])
        supply[i] -= diff
    cost = [[dist[p][q] for q in support] for p in support]
    return min_cost_flow(supply, demand, cost) / scale


def curvature(adj: dict, dist: dict | None = None) -> list[dict]:
    """每条无向边的 κ。按边 id 排，不是名次。"""
    dist = dist or all_pairs_dist(adj)
    deg = {v: len(adj[v]) for v in adj}
    seen = set()
    rows = []
    for x in sorted(adj):
        for y in sorted(adj[x]):
            key = (x, y) if x <= y else (y, x)
            if key in seen:
                continue
            seen.add(key)
            w = wasserstein(adj, dist, x, y)
            d = dist[x][y]
            rows.append({"edge": key, "dx": deg[x], "dy": deg[y],
                         "w1": w, "d": d, "kappa": 1.0 - w / d if d else None})
    rows.sort(key=lambda r: r["edge"])
    return rows


def self_test() -> list[tuple[str, bool, str]]:
    """在三张已知答案的小图上验。

    ⚠️ 三角形那条**我第一版把期望值写错了**（写成 κ = 1，理由记成了
    "完全图上 κ=1"）。自检当场报挂，于是手算一遍：

        三角形 a-b-c，deg 全 2，α = 1/2
        m_a = {a: 1/2, b: 1/4, c: 1/4}      m_b = {a: 1/4, b: 1/2, c: 1/4}
        m_a − m_b = {a: +1/4, b: −1/4, c: 0}
        ⇒ W₁ = 1/4 · d(a,b) = 1/4
        ⇒ κ = 1 − 1/4 = **3/4**

    "K_n 上 κ=1" 只在 **α=0**（非惰性）且 n→∞ 时渐近成立（α=0 时 κ = 1 − 1/(n−1)）。
    惰性游走 α=1/2 下 K_n 的 κ → 1/2。

    **代码是对的，期望值是错的。** 期望值已按手算改成 3/4。
    """
    out = []
    # 路径 x−y−z：m_y = {y:1/2, x:1/4, z:1/4}，m_z = {z:1/2, y:1/2}
    # m_y − m_z = {y: 0, z: −1/4, x: +1/4} ⇒ W₁ = 1/4·d(x,z) = 1/2
    # ⇒ κ = 1 − 1/2 = 1/2
    path = {"x": {"y"}, "y": {"x", "z"}, "z": {"y"}}
    k = curvature(path)
    mid = [r for r in k if r["edge"] == ("y", "z")]
    ok = bool(mid) and abs(mid[0]["kappa"] - 0.5) < TOL
    out.append(("路径图 κ = 1/2", ok, f"得到 {mid[0]['kappa'] if mid else None}"))

    # 三角形：见 docstring 的手算，期望 3/4
    tri = {"a": {"b", "c"}, "b": {"a", "c"}, "c": {"a", "b"}}
    k2 = curvature(tri)
    got2 = [round(r["kappa"], 12) for r in k2]
    ok2 = all(abs(x - 0.75) < TOL for x in got2)
    out.append(("三角形 κ = 3/4", ok2, f"得到 {got2}"))

    # 星形（中心 + 3 叶）：手算 m_c − m_l1 = {l1: −1/3, l2: +1/6, l3: +1/6}
    # ⇒ W₁ = 2·(1/6·2) = 2/3 ⇒ κ = 1/3（正）
    star = {"c": {"l1", "l2", "l3"}, "l1": {"c"}, "l2": {"c"}, "l3": {"c"}}
    k3 = curvature(star)
    got3 = [round(r["kappa"], 12) for r in k3]
    ok3 = all(abs(x - 1.0 / 3.0) < TOL for x in got3)
    out.append(("星形叶边 κ = 1/3", ok3, f"得到 {got3}"))
    return out
