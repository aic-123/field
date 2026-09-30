"""有向谱与有向 Ollivier-Ricci：**把已经丢掉的方向捡回来**。

---
为什么这里有免费的信号
----------------------

`graph.py` 建图时把 `relations` 做成了**无向**边（为了拉普拉斯对称、好做特征分解）。
但数据本身是有向的：

    论据 → 立场 → 议题
    例外 → 被覆盖的节点
    概念：具体 → 基础

这些方向在 `graph.py` 里被保存成 `rel_pairs`，但**从没进入过任何计算**。
也就是说：方向是**已经有、却完全没用上**的结构。

---
有向图给了什么
--------------

1. **平稳分布 π**。有向随机游走 `P = D_out⁻¹A` 的平稳分布一般**不是**均匀的；
   它把"被很多节点指向"的位置抬高。这和度分布是两件不同的事。
2. **Chung 的有向拉普拉斯**：用 π 做对称化

       L = I − ½(Φ^{1/2} P Φ^{−1/2} + Φ^{−1/2} Pᵀ Φ^{1/2})，  Φ = diag(π)

   它是实对称的，所以谱是实的，可以直接和 Phase 1 的谱对比。
3. **有向 Ollivier-Ricci**：把惰性测度换成**出邻居**上的测度，
   而传输代价仍用底层的无向距离。`ollivier.lazy_measure` 只读 `adj[x]`，
   所以传一个有向邻接表进去就是有向版，机器不用改。

---
本模块不做的事
--------------

- 不改 `graph.py`。无向图仍然是无向图，方向只在需要时另建一张。
- 不假设有向版一定更好。**先量出来再说。**
"""

from __future__ import annotations

import math

import ollivier as O


def directed_adj(rel_pairs: list[tuple[str, str]], node_ids: list[str]) -> dict:
    """从 `relations` 的方向建**出邻居**邻接表。边质量都为 1。"""
    adj: dict[str, dict[str, float]] = {v: {} for v in node_ids}
    for a, b in rel_pairs:
        if a in adj and b in adj:
            adj[a][b] = adj[a].get(b, 0.0) + 1.0
    return adj


def out_degree(adj_dir: dict) -> dict:
    return {v: len(nb) for v, nb in adj_dir.items()}


def stationary(adj_dir: dict, order: list[str], iters: int = 20000):
    """有向随机游走的平稳分布。幂迭代到浮点收敛（或撞上限）。

    悬挂点（出度为 0）按均匀分配处理，否则质量会漏掉。
    """
    n = len(order)
    idx = {v: i for i, v in enumerate(order)}
    pi = [1.0 / n] * n
    outd = [len(adj_dir[v]) for v in order]
    it = 0
    delta = float("inf")
    hit_cap = True
    for it in range(1, iters + 1):
        new = [0.0] * n
        dangling = 0.0
        for i, u in enumerate(order):
            d = outd[i]
            if not d:
                dangling += pi[i] / n
                continue
            share = pi[i] / d
            for v in adj_dir[u]:
                new[idx[v]] += share
        for i in range(n):
            new[i] += dangling
        delta = sum(abs(new[i] - pi[i]) for i in range(n))
        pi = new
        if delta == 0.0:
            hit_cap = False
            break
    return pi, it, delta, hit_cap


def row_stochastic(adj_dir: dict, order: list[str]):
    """行随机矩阵 P。**悬挂点（出度 0）按均匀分配**。

    ⚠️ 这一步不能省。第一版让悬挂点留成零行，于是 P 不是行随机的，
    而对称化公式 `I − ½(Φ^{1/2}PΦ^{−1/2} + Φ^{−1/2}PᵀΦ^{1/2})` 的前提
    正是「P 行随机」（否则 A = Φ^{1/2}PΦ^{−1/2} 的特征值会跑出单位圆，
    对称化后出现负特征值）。实测就是那样：λ₁ = −1.585。
    """
    n = len(order)
    idx = {v: i for i, v in enumerate(order)}
    P = [[0.0] * n for _ in range(n)]
    for u in order:
        d = len(adj_dir[u])
        if not d:
            share = 1.0 / n
            for j in range(n):
                P[idx[u]][j] = share
            continue
        for v in adj_dir[u]:
            P[idx[u]][idx[v]] = 1.0 / d
    return P


def chung_laplacian(adj_dir: dict, order: list[str], pi: list[float]):
    """Chung 的有向拉普拉斯（实对称）。返回稠密矩阵。

    `P` 由 `row_stochastic` 给出（悬挂点走均匀），所以前提成立。
    π 与 P 必须来自**同一个**处理方式，否则 Φ 的归一化对不上。
    """
    n = len(order)
    P = row_stochastic(adj_dir, order)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if pi[i] <= 0 or pi[j] <= 0:
                continue
            r = math.sqrt(pi[j] / pi[i])
            s = math.sqrt(pi[i] / pi[j])
            L[i][j] = -0.5 * (P[i][j] * r + P[j][i] * s)
    for i in range(n):
        L[i][i] += 1.0
    return L


def strongly_connected(adj_dir: dict, order: list[str]) -> list[list[str]]:
    """Kosaraju 强连通分量（迭代版，不吃递归深度）。返回分量列表。"""
    g = {v: list(adj_dir[v]) for v in order}
    rg: dict[str, list[str]] = {v: [] for v in order}
    for u in order:
        for v in g[u]:
            rg[v].append(u)
    seen: set = set()
    post: list[str] = []
    for s in order:
        if s in seen:
            continue
        stack = [(s, 0)]
        seen.add(s)
        while stack:
            u, i = stack.pop()
            if i < len(g[u]):
                stack.append((u, i + 1))
                w = g[u][i]
                if w not in seen:
                    seen.add(w)
                    stack.append((w, 0))
            else:
                post.append(u)
    comp: dict[str, int] = {}
    cid = 0
    for s in reversed(post):
        if s in comp:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            u = stack.pop()
            for w in rg[u]:
                if w not in comp:
                    comp[w] = cid
                    stack.append(w)
        cid += 1
    groups: dict[int, list[str]] = {}
    for v, c in comp.items():
        groups.setdefault(c, []).append(v)
    return [sorted(v) for v in groups.values()]


def laplacian_applicable(adj_dir: dict, order: list[str]) -> dict:
    """**Chung 的有向拉普拉斯到底能不能用。**

    这个函数存在的理由：我第一次直接算，得到 λ₁ = −1.585，
    而对称化后的拉普拉斯不该有负特征值。查过之后原因清楚了：

        那个构造的定理前提是**有向图强连通**（平稳流处处为正、唯一）。
        而本数据的 `relations` 有 **36 个强连通分量、每个节点自成一份**
        —— 也就是一张**有向无环图**。这不是意外：`R3` 本来就禁止成环，
        `by-situation.md:298` 也明说「反向关系写在正文里，不写进 relations
        （否则构成环，被 R3 判错）」。

    前提不成立时，那个对称化对象**根本不是拉普拉斯**（实测 7 个负特征值，
    最小 −2.38），它的谱不能读成连通性谱。

    返回前提是否成立，以及不成立时的证据。
    """
    groups = strongly_connected(adj_dir, order)
    sizes = sorted((len(g) for g in groups), reverse=True)
    return {"strongly_connected": len(groups) == 1,
            "n_components": len(groups), "sizes": sizes,
            "largest": sizes[0] if sizes else 0,
            "reason": ("" if len(groups) == 1 else
                       f"有 {len(groups)} 个强连通分量（最大 {sizes[0]} 个节点），"
                       f"**是有向无环图**；Chung 构造的强连通前提不成立")}


def topological_depth(adj_dir: dict, order: list[str]) -> dict:
    """DAG 上的"层级"：从任一源点出发到该点的**最长路径长度**。

    这是无环结构上对的谱替代品。对 DAG 来说，「第几层」比「第几个特征值」有意义得多。
    若无环假设不成立则拒绝计算（不返回一个假装有意义的值）。
    """
    import functools
    colour: dict[str, int] = {}

    def visit(u: str) -> int:
        if u in colour:
            return colour[u]
        colour[u] = -1                     # 正在访问
        best = 0
        for v in adj_dir[u]:
            r = visit(v)
            if r < 0:
                return -1                  # 回边 → 有环
            best = max(best, r + 1)
        colour[u] = best
        return best

    for v in order:
        if visit(v) < 0:
            return {"ok": False, "why": "有环，不是 DAG"}
    return {"ok": True, "depth": {v: colour[v] for v in order}}


def reach_count(adj_dir: dict, order: list[str]) -> dict:
    """每个节点向下能到达的节点数（**可达性**）。

    ⚠️ 这是**结构量**，不是热度量：它完全由图决定，
    与"多少人说"无关。所以它不违反 `§C7.1 ①`。
    """
    memo: dict[str, set] = {}

    def reach(v: str) -> set:
        if v in memo:
            return memo[v]
        memo[v] = set()
        out: set = set()
        for w in adj_dir[v]:
            out.add(w)
            out |= reach(w)
        memo[v] = out
        return out

    return {v: len(reach(v)) for v in order}


def directed_curvature(adj_dir: dict, dist_undirected: dict) -> list[dict]:
    """有向 OR：惰性测度用**出邻居**，传输代价用底层无向距离。

    复用 `ollivier.wasserstein` —— 它只读 `adj[x]`，所以传有向表就是有向版。
    """
    seen: set[tuple[str, str]] = set()
    rows = []
    for x in sorted(adj_dir):
        for y in sorted(adj_dir[x]):
            key = (x, y) if x <= y else (y, x)
            if key in seen:
                continue
            seen.add(key)
            if x not in dist_undirected or y not in dist_undirected:
                continue
            if y not in dist_undirected[x]:
                continue
            w = O.wasserstein(adj_dir, dist_undirected, x, y)
            d = dist_undirected[x][y]
            rows.append({"edge": key, "k_out_x": len(adj_dir[x]),
                         "k_out_y": len(adj_dir[y]), "w1": w,
                         "kappa": 1.0 - w / d if d else None})
    rows.sort(key=lambda r: r["edge"])
    return rows
