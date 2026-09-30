"""图的精确连通性结构：桥、2-边连通分量、边不相交路径数。

用于 Phase 4 的 R2 判据（"这一步是不是跨了一条桥"）。

这条路子在 Phase 1 之后是必须走的：Phase 1 已把**曲率符号**证伪
（Forman 在本图上是度数代理，方差占比 1.054），所以"共识区 = 正曲率"
这个定义不可用。而桥是**精确**的：一条边是桥，当且仅当去掉它图就不连通。

    κ 符号   —— 一个连续量，在稀疏图上退化成度数
    桥       —— 一个布尔量，没有可调参数，没有退化

所以 R2 用桥，不用曲率符号。这不是换到好看的为止，是 Phase 1 已经定下的后果。

Tarjan 低链，递归改用显式栈以免将来节点变多时爆栈。
"""

from __future__ import annotations


def bridges(adj: dict) -> set[tuple[str, str]]:
    """所有桥。边用 (小 id, 大 id) 规范化。"""
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    out: set[tuple[str, str]] = set()
    counter = [0]

    for start in sorted(adj):
        if start in disc:
            continue
        # 显式栈：(节点, 父节点, 已展开到第几个邻居)
        stack = [(start, None, 0)]
        disc[start] = low[start] = counter[0]
        counter[0] += 1
        while stack:
            u, parent, i = stack[-1]
            nbrs = sorted(adj[u])
            if i < len(nbrs):
                stack[-1] = (u, parent, i + 1)
                v = nbrs[i]
                if v not in disc:
                    disc[v] = low[v] = counter[0]
                    counter[0] += 1
                    stack.append((v, u, 0))
                elif v != parent and disc[v] < low[u]:
                    low[u] = disc[v]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if low[u] > disc[p]:
                        out.add((p, u) if p <= u else (u, p))
    return out


def edge_connected_components(adj: dict, br: set) -> list[list[str]]:
    """去掉全部桥之后的连通分量 = 2-边连通分量。"""
    adj2 = {x: {y for y in adj[x] if ((x, y) if x <= y else (y, x)) not in br} for x in adj}
    seen: set[str] = set()
    comps: list[list[str]] = []
    for s in sorted(adj2):
        if s in seen:
            continue
        seen.add(s)
        st = [s]
        c = []
        while st:
            x = st.pop()
            c.append(x)
            for y in sorted(adj2[x]):
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        comps.append(sorted(c))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def component_of(comps: list[list[str]]) -> dict:
    return {v: i for i, c in enumerate(comps) for v in c}


def multiplicity(adj: dict, br: set, entry: str, targets: list[str]) -> dict:
    """每个 target 到 entry 的**边不相交路径**够不够两条。

    判据（精确，无参数）：Tarjan 的桥判据等价于
    "去掉所有桥之后，两个点是否落在同一个 2-边连通分量里"。也就是说：

        同一个 2ecc  =>  至少两条边不相交路径  =>  结构给了多重支撑
        不同 2ecc    =>  必须跨桥               =>  结构只给了单条支撑

    ⚠️ 这是等价，不是近似。所以本函数不需要任何"几条路径算够"的阈值。
    """
    comps = edge_connected_components(adj, br)
    cof = component_of(comps)
    return {t: (cof.get(t) == cof.get(entry)) for t in targets}
