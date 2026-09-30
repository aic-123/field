"""合成图：给"界面桥"这个主张做**机制验证**（n = 1 → n ≥ 5）。

---
先把界限划清楚：什么合成数据是正当的
------------------------------------

我在评审 DCE v2 时批过合成数据，批的是**这一种**：

    从答案生成输入，再检查系统能否恢复答案。
    （`GAPS.md:135-152` 记的那次：把语料原句写进 cues，然后报"缺口降了"。）

那是循环论证——恢复率由构造保证，不由方法保证。

**机制验证是另一回事：**

    造一张图，**埋进去的位置我事先知道**，然后问：
    这个方法有没有那个性质？

它回答的是"方法有没有这个性质"，不涉及"系统准不准"，
而且答案可能是否——本模块就允许它否。

---
本模块要验的那一条
------------------

Phase 1b 在真实图上量到：

    悬边（一端是叶子的桥） OR 均值 +0.1256    n = 15
    界面桥（两端都有实质结构）OR        −0.5000    **n = 1**
    核内边（非桥）            OR 均值 −0.0966    n = 31

方向对得上（最负的正好落在唯一那条界面桥上），但 **n = 1 是轶事不是证据**。

合成图能把 n 抬起来：把 k 个团连成环，就有 k 条界面桥；
每个团再挂若干叶子，就有大量悬边。
**于是"OR 能不能把界面与悬边分开"可以在几十条边上验，而不是一条。**
"""

from __future__ import annotations


def _link(adj: dict, x: str, y: str) -> None:
    adj.setdefault(x, {})[y] = 1.0
    adj.setdefault(y, {})[x] = 1.0


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def two_cliques(s1: int = 6, s2: int = 6) -> dict:
    """两个团用**恰好一条边**连起来。那条边就是埋进去的界面桥。"""
    adj: dict = {}
    a = [f"a{i}" for i in range(s1)]
    b = [f"b{i}" for i in range(s2)]
    for grp in (a, b):
        for i, x in enumerate(grp):
            adj.setdefault(x, {})
            for y in grp[i + 1:]:
                _link(adj, x, y)
    _link(adj, a[0], b[0])
    return {"adj": adj, "interfaces": [(a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])],
            "pendants": [], "order": sorted(adj), "name": f"两团相连({s1}+{s2})"}


def clique_ring(k: int = 5, size: int = 4) -> dict:
    """k 个团连成一个环：**相邻团之间一条边 = k 条界面桥**。"""
    adj: dict = {}
    groups = []
    for g in range(k):
        grp = [f"c{g}_{i}" for i in range(size)]
        groups.append(grp)
        for i, x in enumerate(grp):
            adj.setdefault(x, {})
            for y in grp[i + 1:]:
                _link(adj, x, y)
    interfaces = []
    for g in range(k):
        x, y = groups[g][0], groups[(g + 1) % k][0]
        _link(adj, x, y)
        interfaces.append((x, y) if x <= y else (y, x))
    return {"adj": adj, "interfaces": sorted(interfaces), "pendants": [],
            "order": sorted(adj), "name": f"{k}团成环(size={size})"}


def clique_ring_with_pendants(k: int = 4, size: int = 4, pendants: int = 3) -> dict:
    """k 个团成环（k 条界面桥），每个团再挂 `pendants` 片叶子（悬边）。

    **这是本模块最要紧的生成器**：它让"界面 vs 悬边"这个区分
    可以在 2k 类不同的边上比，而不是在一条边上。
    """
    base = clique_ring(k, size)
    adj = base["adj"]
    pendant_edges = []
    for g in range(k):
        for p in range(pendants):
            leaf = f"p{g}_{p}"
            adj.setdefault(leaf, {})
            host = f"c{g}_{p % size}"
            _link(adj, leaf, host)
            pendant_edges.append((leaf, host) if leaf <= host else (host, leaf))
    return {"adj": adj, "interfaces": base["interfaces"],
            "pendants": sorted(pendant_edges), "order": sorted(adj),
            "name": f"{k}团成环+{pendants}叶/团"}


def random_tree(n: int = 24, seed: int = 7) -> dict:
    """随机树：**每条边都是桥**，但没有"两端都有实质结构"的界面。

    它是反面对照：如果 OR 只是"见桥就给负"，树上的负值应当与
    界面图上的界面桥一样多。
    """
    adj: dict = {"t0": {}}
    st = seed
    for i in range(1, n):
        st = _xorshift(st)
        parent = f"t{st % i}"
        child = f"t{i}"
        adj.setdefault(child, {})
        _link(adj, child, parent)
    edges = []
    seen = set()
    for x in sorted(adj):
        for y in adj[x]:
            key = (x, y) if x <= y else (y, x)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return {"adj": adj, "interfaces": [], "pendants": [], "order": sorted(adj),
            "all_edges": sorted(edges), "name": f"随机树(n={n})"}


def edge_groups(g: dict) -> dict:
    """把边分成三组：界面桥 / 悬边 / 核内边。

    **靠已知的埋点分，不靠任何算法推断**——这正是机制验证的前提。
    """
    from connectivity import bridges
    br = bridges(g["adj"])
    iface = set(tuple(e) for e in g["interfaces"])
    pend = set(tuple(e) for e in g["pendants"])
    all_edges = set()
    for x in g["adj"]:
        for y in g["adj"][x]:
            all_edges.add((x, y) if x <= y else (y, x))
    other = sorted(e for e in all_edges if e not in iface and e not in pend and e in br)
    core = sorted(e for e in all_edges if e not in iface and e not in pend and e not in br)
    return {"interface": sorted(iface & all_edges), "pendant": sorted(pend & all_edges),
            "other_bridge": other, "core": core, "n_bridges": len(br)}
