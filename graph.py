"""Phase 1 第一步：把 `nodes/` 建成一张图。

只取结构，不取内容。入图的东西只有四样：

    id          节点身份
    type        节点类型
    relations   指向其他节点的 id 列表
    cues        情境索引（字符串）

`notes` / 正文 / `source` / `evidence_status` / `filled_by` **一律不进图**。
这一步是纪律的落点，手法与 `find_path._match_record()` 完全一致：

    不进内存的东西，下游再也拿不回来。

---
图的两部分，以及一个必须说破的事实
-----------------------------------

    N     = 36 个节点
    C     = 112 条 cues
    E_rel = relations 边（节点 ↔ 节点）
    E_cue = 节点 ↔ 它自己的一条 cue

⚠️ **cue 是叶子**（度为 1，只挂在它所属的那个节点上）。所以 cue 之间不存在
有意义的几何——它们不是"处境空间"，是**节点的标签**。几何只长在 36 个节点上。

这件事必须写进报告，不能让读者自己发现：它决定了"意图 = 方向"这句话
**只对节点空间成立**，也决定了入口匹配（Phase 3）只能是匹配器的事，
不可能是几何的事。

---
本模块不做的事
--------------

- 每条边质量都是 1，不做任何倍率。要改倍率是一次设计变更，不是调参。
- 不推断缺失的关系。缺就是缺。
- 不做文本相似度（`DECISIONS.md` D-07 的立场：相似度是一种距离度量，不是结构量）。
- 不排序。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

import corpus as C  # noqa: E402

CUE_PREFIX = "C"
EDGE_MASS = 1.0


def load_nodes() -> tuple[dict, dict]:
    """读节点。**语料的挂载延迟到这里。**

    为什么必须延迟：本模块同时装着两种东西——

        纯图工具      degrees / triangles_through / subgraph …
        语料绑定      load_nodes / build

    原来在**模块导入时**就 `C.attach()`，于是任何只是想拿到一个纯图工具的
    模块（例如 `curvature.py` 只要 `degrees`）都会被拖去要语料——
    「机制层不碰语料」那句话就当场变成假的。**实测就是这么被自己的测试打掉的。**

    延迟之后：`import graph` 不需要语料；只有 `load_nodes()` 需要，
    而且缺语料时给出的是清楚的指示，不是一句 ImportError。
    """
    _root, fp = C.attach()
    return fp.load_all()


def build(raw_by_id: dict) -> dict:
    """建图。返回一个 dict，含邻接表、cue 归属、悬挂引用。"""
    adj: dict[str, dict[str, float]] = {}
    cue_owner: dict[str, str] = {}
    cue_order: list[str] = []
    dangling: list[tuple[str, str]] = []

    def touch(x: str) -> None:
        adj.setdefault(x, {})

    def link(x: str, y: str) -> None:
        touch(x)
        touch(y)
        adj[x][y] = adj[x].get(y, 0.0) + EDGE_MASS
        adj[y][x] = adj[y].get(x, 0.0) + EDGE_MASS

    node_ids = sorted(raw_by_id)
    for nid in node_ids:
        touch(nid)

    # ① cues：一条 cue 一个叶子。顺序按节点 id + 出现次序，确定性。
    n = 0
    for nid in node_ids:
        for raw_cue in (raw_by_id[nid].get("cues") or []):
            s = str(raw_cue).strip()
            if not s:
                continue
            n += 1
            cid = f"{CUE_PREFIX}{n:04d}"
            cue_owner[cid] = nid
            cue_order.append(cid)
            link(nid, cid)

    # ② relations：数据里是有向的，图里按无向连（方向另存，见 rel_pairs）。
    rel_pairs: list[tuple[str, str]] = []
    for nid in node_ids:
        for raw_rel in (raw_by_id[nid].get("relations") or []):
            r = str(raw_rel).strip()
            if not r:
                continue
            if r not in raw_by_id:
                dangling.append((nid, r))
                continue
            rel_pairs.append((nid, r))
            link(nid, r)

    return {
        "adj": adj,
        "cue_owner": cue_owner,
        "cue_order": cue_order,
        "rel_pairs": rel_pairs,
        "dangling": dangling,
        "node_ids": node_ids,
    }


def subgraph(adj: dict, keep: list[str]) -> dict:
    """只留指定顶点的诱导子图。"""
    keep_set = set(keep)
    out: dict[str, dict[str, float]] = {x: {} for x in keep}
    for x in keep:
        for y, m in adj[x].items():
            if y in keep_set and y != x:
                out[x][y] = m
    return out


def degrees(adj: dict) -> dict:
    return {x: len(nb) for x, nb in adj.items()}


def volume(adj: dict) -> float:
    return sum(sum(nb.values()) for nb in adj.values()) / 2.0


def components(adj: dict) -> list[list[str]]:
    """连通分量。确定性顺序：分量内按 id 排，分量间按首个 id 排。"""
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[str] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])
    return comps


def triangles_through(adj: dict, x: str, y: str) -> int:
    """边 (x,y) 所在三角形的个数。"""
    nx = set(adj[x]) - {y}
    ny = set(adj[y]) - {x}
    return len(nx & ny)


def stats(adj: dict) -> dict:
    deg = degrees(adj)
    comps = components(adj)
    cue_ids = [x for x in adj if x.startswith(CUE_PREFIX) and x[1:].isdigit()]
    node_ids = [x for x in adj if x not in set(cue_ids)]
    cue_deg = sorted({deg[c] for c in cue_ids})
    return {
        "n_nodes": len(node_ids),
        "n_cues": len(cue_ids),
        "n_vertices": len(adj),
        "n_edges": int(sum(deg.values()) / 2),
        "volume": volume(adj),
        "n_components": len(comps),
        "components": comps,
        "cue_degrees": cue_deg,
        "all_cues_are_leaves": cue_deg == [1],
        "degree_hist": _hist(sorted(deg.values())),
    }


def _hist(vals: list[int]) -> dict:
    h: dict[int, int] = {}
    for v in vals:
        h[v] = h.get(v, 0) + 1
    return dict(sorted(h.items()))
