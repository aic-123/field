"""Phase 1 第三步：曲率（Forman）。

⚠️ 这一版用 **Forman 曲率**顶替设计稿原本指定的 Ollivier-Ricci。
替代的后果已被量出（`curvature_split.py`：度数项方差占比 1.054），
**Forman 在本图上是度数代理，不是桥探测**。
真正的 OR 在 `ollivier.py` 里补做了，对比见 `curvature_compare.py`。

    F(e) = 4 − d(x) − d(y) + 3·t(e)

    d(v) = v 的度
    t(e) = 含边 e 的三角形个数

它是一个**精确**的离散曲率（不需要近似）。

---
本模块不做的事
--------------

- 不把曲率大小排名次。表格按**边 id** 排，读者自己看。
- 不给曲率设任何"过线"判定。正负零是曲率自己的符号，不是我拍的线。
"""

from __future__ import annotations

import graph as G


def forman(adj: dict, order: list[str]) -> list[dict]:
    """每条无向边一行。边按 (较小端点 id, 较大端点 id) 排序 —— 确定性，且不是名次。"""
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    deg = G.degrees(adj)
    for x in order:
        for y in adj[x]:
            if y not in deg:
                continue
            key = (x, y) if x <= y else (y, x)
            if key in seen:
                continue
            seen.add(key)
            tri = G.triangles_through(adj, x, y)
            f = 4 - deg[x] - deg[y] + 3 * tri
            rows.append({
                "edge": key,
                "dx": deg[x],
                "dy": deg[y],
                "triangles": tri,
                "F": f,
            })
    rows.sort(key=lambda r: r["edge"])
    return rows


def sign_counts(rows: list[dict]) -> dict:
    pos = sum(1 for r in rows if r["F"] > 0)
    neg = sum(1 for r in rows if r["F"] < 0)
    zero = sum(1 for r in rows if r["F"] == 0)
    return {"edges": len(rows), "positive": pos, "negative": neg, "flat": zero,
            "distinct_values": len({r["F"] for r in rows})}
