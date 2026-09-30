"""Phase 1 第三步补：曲率项的精确分解。

Forman 曲率在无向图上是一个**精确的线性分解**，所以不需要回归：

    F(e) = 4 − (d(x) + d(y)) + 3·t(e)
           └──── 常数 ────┘└─ 度数项 ─┘  └ 三角形项 ┘

于是"曲率到底是度数代理还是结构量"可以**算出来**，不用争论：
量出两个项各自的取值范围与方差贡献即可。

这一步是为了回答一个真问题，不是为了给曲率找台阶下：

    如果度数项跨 2..16、三角形项跨 0..9，而两者几乎不相关，
    那么 F 的符号主要由度数决定 —— 那么"负曲率 = 分歧焦点"这条
    在稀疏图上就不成立（负曲率会覆盖大半张图）。

这个结论如果是坏的，就照实写。
"""

from __future__ import annotations


def decompose(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    deg_term = [r["dx"] + r["dy"] for r in rows]
    tri_term = [3 * r["triangles"] for r in rows]
    F = [r["F"] for r in rows]

    def _stats(v: list[float]) -> dict:
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / len(v)
        return {"min": min(v), "max": max(v), "span": max(v) - min(v),
                "mean": m, "var": var}

    # F = 4 − deg + tri，所以 Var(F) = Var(deg) + Var(tri) − 2·Cov(deg, tri)
    md = sum(deg_term) / n
    mt = sum(tri_term) / n
    cov = sum((a - md) * (b - mt) for a, b in zip(deg_term, tri_term)) / n
    var_f = sum((x - sum(F) / n) ** 2 for x in F) / n

    return {
        "n_edges": n,
        "degree_term": _stats(deg_term),
        "triangle_term": _stats(tri_term),
        "F": _stats(F),
        "cov_degree_triangle": cov,
        "var_F": var_f,
        "share_degree": (_stats(deg_term)["var"] / var_f) if var_f else None,
        "share_triangle": (_stats(tri_term)["var"] / var_f) if var_f else None,
    }
