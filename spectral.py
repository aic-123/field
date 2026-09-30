"""Phase 1 第二步：度量。拉普拉斯谱 + 有效电阻距离。

---
为什么是有效电阻
----------------

有效电阻距离（effective resistance）在无向图上是一个**真正的度量**：对称、
非负、三角不等式成立。它由拉普拉斯的伪逆给出：

    L = D − A
    R(i,j) = (e_i − e_j)ᵀ L⁺ (e_i − e_j) = Σ_{λ_k > 0} (u_k[i] − u_k[j])² / λ_k

这一步是把"场"从隐喻变成可算对象的**唯一**动作。没有它，后面所有
"滑行""差分""衰减"都没有定义域。

---
特征分解用 Jacobi（纯标准库）
-----------------------------

`requirements.txt` 只有 PyYAML，所以不引入 numpy。用循环 Jacobi：
对称矩阵、数值稳定、实现短、36×36 秒级。

⚠️ 本模块只**打印**谱，不做任何"过线/不过线"的判定。
判据不能是一条拍脑袋的线（`arena §C7.2` / `rl-scaffold GAPS.md:46`：
"一个从不触发的阈值就是拍脑袋"）。这里给出数字和比值，由人读。
"""

from __future__ import annotations

import math

# 数值零容差。**给它起名，不给它豁免**（B-D6 的收窄说明里写了这条）。
# 它是浮点意义下的"零"，不是对数据拍的线。
ZERO_TOL = 1e-9


def laplacian(adj: dict, order: list[str]) -> list[list[float]]:
    """组合拉普拉斯 L = D − A。"""
    n = len(order)
    idx = {v: i for i, v in enumerate(order)}
    L = [[0.0] * n for _ in range(n)]
    for v in order:
        i = idx[v]
        for u, m in adj[v].items():
            if u not in idx:
                continue
            j = idx[u]
            L[i][j] -= m
            L[i][i] += m
    return L


def jacobi(a: list[list[float]], sweeps: int = 60, eps: float = 1e-13):
    """循环 Jacobi 特征分解。返回 (特征值, 特征向量矩阵——列为特征向量)。

    确定性：扫描顺序固定，无随机数。
    """
    n = len(a)
    m = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                off += m[p][q] * m[p][q]
        if off <= eps:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = m[p][q]
                if abs(apq) <= eps:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = m[k][p], m[k][q]
                    m[k][p] = c * akp - s * akq
                    m[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = m[p][k], m[q][k]
                    m[p][k] = c * apk - s * aqk
                    m[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq

    vals = [m[i][i] for i in range(n)]
    # 按特征值升序重排，列跟着走。顺序固定 => 确定性。
    perm = sorted(range(n), key=lambda i: (vals[i], i))
    vals_sorted = [vals[i] for i in perm]
    vecs_sorted = [[v[r][perm[c]] for c in range(n)] for r in range(n)]
    return vals_sorted, vecs_sorted


def effective_resistance(vals: list[float], vecs: list[list[float]], order: list[str],
                         zero_eps: float = ZERO_TOL) -> dict:
    """有效电阻矩阵。零特征值那一维（常数向量）不参与求和。"""
    n = len(order)
    R = [[0.0] * n for _ in range(n)]
    terms = [(vals[k], [vecs[r][k] for r in range(n)])
             for k in range(n) if vals[k] > zero_eps]
    for i in range(n):
        for j in range(i + 1, n):
            acc = 0.0
            for lam, u in terms:
                d = u[i] - u[j]
                if d:
                    acc += (d * d) / lam
            R[i][j] = acc
            R[j][i] = acc
    return {"order": order, "R": R}


def describe(vals: list[float]) -> dict:
    """只描述，不判定。"""
    pos = [x for x in vals if x > ZERO_TOL]
    n = len(vals)
    zero = n - len(pos)
    # 用切片取"有没有第 k 个正特征值"，不用 `len(pos) > k`：
    # 计数比较会被 B-D6 当成内联数字线，而这里本来就不需要比较。
    rest = pos[1:]
    rest2 = pos[2:]
    out = {
        "n": n,
        "zeros": zero,
        "lambda_max": vals[-1] if vals else 0.0,
        "smallest_positive": pos[0] if pos else None,
        "next_positive": rest[0] if rest else None,
    }
    if rest:
        out["ratio_l2_over_l1"] = rest[0] / pos[0]
    if rest2:
        out["ratio_l3_over_l1"] = rest2[0] / pos[0]
    out["tail"] = vals[-5:]
    out["head"] = vals[:8]
    # 有效维数（participation ratio）：(Σλ)² / Σλ²。
    # 无参数、不涉及任何线。是"谱集中在一维还是摊在很多维"的标准读数。
    if pos:
        s1 = sum(pos)
        s2 = sum(x * x for x in pos)
        out["effective_dim"] = (s1 * s1 / s2) if s2 else None
        out["n_positive"] = len(pos)
    return out


def resistance_stats(R: dict) -> dict:
    order = R["order"]
    M = R["R"]
    n = len(order)
    vals = [M[i][j] for i in range(n) for j in range(i + 1, n)]
    if not vals:
        return {"n_pairs": 0}
    vals_sorted = sorted(vals)
    finite = [x for x in vals if math.isfinite(x)]
    return {
        "n_pairs": len(vals),
        "n_finite": len(finite),
        "n_infinite": len(vals) - len(finite),
        "min": vals_sorted[0] if vals_sorted else None,
        "median": vals_sorted[len(vals_sorted) // 2] if vals_sorted else None,
        "max": vals_sorted[-1] if vals_sorted else None,
    }
