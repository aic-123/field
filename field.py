"""Phase 2 的机器：把"滑行 + 差分"写成可以算的东西。

---
场、意图、滑行
--------------

场是图上的一个实值函数（每个节点一个数）。滑行是它随时间的演化：

    dK/dt = −L K + α·S           K(0) = δ_e

    L   = 组合拉普拉斯（Phase 1 已算）
    e   = 入口（哪片结构被激活）
    S   = 意图（零和的有符号种子向量：正 = 往那边走，负 = 避开）
    α   = 意图强度

这是线性 ODE，有闭式解：

    K(t) = exp(−tL)·δ_e + α·(I − exp(−tL))·L⁺S

三项各有名字：

    exp(−tL)·δ_e              处境自己带来的扩散（无意图时的滑行）
    L⁺S                      意图把场拉到的平衡态
    (I − exp(−tL))            从起点到平衡态之间的插值

弧长就是 t：t→0 几乎只在入口附近，t→∞ 走到意图的平衡态。

---
⚠️ 一个必须先说破的陷阱：可控性会平凡成立
------------------------------------------

`S ↦ K(t)` 是**线性**的，而 `L⁺` 在常数向量的正交补上可逆。所以
"不同意图给出不同场"是**恒真**的，不需要实验。

如果 C2 只测这个，它会通过，而且什么都没证明。

所以 C2 必须落在**读数**上。构造不是场，是从场里读出来的东西：

    construction(e, S, t) = readout(K(t))

读数带选择（只留下某个水平集），选择会丢信息，于是"不同意图给出不同构造"
就不再恒真。**C2 测的是读数，不是场。** 这一点写在这里，防止后面偷换。

---
读数怎么定（无参数）
--------------------

    readout(K) = { i : K(i) > mean(K) }

用场自己的均值做水平，不引入任何常数。均值是场的内在量，不是拍出来的线。
（这是本模块唯一一处"取值比较"，而比较的对象来自数据本身，见 B-D6。）
"""

from __future__ import annotations

import math


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def heat(vals, vecs, v: list[float], t: float) -> list[float]:
    """exp(−tL)·v。"""
    n = len(v)
    coef = [_dot([vecs[i][k] for i in range(n)], v) for k in range(n)]
    out = [0.0] * n
    for k in range(n):
        f = math.exp(-t * vals[k]) * coef[k]
        if f == 0.0:
            continue
        for i in range(n):
            out[i] += f * vecs[i][k]
    return out


def pinv_apply(vals, vecs, s: list[float], zero_eps: float = 1e-9) -> list[float]:
    """L⁺·s（零特征值那一维跳过）。"""
    n = len(s)
    coef = [_dot([vecs[i][k] for i in range(n)], s) for k in range(n)]
    out = [0.0] * n
    for k in range(n):
        if vals[k] <= zero_eps:
            continue
        f = coef[k] / vals[k]
        for i in range(n):
            out[i] += f * vecs[i][k]
    return out


def trajectory(vals, vecs, e_index: int, s: list[float], alpha: float, t: float,
               zero_eps: float = 1e-9) -> list[float]:
    """K(t) = exp(−tL)δ_e + α(I − exp(−tL))L⁺S。"""
    n = len(vals)
    delta = [1.0 if i == e_index else 0.0 for i in range(n)]
    base = heat(vals, vecs, delta, t)
    eq = pinv_apply(vals, vecs, s, zero_eps)
    smooth = heat(vals, vecs, eq, t)
    return [base[i] + alpha * (eq[i] - smooth[i]) for i in range(n)]


def readout(k: list[float], order: list[str]) -> frozenset:
    """读数：场自己的均值以上的那些节点。无参数。"""
    if not k:
        return frozenset()
    mean = sum(k) / len(k)
    return frozenset(order[i] for i, x in enumerate(k) if x > mean)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 1.0 - (len(a & b) / u if u else 0.0)


def l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def normalize_to_unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v[:]
    return [x / n for x in v]
