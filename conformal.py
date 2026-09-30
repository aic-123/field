"""Conformal prediction：把"那条线"换成**校准出来的分位数**。

---
它替掉的是什么
--------------

原来判"接不接受"靠 `TAU = 0.18` 一条手写的线。Phase 3 已经量出这条线
**在结构上不可能分开正负例**：没过线的正例最低 0.138，越界负例最高 0.150。

我上一轮先用了一个无参数的替代（候选少于 2 个就拒绝），方向对，
但它只覆盖了一半负例——它管的是"根本构不出来"，管不了"构出来了但没根据"。

conformal 管的就是后者，而且它给出的门是**从数据里校准出来的**：

    校准集读数 x₁..xₙ 升序，取第 ⌈(n+1)(1−α)⌉ 个当门
    新样本读数低于该门 → 判为"不在分布内"

有限样本保证（在可交换性下）：新样本被误收的概率 ≤ α。
**门不是我拍的，是分位数。**

---
⚠️ 它的前提，以及我们离前提有多远
--------------------------------

保证依赖**可交换性**：校准集与待判样本必须同分布。

我们手上的东西恰好在这一点上不合格：

    112 条 cues     是训练句（它们本身就写在节点里），读数会退化地好
    17 条留出句     只有 9 条是真改写，其余 8 条是 cue 原文

**拿 cues 当校准集、拿改写句当测试集，就是可交换性被破坏**——
校准出来的门会偏松，覆盖保证不成立。

所以本模块分两块，**并把这件事写在报告里，不许含糊**：

    `coverage_experiment`   机制验证：在同分布的可交换数据上，实际覆盖率
                            是否等于名义覆盖率 1−α。这是关于方法性质的检验。
    `apply_with_caveat`     真实数据上的应用：报出校准门与它在 9 条真改写上的
                            行为，并同时报出"校准集与测试集不同分布"这个缺陷。

第一块能过，说明实现是对的；第二块过不了，说明**样本还不够**——
那正是登记待办里的第一条。
"""

from __future__ import annotations

import math


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def conformal_cut(calibration: list[float], alpha: float):
    """返回 (门, 需要的分位下标, 是否可给出保证)。

    下标 = ⌈(n+1)(1−α)⌉。若超出 n，则**这条保证给不出来**
    （样本太少，无法在 1−α 水平上校准），此时返回 None 而不是硬给一个门。
    """
    s = sorted(x for x in calibration if x is not None)
    n = len(s)
    if n == 0:
        return None, None, False
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return None, k, False
    return s[k - 1], k, True


def in_distribution(x: float, cut: float) -> bool:
    """读数低于门即判为"在分布内"（对电导这类**越小越好**的量）。"""
    if x is None or cut is None:
        return False
    return x <= cut


def coverage_experiment(alpha: float = 0.1, n_cal: int = 100, trials: int = 2000,
                        seed: int = 20261002) -> dict:
    """机制验证：同分布数据上的实际覆盖率。

    生成方式完全是人为的（这里不谈语义），只为检验**实现**：
    校准集与测试样本抽自同一个分布，看被判"在分布内"的比例
    是否等于名义的 1−α。
    """
    st = seed

    def draw():
        nonlocal st
        st = _xorshift(st)
        return (st % 100000) / 100000.0     # U[0,1)

    covered = 0
    counted = 0
    for _ in range(trials):
        cal = [draw() for _ in range(n_cal)]
        cut, _k, ok = conformal_cut(cal, alpha)
        if not ok:
            continue
        counted += 1
        x = draw()
        if in_distribution(x, cut):
            covered += 1
    rate = covered / counted if counted else None
    return {"alpha": alpha, "nominal": 1.0 - alpha, "actual": rate,
            "n_cal": n_cal, "trials": counted}


def apply_with_caveat(calibration, test_pos, test_neg, alpha: float = 0.1) -> dict:
    """真实数据上的应用，**连同它的缺陷一起报**。"""
    cut, k, ok = conformal_cut(calibration, alpha)
    out = {"alpha": alpha, "n_cal": len(calibration), "k": k, "cut": cut, "ok": ok}
    if not ok:
        out["why"] = "校准集太小，无法在 1−α 水平上给出分位门"
        return out
    out["pos_accepted"] = sum(1 for x in test_pos if in_distribution(x, cut))
    out["n_pos"] = len(test_pos)
    out["neg_accepted"] = sum(1 for x in test_neg if in_distribution(x, cut))
    out["n_neg"] = len(test_neg)
    return out
