"""评价指标：排序指标 + 优超概率（替代小样本 p 值）。

---
为什么要换
----------

Phase 3 量出一个事实：**瓶颈是那条判据线，不是排序**。

    留出集 argmax 正确 17/17；过 TAU 只有 15/17；没过线的那 2 句 argmax 都是第一位。

所以用"接受/拒绝"这种二值做判据，是把一个**已经完美的东西**（排序）
压成一个**丢信息的二值**，然后拿它做统计。这是自己浪费功效。

两处具体的浪费：

1. **二值 vs 位次**：二值只给 9 正例 / 4 负例；位次给每条查询一个数，
   同一批数据的信息量差一个量级。
2. **p 值 vs 优超概率**：9 vs 4 的精确置换，p 的分辨率下限是 1/715 ≈ 0.0014。
   也就是说 p 这个尺度在 n=13 上几乎是哑的——观测到的 p=0.0028 已经是
   "除了一次都不可能出现之外最极端"的那一档，再想区分强弱就没刻度了。

---
本模块给的两个量
----------------

    recall_at_k / 位次分布   排序质量，不经过任何线
    优超概率 AUC             随机取一个正例、一个负例，前者读数更高的概率

AUC 的性质正好对上我们的难处：与阈值无关、不需要分布假设、
在小样本上仍然是连续的（不像 p 的跳变），而且并列按 0.5 计。
"""

from __future__ import annotations


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def pearson_sq(xs: list[float], ys: list[float]) -> float:
    """皮尔逊相关系数的平方。用来量"某个量是不是另一个量的代理"。

    例：OR 曲率对 (d(x)+d(y)) 的 R² —— 高就说明它其实在度数量级上说话。
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return (sxy * sxy) / (sxx * syy)


def position_of(ordered: list[str], target: str) -> int:
    """target 在有序表里的位次（1 起）。不在里面返回 −1。

    这是诊断量，不是产物里的排名轴：它衡量的是**某个排序方案**，
    不是给来源或主张排序（上游 B2 禁的那件事）。
    """
    for i, x in enumerate(ordered, 1):
        if x == target:
            return i
    return -1


def recall_at_k(ordered: list[str], targets: list[str], k: int) -> float:
    if not targets:
        return 0.0
    head = set(ordered[:k])
    return sum(1 for t in targets if t in head) / len(targets)


def reciprocal_position(ordered: list[str], target: str) -> float:
    p = position_of(ordered, target)
    return (1.0 / p) if p > 0 else 0.0


def auc(pos: list[float], neg: list[float]) -> float:
    """优超概率：随机取一个正例、一个负例，正例读数更高的概率。并列按 0.5。

    无参数、无分布假设、与阈值无关。小样本上仍然连续。
    """
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def auc_from_labels(values: list[float], labels: list[int]) -> float:
    """同一个量，输入是 (读数, 标签)。标签 1 = 正例。"""
    pos = [v for v, l in zip(values, labels) if l == 1]
    neg = [v for v, l in zip(values, labels) if l == 0]
    return auc(pos, neg)


def exact_perm_diff(pos: list[float], neg: list[float]):
    """精确置换检验：均值差 + 双侧 p（枚举全部分组，不抽样）。"""
    from itertools import combinations
    pool = pos + neg
    n1 = len(pos)
    if not pos or not neg:
        return None, None, None
    obs = sum(pos) / len(pos) - sum(neg) / len(neg)
    ge = total = 0
    for idx in combinations(range(len(pool)), n1):
        s = set(idx)
        b = [pool[i] for i in range(len(pool)) if i not in s]
        if not b:
            continue
        total += 1
        a = [pool[i] for i in idx]
        if abs(sum(a) / len(a) - sum(b) / len(b)) >= abs(obs):
            ge += 1
    return obs, ((ge + 1) / (total + 1) if total else 1.0), total


def bootstrap_auc_ci(pos: list[float], neg: list[float], alpha: float = 0.05,
                     rounds: int = 2000, seed: int = 20261001):
    """AUC 的 bootstrap 区间。确定性伪随机。

    小样本上必须报区间而不是只报一个点：n=9/4 的 AUC 点估计不稳。
    """
    if not pos or not neg:
        return None, None, None
    point = auc(pos, neg)
    stats = []
    st = seed
    for _ in range(rounds):
        rp, rn = [], []
        for _ in range(len(pos)):
            st = _xorshift(st)
            rp.append(pos[st % len(pos)])
        for _ in range(len(neg)):
            st = _xorshift(st)
            rn.append(neg[st % len(neg)])
        v = auc(rp, rn)
        if v is not None:
            stats.append(v)
    if not stats:
        return point, None, None
    stats.sort()
    lo = stats[max(0, int((alpha / 2) * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return point, lo, hi


def paired_auc_test(values_a: list[float], values_b: list[float], labels: list[int],
                    max_exact: int = 22):
    """**配对**的 AUC 比较：精确符号翻转检验。

    为什么必须配对：两个统计量是在**同一批查询**上算出来的。
    各自做单样本检验等于假装它们是两组独立样本——既丢掉配对信息，
    又高估了有效样本量。这是方法错误，不是保守与否的问题。

    成熟做法有两条（查过之后选后者）：
      · DeLong 检验（1988）—— 配对 ROC 比较的经典非参数检验，但基于渐近正态，
        在 n=13 这种规模上不可靠；有文献报告它"sometimes"出问题而改用置换。
      · **配对置换 / 符号翻转检验** —— 对每条查询独立地交换两个统计量的取值，
        枚举全部 2ⁿ 种。n=13 时 8192 种，可以**精确枚举**，不需要抽样。

    零假设：两个统计量给出的排序质量相同。
    """
    n = len(labels)
    if n != len(values_a) or n != len(values_b):
        raise ValueError("三个列表长度必须一致")
    if n > max_exact:
        raise ValueError(f"n={n} 超过精确枚举上限 {max_exact}（2ⁿ 会爆）")

    def auc_of(vals):
        pos = [v for v, l in zip(vals, labels) if l == 1]
        neg = [v for v, l in zip(vals, labels) if l == 0]
        return auc(pos, neg)

    obs = auc_of(values_a) - auc_of(values_b)
    total = 1 << n
    ge = 0
    for mask in range(total):
        sa, sb = [], []
        for i in range(n):
            if (mask >> i) & 1:
                sa.append(values_b[i])
                sb.append(values_a[i])
            else:
                sa.append(values_a[i])
                sb.append(values_b[i])
        d = auc_of(sa) - auc_of(sb)
        if abs(d) >= abs(obs):
            ge += 1
    return {"auc_a": auc_of(values_a), "auc_b": auc_of(values_b), "diff": obs,
            "p_exact": ge / total, "n_flips": total}


def holm(pvalues: list[float], alpha: float = 0.05):
    """Holm–Bonferroni 多重比较校正。

    为什么要它：本模块会一次比较多个统计量 × 多个参数档，
    单看每一个的 p 就是多重比较。这是成熟且无需分布假设的做法
    （比 Bonferroni 严格更弱、功效更高）。
    返回 [(原 p, 校正后 p, 是否在 alpha 下显著)]，保持输入顺序。
    """
    m = len(pvalues)
    idx = sorted(range(m), key=lambda i: pvalues[i])
    out = [None] * m
    running = 0.0
    for step, i in enumerate(idx):
        adj = min(1.0, (m - step) * pvalues[i])
        running = max(running, adj)
        out[i] = (pvalues[i], running, running < alpha)
    return out


def summarize(pos: list[float], neg: list[float], label: str = "") -> dict:
    """把三件事一次报齐：优超概率 + 区间 + 均值差与精确 p。

    **分开报，不合并成一个数**（上游 R-8：两个指标永不合并）。
    """
    from itertools import combinations
    total = len(list(combinations(range(len(pos) + len(neg)), len(pos)))) if pos and neg else 0
    diff, p, counted = exact_perm_diff(pos, neg)
    a, lo, hi = bootstrap_auc_ci(pos, neg)
    return {
        "label": label,
        "n_pos": len(pos), "n_neg": len(neg),
        "auc": a, "auc_lo": lo, "auc_hi": hi,
        "mean_pos": (sum(pos) / len(pos)) if pos else None,
        "mean_neg": (sum(neg) / len(neg)) if neg else None,
        "diff": diff, "p_exact": p,
        "p_floor": (1.0 / (total + 1)) if total else None,
    }


def fmt(s: dict) -> str:
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"     # noqa: E731
    return (f"{s['label']:<22s} n={s['n_pos']}/{s['n_neg']}  "
            f"AUC {f(s['auc'])} [{f(s['auc_lo'])}, {f(s['auc_hi'])}]  "
            f"均值差 {f(s['diff'])}  p={f(s['p_exact'],4)} (下限 {f(s['p_floor'],4)})")
