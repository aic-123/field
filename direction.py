"""方向一致性 consensus：**看方向，不看数量**。

---
它替代的是什么
--------------

原来的 consensus 是数「节点出现在几个视图里」——**那是热度信号**，
`arena §C7.1 ①` 明令禁止（"允许的只有结构性的量"，"投票数/热度/参与量/曝光量"禁止）。

§C7.1 给的替代定义是**跨群共识**：

    上层信号**不是「多少人说」，而是「是否在所有观点群里都成立」**。
    理由：它的判定条件里就要求**同时**满足所有群，
    多数派的规模优势被抵消掉了 —— 它在结构上不可能违反 #5。

本模块把这句话落地：

    每个视图的场是一个向量；把 m 个视图摆成矩阵，
    **第一主成分是共享方向（共识），每条的残差是分歧**。
    看的是方向，不是条数。

---
"多数派免疫"是一条**可测的性质**，不是口号
------------------------------------------

    把其中一个视图**复制 100 遍**，再算共享方向：
    方向应当几乎不变（|cos| ≈ 1）。
    对照组：一个按条数取胜的规则（出现最多的节点），复制之后它会翻。

这一对就是 §C7.1 那句"结构上不可能违反 #5"的可执行形式。
"""

from __future__ import annotations

import math

# 幂迭代的收敛判据：更新量小于它就算停住。
# 取机器精度的量级（1e-15）；这是数值收敛阈值，不是判据线。
CONV_TOL = 1e-15


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v[:]


def shared_direction(fields: list[list[float]], iters: int = 500) -> dict:
    """第一主成分方向。走 Gram 矩阵（m×m，m 是视图数），幂迭代。

    不做中心化：这里的"共识"是**方向上的共同分量**，
    减去均值会把"所有视图都同意的那部分"也减掉，正好减错东西。
    """
    m = len(fields)
    if m == 0:
        return {"ok": False}
    n = len(fields[0])
    gram = [[sum(fields[i][k] * fields[j][k] for k in range(n)) for j in range(m)]
            for i in range(m)]
    v = [1.0 / math.sqrt(m)] * m
    for _ in range(iters):
        nv = [sum(gram[i][j] * v[j] for j in range(m)) for i in range(m)]
        nn = math.sqrt(sum(x * x for x in nv))
        if nn == 0:
            break
        nv = [x / nn for x in nv]
        if all(abs(nv[i] - v[i]) < CONV_TOL for i in range(m)):
            v = nv
            break
        v = nv
    direction = [sum(v[i] * fields[i][k] for i in range(m)) for k in range(n)]
    return {"ok": True, "direction": _normalize(direction), "loadings": v,
            "top": v[0]}


def variance_share(fields: list[list[float]], direction: list[float]) -> float:
    """第一主成分占总能量的比例。高 = 视图之间确实有一个共同方向。"""
    total = sum(sum(x * x for x in f) for f in fields)
    if total == 0:
        return 0.0
    proj = 0.0
    for f in fields:
        p = sum(f[k] * direction[k] for k in range(len(f)))
        proj += p * p
    return proj / total


def residuals(fields: list[list[float]], direction: list[float]) -> list[float]:
    """每条视图在共享方向之外的残差范数。它就是"分歧"。"""
    out = []
    for f in fields:
        p = sum(f[k] * direction[k] for k in range(len(f)))
        r = math.sqrt(sum((f[k] - p * direction[k]) ** 2 for k in range(len(f))))
        out.append(r)
    return out


def mean_consensus(fields: list[list[float]]) -> list[float]:
    """求和/均值方向。**这个是拿来当反例的**：复制一个视图就等于给它加权。"""
    n = len(fields[0])
    s = [sum(f[k] for f in fields) for k in range(n)]
    return _normalize(s)


def agreeing_direction(fields: list[list[float]], iters: int = 4000,
                       eta0: float = 0.05):
    """**所有视图都同意**的方向：max-min。

        d* = argmax_{‖d‖=1}  min_i ⟨f_i, d⟩

    为什么是它，而不是第一主成分：

        §C7.1 ① 要的是「跨群共识」——判据是**是否在每个观点群里都成立**，
        「要求同时满足所有群，单群不成立」。这是**合取**，不是求和。

        第一主成分是求和意义下最优（`Σ‖f_i‖²` 的投影最大），
        复制一个视图就等于把它的权重乘 100 —— **实测 |cos| 掉到 0.40–0.52**。
        min 对重复成员**不敏感**（重复项的值与原件相同，min 不变），
        所以 max-min 才是那句"结构上不可能违反 #5"的正确形式化。

    ⚠️ **先按精确相等去重。** 理由是精确的而不是近似：
    `min` 在一个多重集上等于它在底集上的值。去重之后，
    "复制一个视图 100 遍"与"不复制"是**同一个优化问题**，
    所以结果逐位相同 —— 免疫性由构造保证，不依赖求解器的偶然行为。

    （第一版没去重，实测 |cos| 只有 0.948–0.996。那不是准则的问题，
    是**求解器**的问题：次梯度法的迭代路径会因重复成员而改变。
    准则的不变量 ≠ 求解器的不变量，这个区分必须留着。）

    解法：投影次梯度上升。每步找投影最小的那个视图，
    朝它走一小步再归一化。确定性、无随机数，是局部方法。
    """
    # 精确去重（保序）
    uniq: list[list[float]] = []
    seen: set = set()
    for f in fields:
        key = tuple(f)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    if not uniq:
        return {"ok": False}
    m = len(uniq)
    n = len(uniq[0])
    d = mean_consensus(uniq)
    if not any(d):
        return {"ok": False}
    best_min = None
    for step in range(iters):
        projs = [sum(f[k] * d[k] for k in range(n)) for f in uniq]
        i_star = min(range(m), key=lambda i: projs[i])
        cur_min = projs[i_star]
        if best_min is None or cur_min > best_min:
            best_min = cur_min
        eta = eta0 / math.sqrt(step + 1.0)
        nd = _normalize([d[k] + eta * uniq[i_star][k] for k in range(n)])
        if all(abs(nd[k] - d[k]) < CONV_TOL for k in range(n)):
            d = nd
            break
        d = nd
    projs = [sum(f[k] * d[k] for k in range(n)) for f in uniq]
    return {"ok": True, "direction": d, "min_projection": min(projs),
            "projections": projs, "n_distinct": m, "n_input": len(fields)}


def plurality_consensus(views: list[set]) -> str:
    """对照组：**按条数取胜**的规则（出现最多的那个节点）。

    这不是本模块推荐的做法，是拿来当反例的：它违反 §C7.1 ①，
    而且下面那条性质测试会显示它对"复制一个视图"毫无抵抗力。
    """
    count: dict[str, int] = {}
    for v in views:
        for x in v:
            count[x] = count.get(x, 0) + 1
    if not count:
        return ""
    # 并列按 id 破，确定性
    return max(sorted(count), key=lambda x: count[x])


def cos_sim(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return abs(sum(x * y for x, y in zip(a, b)) / (na * nb))


def popularity_immunity_test(fields: list[list[float]], views: list[set],
                             duplicate_index: int = 0, copies: int = 100) -> dict:
    """多数派免疫：把一个视图复制 `copies` 遍，**三种规则各受多少影响**。

    三种规则：
        求和/均值方向   复制一个视图 = 给它加权     → 预期**不免疫**
        第一主成分      求和意义下最优             → 预期**不免疫**（实测 0.40–0.52）
        max-min 方向    min 对重复成员不敏感        → 预期**免疫**（|cos| = 1）

    实测哪一种免疫，就是"看方向不看数量"这句话能不能成立的分界。
    """
    if not fields or duplicate_index >= len(fields):
        return {"ok": False}
    dup_fields = fields + [fields[duplicate_index]] * copies
    dup_views = views + [views[duplicate_index]] * copies

    base_mean = mean_consensus(fields)
    base_pc = shared_direction(fields)["direction"]
    base_agree = agreeing_direction(fields)["direction"]

    return {"ok": True, "copies": copies, "duplicated": duplicate_index,
            "cos_mean": cos_sim(base_mean, mean_consensus(dup_fields)),
            "cos_pc": cos_sim(base_pc, shared_direction(dup_fields)["direction"]),
            "cos_agree": cos_sim(base_agree,
                                 agreeing_direction(dup_fields)["direction"]),
            "plurality_before": plurality_consensus(views),
            "plurality_after": plurality_consensus(dup_views),
            "plurality_changed": plurality_consensus(views) != plurality_consensus(dup_views)}
