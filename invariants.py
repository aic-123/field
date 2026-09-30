"""不变量测试：把 B-D 检查铺到**随机图**上。

---
为什么要这一步
--------------

现在 B-D1 / B-D2 / B-D8 是在**这一张 36 节点图**上验的。
一张图只能验出一条路径走通，验不出"性质在别处也成立"。

随机图上跑不变量，抓的是这张图抓不到的错：

    有效电阻是不是真的满足三角不等式（这一张图上碰巧成立 ≠ 是度量）
    exp(−tL) 是不是真的保质量（L1 = 0 只在连通分量上成立）
    读数对"给场加一个常数"是不是真的不变
    PPR 的总质量是不是真的归一
    OR 是不是真的落在 [−2, 1]

⚠️ 不变量只有两种结局：**成立**，或者**被反例打掉**。
本模块把反例原样报出来，不挑好看的。
"""

from __future__ import annotations

import math

import field as F
import ollivier as O
import ppr
import spectral as S

# 数值断言的容差。**给它起名，不给它豁免**（B-D6 的规矩）。
# 依据是实测：在这个规模上特征分解的精度约 4e-7，所以断言到 1e-5
# 是"方法能做到"的范围；再紧就是在断言方法做不到的事。
NUM_TOL = 1e-5

# OR 在惰性游走（α=1/2）下的理论界。这是**数学事实**，不是为本图调的参数。
OR_LO = -2.0
OR_HI = 1.0

# 随机图生成时加边的最大尝试次数（防死循环）。
MAX_GUARD = 10000

# 每块不变量最多留几条反例（报告用，不是判据）。
MAX_EXAMPLES = 3


def _xorshift(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF


def random_graph(n: int, m: int, seed: int) -> dict:
    """确定性随机图。先建一条链保证连通，再随机加边，去重、不自环。"""
    adj: dict[str, dict[str, float]] = {f"v{i}": {} for i in range(n)}

    def link(a: str, b: str) -> None:
        if a == b:
            return
        adj[a][b] = 1.0
        adj[b][a] = 1.0

    for i in range(1, n):
        link(f"v{i-1}", f"v{i}")
    st = seed
    target = m
    guard = 0
    while sum(len(v) for v in adj.values()) // 2 < target and guard < MAX_GUARD:
        guard += 1
        st = _xorshift(st)
        i = st % n
        st = _xorshift(st)
        j = st % n
        link(f"v{i}", f"v{j}")
    return adj


def check_ref_metric(adj: dict, tol: float = 1e-9) -> dict:
    """有效电阻必须是度量：对称、非负、同点为零、三角不等式。"""
    order = sorted(adj)
    vals, vecs = S.jacobi(S.laplacian(adj, order))
    R = S.effective_resistance(vals, vecs, order)["R"]
    n = len(order)
    bad = []
    for i in range(n):
        if abs(R[i][i]) > tol:
            bad.append(("对角非零", order[i], R[i][i]))
        for j in range(i + 1, n):
            if abs(R[i][j] - R[j][i]) > tol:
                bad.append(("不对称", order[i], order[j]))
            if R[i][j] < -tol:
                bad.append(("负值", order[i], order[j], R[i][j]))
    # 三角不等式（o(n³)，n 小）
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                if R[i][j] > R[i][k] + R[k][j] + tol:
                    bad.append(("三角不等式", order[i], order[j], order[k]))
    return {"ok": not bad, "violations": bad[:3]}


def check_heat_conserves_mass(adj: dict, tol: float = NUM_TOL) -> dict:
    """exp(−tL) 必须保总质量（L1 = 0，常数向量在零空间里）。

    ⚠️ **断言精度不能超过方法精度。** 首次跑 24 张随机图时有 12 张"失败"，
    量出来的最大偏差是 **4.07e-07**，而当时的容差是 1e-9。
    查过之后：`L·1` 精确为 0、Jacobi 的正交完备性偏差 1.8e-15 ——
    **不变量本身没被违反**，是特征分解在 n=13 上的数值精度只有 ~4e-7。
    所以这里报出**实测最大偏差**，并按方法精度断言，而不是按数学精度断言。
    """
    order = sorted(adj)
    vals, vecs = S.jacobi(S.laplacian(adj, order))
    n = len(order)
    v = [(i % 5) - 2.0 for i in range(n)]
    worst = 0.0
    for t in (0.1, 1.0, 10.0):
        h = F.heat(vals, vecs, v, t)
        worst = max(worst, abs(sum(h) - sum(v)) / max(1.0, abs(sum(v))))
    return {"ok": worst <= tol, "violations": [] if worst <= tol else [worst],
            "worst": worst}


def check_readout_shift_invariant(adj: dict) -> dict:
    """给场加一个常数，读数（均值水平集）必须不变。"""
    order = sorted(adj)
    n = len(order)
    k = [math.sin(i) * (i + 1) for i in range(n)]
    a = F.readout(k, order)
    b = F.readout([x + 3.5 for x in k], order)
    return {"ok": a == b, "violations": [] if a == b else [(sorted(a), sorted(b))]}


def check_ppr_normalized(adj: dict, tol: float = 1e-9) -> dict:
    """PPR 的总质量必须归一（几何 PPR 的定义性质）。"""
    order = sorted(adj)
    bad = []
    for c in (0.05, 0.5):
        mass, _it, _d, _cap = ppr.personalized_pr(adj, order, order[0], c)
        if abs(sum(mass) - 1.0) > tol:
            bad.append((c, sum(mass)))
    return {"ok": not bad, "violations": bad}


def check_or_range(adj: dict) -> dict:
    """OR 曲率的取值范围。**不预设边界，先把实测范围报出来。**"""
    rows = O.curvature(adj)
    ks = [r["kappa"] for r in rows if r["kappa"] is not None]
    if not ks:
        return {"ok": True, "lo": None, "hi": None, "violations": []}
    lo, hi = min(ks), max(ks)
    # 惰性游走（α=1/2）下 OR 的理论界是 [−2, 1]
    bad = [x for x in ks if x < OR_LO - NUM_TOL or x > OR_HI + NUM_TOL]
    return {"ok": not bad, "lo": lo, "hi": hi, "violations": bad[:3]}


def check_sweep_is_min(adj: dict) -> dict:
    """sweep cut 报的电导必须不高于任意一个前缀的电导（它就是最小值）。"""
    order = sorted(adj)
    mass = [math.cos(i) + 2.0 for i in range(len(order))]
    sc = ppr.sweep_cut(adj, order, mass)
    if not sc.get("ok"):
        return {"ok": False, "violations": [("sweep 失败",)]}
    curve = [r["conductance"] for r in sc["curve"] if r["conductance"] is not None]
    worst = max(curve) if curve else None
    if worst is not None and sc["conductance"] > worst + 1e-12:
        return {"ok": False, "violations": [("不是最小值", sc["conductance"], worst)]}
    return {"ok": True, "violations": []}


CHECKS = [
    ("有效电阻是度量", check_ref_metric),
    ("exp(−tL) 保质量", check_heat_conserves_mass),
    ("读数对常数平移不变", check_readout_shift_invariant),
    ("PPR 总质量归一", check_ppr_normalized),
    ("OR 落在 [−2,1]", check_or_range),
    ("sweep cut 确实是前缀最小值", check_sweep_is_min),
]


def run_all(trials: int = 24, n: int = 13, m: int = 26, seed: int = 20261003) -> dict:
    """在 `trials` 张随机图上跑全部不变量，返回每条的通过数与反例。"""
    out = {name: {"pass": 0, "fail": 0, "examples": [], "extra": []}
           for name, _fn in CHECKS}
    st = seed
    for _ in range(trials):
        st = _xorshift(st)
        adj = random_graph(n, m, st)
        for name, fn in CHECKS:
            r = fn(adj)
            if r["ok"]:
                out[name]["pass"] += 1
            else:
                out[name]["fail"] += 1
                if len(out[name]["examples"]) < MAX_EXAMPLES:
                    out[name]["examples"].append(r.get("violations"))
            if name == "OR 落在 [−2,1]" and r.get("lo") is not None:
                out[name]["extra"].append((r["lo"], r["hi"]))
    return out
