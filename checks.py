"""B-D 系列否证检查。

`arena/spec/Arena-Scaffold-TASK.md:128-149` 的规矩：

    「B 类第 4 项不能只是说法，必须落成一条可执行的否证检查」
    「检查跑不通 = 声明不成立 = 停下提问」

所以本模块的每一条检查，抓的都是**产物里不许出现的东西**。命中即声明不成立。

跑法：

    python field/checks.py          # 退出码 0 = 产物检查全过

---
编号与含义
----------

    B-D1   构造必须确定性：同一 (入口, 意图, α, t) 两次 → 输出逐元素相同
    B-D2   意图必须影响构造：α=0 时不同意图的构造必须全同
    B-D4   产物里不许出现真值量词汇（weight / score / rank / truth / threshold）
    B-D4a  裸禁词必须仍然抓到（防白名单变成后门）
    B-D4b  上游 API 白名单必须恰好两条
    B-D5   判据的输入里不许出现节点正文（notes / _body）
    B-D6   判据逻辑里不许出现与 0/1 之外的内联数字比较
    B-D6a  种入违规必须抓到（防空转）
    B-D6b  位移与 f-string 格式符不许误报
    B-D6c  字符串字面量里的比较不许误报
    B-D8   曲率必须对上三条手算值（这条线最容易骗人，专防）
    B-D16  报告必须走 paths.write（不许直接 write_text：Windows 上会写出 CRLF，
           让 CI 那步「报告与代码同步」每次假红）
    B-D16a 种入一处直接 write_text 必须抓到（防空转）

数据检查（靶子在上游数据，不在本模块产物）：

    B-D7   留出集里不许出现 cue 原文

---
豁免与自我钉住
--------------

`checks.py` 自己**必须**写出这些禁词才能拿它们当检查用，所以它豁免自己。
但豁免是个洞，所以钉住：`test_exemption_is_exactly_these_files()` 断言
豁免集合**恰好**是本文件。将来想多豁免一个，必须改那行，改的时候会被看见。
（手法照抄 `nested/checks.py:58-65`。）
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXEMPT = {"checks.py"}

# 语料位置显式化。见 corpus.py：默认 ../rl-scaffold，可用 FIELD_CORPUS 覆盖。
import corpus as C          # noqa: E402

# 需要语料的三条检查在语料缺失时**显式跳过**，不是假装通过。
# 「跳过」必须看得见 —— 静默变绿和静默变红一样，都是把没验的东西说成验过了。
SKIP_NO_CORPUS = (f"**跳过**：没有语料。设 {C.ENV_VAR}=<rl-scaffold 路径> 后再跑")
#
# 需要语料的是这几条，理由各不同：
#   B-D1 / B-D2   构造与意图，必须在一张真图上才有意义
#   B-D7          靶子就是上游数据本身（HELD_OUT）
#   B-D11/D14/D15 性质本身不需要语料，但**实例**用的是真图
#                 （换成合成图也成立，只是那样钉住的就不是这张图上的数了）

# B-D4：真值量词汇。与 nested `checks.py:73-74` 的 B2 同一张词表。
TRUTH_WORDS = re.compile(r"weight|score|rank|truth|threshold", re.I)

# B-D5：节点正文。判据的输入里出现这些，就是把内容偷偷搬进了判据。
CONTENT_KEYS = re.compile(r"\bnotes\b|_body", re.I)

# B-D6：与数字字面量直接比较。**这是收窄过的一条，理由必须写下来。**
#
# 为什么要收窄（首次跑出 37 处命中，逐条看过之后）：
#
#   16 处根本不是比较，是正则误报：
#       `x ^= (x << 13)` / `x ^= x >> 7`      ← 位移，不是小于/大于
#       f"{e:<10s}" / f"λ{i:<3d}"             ← f-string 格式符
#     这不是放宽判据，是**修判据的 bug**：判据要抓的是"比较"。
#
#   21 处是与 0 的比较，全部是**符号边界或守卫**，不是拍出来的线：
#       曲率/特征向量的符号（0 就是符号的分界，不是我选的数）
#       空集守卫 `if n == 0`、`if k == 0`
#       Jacobi 旋转的符号 `theta >= 0`
#       下标守卫
#     所以把 `0` 与 `1` 排除（单位元/边界，不是被选中的值）。
#
#   剩下唯一一类真候选是**数值零容差** `1e-9`。处理办法不是豁免，是**给它起名**：
#   见 spectral.ZERO_TOL / ollivier.TOL / phase2c.VAR_TOL。判据仍然抓内联字面量，
#   只是不许匿名。
#
# 净效果：B-D6 现在抓的是"与 0/1 之外的内联数字比较" —— 也就是"拍出来的线"。
# 收窄是否成立，由 `test_bd6_fires_on_a_planted_line` 证明：种一组违规进去，
# 它必须全红。**检查不能是空转的。**
NUM_CMP = re.compile(
    r"(?<![:<>=!])"
    r"(?:>=|<=|==|!=|(?<!<)<(?!<)|(?<!>)>(?!>))"
    r"\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b"
)

# 单位元与边界：不是被选中的值，是数学意义上的分界。
NUM_IDENTITY = {"0", "0.0", "1", "1.0"}


def code_lines(path: Path):
    """(行号, 去掉注释与 docstring 之后的源码行)。

    为什么必须去掉：**文档不是产物行为**。docstring 里写不出一个判据来，
    而 B-D4 / B-D6 要抓的是产物**在产生**真值量与常数比较。
    （手法照抄 `nested/checks.py:106-144`。）
    """
    src = path.read_text(encoding="utf-8")
    cut: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cut.setdefault(tok.start[0], tok.start[1])
    doc: set[int] = set()
    fresh = True
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        if tok.type == tokenize.STRING and fresh:
            doc.update(range(tok.start[0], tok.end[0] + 1))
        fresh = tok.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    for n, line in enumerate(src.splitlines(), 1):
        if n in doc:
            continue
        yield n, (line[:cut[n]] if n in cut else line).rstrip()


def code_lines_no_strings(path: Path):
    """同 `code_lines`，但**额外挖掉所有字符串字面量的内容**。

    为什么只给 B-D6 用、不给 B-D4/B-D5 用：

      B-D4/B-D5 是**子串扫描**，上游刻意保留字符串——`{"score": 1}` 里的
      `score` 是产物行为（一个 dict 键），必须抓。所以它们继续看字符串。

      B-D6 抓的是**比较**。写在字符串里的 `"如果 x >= 4 就……"` 是一个字面量，
      不是一个比较。把它当比较是判据的 bug，不是产物的问题。

    手法与 `code_lines` 相同的原则：**文档不是产物行为。**
    """
    src = path.read_text(encoding="utf-8")
    comment_cut: dict[int, int] = {}
    spans: list[tuple[int, int, int, int]] = []
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comment_cut.setdefault(tok.start[0], tok.start[1])
        elif tok.type == tokenize.STRING or (
                fstring_middle is not None and tok.type == fstring_middle):
            # ⚠️ Python 3.12 起，f-string 的**字面量部分**不再是 STRING token，
            # 而是 FSTRING_MIDDLE。只挖 STRING 会漏掉 `f"... p<0.05 ..."` 里的比较
            # —— 那是真缺陷，这一条是被自检抓出来的。
            spans.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))

    lines = src.splitlines()
    for (r0, c0, r1, c1) in spans:
        for r in range(r0, r1 + 1):
            if r - 1 >= len(lines):
                continue
            line = lines[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(line)
            lines[r - 1] = line[:a] + " " * max(0, b - a) + line[b:]

    for n, line in enumerate(lines, 1):
        yield n, (line[:comment_cut[n]] if n in comment_cut else line).rstrip()


def _scan(pattern, only_files: tuple = (), provider=code_lines) -> list[tuple[str, int, str]]:
    hits = []
    targets = [ROOT / f for f in only_files] if only_files else sorted(ROOT.glob("*.py"))
    for path in targets:
        if path.name in EXEMPT or not path.exists():
            continue
        for n, line in provider(path):
            if pattern.search(line):
                hits.append((path.name, n, line.strip()))
    return hits


# ── 静态检查 ──────────────────────────────────────────────────────────

# 上游 API 里名字带禁词的符号。**调用上游不算本模块引入真值量词汇**——
# 名字的处置权在上游，而 probe 需要借用上游的排序函数来**度量**排序好不好。
#
# ⚠️ 这是收窄，所以钉住两条：
#   (a) 白名单必须恰好是这两个（`test_upstream_symbols_are_exactly_these`）；
#   (b) 裸的 `rank` / `score` 仍然照抓（`test_bd4_fires_on_a_bare_symbol`）。
# 「诊断可以读排名，产物不许有排名轴」——本检查守的是后者。
UPSTREAM_SYMBOLS = ("fp.rank(", "find_path.rank(")


def bd4_no_truth_vocabulary():
    hits = []
    for name, n, line in _scan(TRUTH_WORDS):
        stripped = line
        for sym in UPSTREAM_SYMBOLS:
            stripped = stripped.replace(sym, "")
        if TRUTH_WORDS.search(stripped):
            hits.append((name, n, line))
    return hits


def test_upstream_symbols_are_exactly_these() -> tuple[bool, str]:
    want = ("fp.rank(", "find_path.rank(")
    return UPSTREAM_SYMBOLS == want, f"UPSTREAM_SYMBOLS = {UPSTREAM_SYMBOLS}"


def test_bd4_fires_on_a_bare_symbol() -> tuple[bool, str]:
    """裸的禁词必须仍然抓到。否则白名单就成了后门。"""
    planted = "def my_rank(x):\n    return x\n" "s = {'score': 1}\n"
    tmp = ROOT / "_planted_probe.py"
    try:
        tmp.write_text(planted, encoding="utf-8")
        hits = [h for h in bd4_no_truth_vocabulary() if h[0] == "_planted_probe.py"]
    finally:
        if tmp.exists():
            tmp.unlink()
    if len(hits) < 2:
        return False, f"种了 2 条裸禁词，只抓到 {len(hits)} 条"
    return True, "裸的 my_rank / 'score' 都抓到"


def bd5_no_content_in_criteria():
    return _scan(CONTENT_KEYS)


def bd6_no_number_literal_comparison():
    """判据里不许出现与 0/1 之外的内联数字的比较 —— 也就是不许有拍出来的线。

    ⚠️ 两处刻意的边界，都有自检钉着：
      `x > mean` 不被抓 —— 比较对象是**数据的均值**，不是字面量。
      `"如果 x >= 4 就……"` 不被抓 —— 那是字符串字面量，不是比较。

    收窄理由见文件顶部 `NUM_CMP` 的说明。
    """
    hits = []
    for name, n, line in _scan(NUM_CMP, provider=code_lines_no_strings):
        m = NUM_CMP.search(line)
        if m and m.group(1) in NUM_IDENTITY:
            continue
        hits.append((name, n, line))
    return hits


def test_bd6_ignores_comparison_inside_a_string() -> tuple[bool, str]:
    """字符串里的数字比较不许误报 —— **含 f-string 的字面量部分**。

    ⚠️ 这条自检抓到过一个真缺陷：Python 3.12 起 f-string 的字面量
    不再是 STRING token 而是 FSTRING_MIDDLE，原来的挖字符串逻辑漏掉它，
    于是 `f"... p<0.05 ..."` 这种显示文本被当成判据。已修。
    """
    planted = ('msg = "如果 x >= 4 就停"\n'
               'label = f"d>={4} 底"\n'
               'note = f"显著组数（p<0.05，共 {n}）"\n')
    tmp = ROOT / "_planted_probe.py"
    try:
        tmp.write_text(planted, encoding="utf-8")
        hits = [h for h in bd6_no_number_literal_comparison() if h[0] == "_planted_probe.py"]
    finally:
        if tmp.exists():
            tmp.unlink()
    if hits:
        return False, f"字符串里的比较被误报（含 f-string）：{hits}"
    return True, "普通字符串与 f-string 字面量内的比较都不误报"


def test_bd6_fires_on_a_planted_line() -> tuple[bool, str]:
    """种一组真违规进去，B-D6 必须全部抓到。否则这个检查是空转的。

    不只种一条：判据收窄过，所以要证明它对**一整族**违规都还灵敏。
    """
    planted = ("if reading > 0.18:\n    pass\n"
               "if x >= 0.5:\n    pass\n"
               "if y < 1e-3:\n    pass\n"
               "if z == 3:\n    pass\n"
               "if w != 0.75:\n    pass\n")
    tmp = ROOT / "_planted_probe.py"
    try:
        tmp.write_text(planted, encoding="utf-8")
        hits = [h for h in bd6_no_number_literal_comparison() if h[0] == "_planted_probe.py"]
    finally:
        if tmp.exists():
            tmp.unlink()
    if len(hits) < 5:
        return False, f"种了 5 条违规，只抓到 {len(hits)} 条 —— 检查不够灵敏：{hits}"
    return True, "种入 0.18 / 0.5 / 1e-3 / 3 / 0.75 五条全部抓到"


def test_bd6_ignores_bitshift_and_format_spec() -> tuple[bool, str]:
    """位移与 f-string 格式符不许误报。这是收窄要修的那个 bug。"""
    planted = 'x ^= (y << 13)\nx ^= y >> 7\ns = f"{v:<10s}"\n'
    tmp = ROOT / "_planted_probe.py"
    try:
        tmp.write_text(planted, encoding="utf-8")
        hits = [h for h in bd6_no_number_literal_comparison() if h[0] == "_planted_probe.py"]
    finally:
        if tmp.exists():
            tmp.unlink()
    if hits:
        return False, f"位移/格式符被误报：{hits}"
    return True, "位移与格式符均不误报"


def test_exemption_is_exactly_these_files() -> tuple[bool, str]:
    """豁免集合必须恰好是 {checks.py}。多一个就必须改这行。"""
    want = {"checks.py"}
    return EXEMPT == want, f"EXEMPT = {sorted(EXEMPT)}"


def bd8_curvature_matches_hand_computed_values() -> tuple[bool, str]:
    """Ollivier-Ricci 必须在**手算过的小图**上给出正确值。

    为什么这条检查必须存在：曲率这条线上，Phase 1 已经出过一次
    「机制被证伪」，而 Phase 4 又出过一次「看起来成功其实是混淆」。
    一个没有独立验算的曲率实现，是这条线上最容易骗人的东西。

    三条手算（见 `ollivier.self_test` 的推导）：
        路径 x−y−z 的中间边   κ = 1/2
        三角形                κ = 3/4     ← 我第一版把期望值写成 1，被自检抓出
        星形（中心+3 叶）的叶边 κ = 1/3
    """
    import ollivier as O
    res = O.self_test()
    bad = [f"{name}（{why}）" for name, ok, why in res if not ok]
    if bad:
        return False, "手算值对不上：" + "；".join(bad)
    return True, f"{len(res)} 条手算值全对"


# ── 数据卫生检查（靶子在上游数据，不在本模块产物）────────────────────

def _graph():
    import graph
    return graph


def bd7_heldout_must_not_contain_cues():
    """留出集里不许出现 cue 原文。

    `probe_gaps.py:125-136` 自己写下了这条教训：

        「CORPUS 是**我自己写的**……所以 CORPUS 上的分数不能证明覆盖」
        「**不要用「往 cues 里多抄几句」来压这个数**——那是把测试集喂成训练集」

    `HELD_OUT` 的注释也声称是「与 cues 不同的说法」。

    **但实测有 8/17 是 cue 原文。** 所以"15/17 同义改写命中率"这个数被抬高了；
    真正是改写的只有 9 句。

    ⚠️ 本检查的靶子是**上游数据**，不是 field 模块的产物。
    修法有两条，都需要拍板，所以本检查保持命中、不自行放宽：
      (a) 给 HELD_OUT 每行加一个「原文/改写」标记，指标只报改写那一半；
      (b) 承认 15/17 不是同义改写命中率，改个名字。
    两条都不许删数据（上游纪律：语料只加不减）。
    """
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    C.attach()
    import probe_gaps as pg
    raw, _ = _graph().load_nodes()
    hits = []
    for want, q in pg.HELD_OUT:
        cues = [str(c).strip() for c in (raw[want].get("cues") or [])]
        if q in cues:
            hits.append(("probe_gaps.HELD_OUT", 0, f"{want}: {q}"))
    return hits


# ── 可执行检查（跑一遍，不是扫）────────────────────────────────────────

def bd1_determinism() -> tuple[bool, str]:
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    import graph as G
    import spectral as S
    import field as F
    raw, _ = G.load_nodes()
    full = G.build(raw)
    nid = full["node_ids"]
    adj = G.subgraph(full["adj"], nid)
    vals, vecs = S.jacobi(S.laplacian(adj, nid))
    n = len(nid)
    s = [1.0 if x == nid[0] else 0.0 for x in nid]
    m = sum(s) / n
    s = [x - m for x in s]
    a = F.trajectory(vals, vecs, 0, s, 1.0, 2.0)
    b = F.trajectory(vals, vecs, 0, s, 1.0, 2.0)
    ok = a == b and F.readout(a, nid) == F.readout(b, nid)
    return ok, "两次运行的场逐元素相同，读数相同" if ok else "两次运行不一致"


def bd2_intent_must_matter() -> tuple[bool, str]:
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    import graph as G
    import spectral as S
    import field as F
    raw, _ = G.load_nodes()
    full = G.build(raw)
    nid = full["node_ids"]
    adj = G.subgraph(full["adj"], nid)
    vals, vecs = S.jacobi(S.laplacian(adj, nid))
    n = len(nid)
    intents = []
    for a in sorted(nid)[:16]:
        v = [1.0 if x == a else 0.0 for x in nid]
        mm = sum(v) / n
        intents.append([x - mm for x in v])
    sets = {F.readout(F.trajectory(vals, vecs, 0, s, 0.0, 2.0), nid) for s in intents}
    if len(sets) != 1:
        return False, f"α=0 时不同意图竟给出 {len(sets)} 个不同构造 —— 扰动不来自意图"
    sets1 = {F.readout(F.trajectory(vals, vecs, 0, s, 1.0, 2.0), nid) for s in intents}
    if len(sets1) <= 1:
        return False, "α=1 时不同意图给出同一个构造 —— 意图不影响构造"
    return True, f"α=0 → 1 个构造；α=1 → {len(sets1)}/{len(intents)} 个构造"


# ── 方法推荐 ⑤–⑩ 带进来的检查 ────────────────────────────────────────

def bd9_invariants_on_random_graphs() -> tuple[bool, str]:
    """不变量必须在**随机图**上成立，而不是只在这一张 36 节点图上。

    一张图只能验出一条路径走通，验不出"性质在别处也成立"。

    防空转：**临时把 CHECKS 换成一个故意坏的不变量**，
    必须看到 run_all 把它记成失败。否则这个检查是空转的。
    """
    import invariants as INV
    res = INV.run_all(trials=12, n=11, m=22)
    fails = {k: v["fail"] for k, v in res.items() if v["fail"]}
    if fails:
        return False, f"{len(fails)} 条不变量有反例：{list(fails)[:2]}"
    saved = INV.CHECKS
    try:
        INV.CHECKS = [("种入的坏不变量", lambda adj: {"ok": False, "violations": [("种入",)]})]
        r2 = INV.run_all(trials=2, n=8, m=14)
    finally:
        INV.CHECKS = saved
    got = r2["种入的坏不变量"]["fail"]
    if got != 2:
        return False, f"防空转失效：种入的坏不变量只被记了 {got} 次失败（应为 2）"
    return True, f"{len(res)} 条不变量 × 12 张随机图全过；种入坏不变量能被记成失败"


def bd10_synthetic_interfaces_separated() -> tuple[bool, str]:
    """合成图上埋进去的界面桥必须被 OR 与悬边**完全分开**。

    这条守的是 Phase 1b 那个主张。真实图上只有 1 条界面桥（n=1 是轶事），
    合成图把它抬到 n≥4，于是它能被证伪。

    防空转：**把一条悬边错标成界面**，分离度必须掉下来。
    """
    import ollivier as O
    import synthetic as SYN
    import metrics as M
    g = SYN.clique_ring_with_pendants(4, 4, 3)
    rows = O.curvature(g["adj"])
    kappa = {r["edge"]: r["kappa"] for r in rows}
    grp = SYN.edge_groups(g)
    iface = [kappa[e] for e in grp["interface"]]
    pend = [kappa[e] for e in grp["pendant"]]
    if len(iface) < 4 or len(pend) < 4:
        return False, f"分组不对：界面 {len(iface)} 条、悬边 {len(pend)} 条"
    a = M.auc([-x for x in iface], [-x for x in pend])
    if a < 1.0:
        return False, f"界面与悬边没有完全分开：AUC = {a:.3f}"
    mixed = [kappa[e] for e in (grp["interface"] + [grp["pendant"][0]])]
    a2 = M.auc([-x for x in mixed], [-x for x in pend[1:]])
    if a2 >= 1.0:
        return False, f"防空转失效：混进一条悬边后 AUC 仍是 {a2:.3f}"
    return True, (f"界面 {len(iface)} 条 vs 悬边 {len(pend)} 条 AUC = 1.000；"
                  f"混入一条悬边后掉到 {a2:.3f}")


def bd11_sweep_cut_parameter_insensitive() -> tuple[bool, str]:
    """sweep cut 必须在传送参数网格上给出**同一个切**。

    这是"没有位置参数"这句话的可执行形式：如果换一个 c 就换一个构造，
    那 c 就是一个没被承认的位置参数。
    """
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    import graph as G
    import ppr
    raw, _ = G.load_nodes()
    full = G.build(raw)
    nid = full["node_ids"]
    adj = G.subgraph(full["adj"], nid)
    out = {}
    for c in (0.05, 0.15, 0.35):
        mass, _it, _d, _cap = ppr.personalized_pr(adj, nid, "judge-0010", c)
        sc = ppr.sweep_cut(adj, nid, mass)
        out[c] = (sc["k"], round(sc["conductance"], 12), tuple(sc["members"]))
    uniq = set(out.values())
    if len(uniq) != 1:
        return False, f"不同 c 给出不同构造：{ {c: v[:2] for c, v in out.items()} }"
    k, cond, _m = next(iter(uniq))
    return True, f"c=0.05/0.15/0.35 都给出 k={k}、电导={cond}"


def bd12_paired_test_is_self_consistent() -> tuple[bool, str]:
    """配对检验必须自洽：**相同输入给差 0、p = 1；完全分开给小的 p**。

    这条守的是"检验本身是对的"。配对检验是 C5 那条结论被推翻的直接依据，
    它自己出错的话，那次推翻也就无效。
    """
    import metrics as M
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    same = [9, 8, 7, 6, 5, 4, 3, 2]
    r0 = M.paired_auc_test(same, list(same), labels)
    if r0["diff"] != 0.0 or r0["p_exact"] != 1.0:
        return False, f"相同输入竟给出 差={r0['diff']} p={r0['p_exact']}（应为 0 与 1）"
    a = [7, 6, 5, 4, 3, 2, 1, 0]
    b = [3, 2, 1, 0, 7, 6, 5, 4]        # 在 b 里正例全低于负例 → AUC(b)=0
    r1 = M.paired_auc_test(a, b, labels)
    if r1["auc_a"] != 1.0 or r1["auc_b"] != 0.0:
        return False, f"构造的完全分开样本不对：AUC_a={r1['auc_a']} AUC_b={r1['auc_b']}"
    if r1["diff"] != 1.0:
        return False, f"差应为 1.0，实测 {r1['diff']}"
    if r1["p_exact"] > 0.05:
        return False, f"完全分开竟给出 p={r1['p_exact']}（不该这么大）"
    return True, (f"相同输入 差=0 p=1；完全分开 差=1.0 p={r1['p_exact']:.4f}"
                  f"（n=8 枚举下界 1/70≈0.0143）")


def bd13_conformal_coverage_holds() -> tuple[bool, str]:
    """conformal 必须在**同分布数据**上兑现名义覆盖率，且**样本不足时拒绝给门**。

    这是推荐④的机制验证被搬进检查。覆盖保证是它可以取代手写判据线的全部理由；
    如果覆盖率不兑现，它只是一条形式不同的线。
    """
    import conformal as CF
    out = []
    for alpha in (0.05, 0.10, 0.20):
        r = CF.coverage_experiment(alpha=alpha, n_cal=100, trials=1500)
        gap = abs(r["actual"] - r["nominal"])
        out.append((alpha, r["nominal"], r["actual"], gap))
        if gap > 0.02:
            return False, (f"α={alpha} 名义 {r['nominal']:.2f} 实测 {r['actual']:.4f}"
                           f"，偏差 {gap:.4f} 超过 0.02")
    _cut, _k, ok = CF.conformal_cut([i / 100 for i in range(5)], 0.10)
    if ok:
        return False, "校准集只有 5 条却给出了分位门 —— 它在给一个给不出的保证"
    return True, ("、".join(f"α={a} 名义 {n:.2f}→实测 {x:.4f}" for a, n, x, _g in out)
                  + "；n_cal=5 时拒绝给门")


def bd14_consensus_is_popularity_immune() -> tuple[bool, str]:
    """方向一致性必须对"复制一个视图"**精确免疫**，而求和与主成分不免疫。

    这是推荐⑦的核心性质。第一版用第一主成分实现，实测 |cos| 只有 0.40–0.52
    ——**被这条性质打掉**，换成 max-min + 先去重才成立。
    所以这条检查有双向作用：既要 max-min 免疫，也要**证明不免疫的那种会被抓**。
    """
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    import graph as G
    import spectral as S
    import field as F
    import direction as D
    import ppr
    raw, _ = G.load_nodes()
    full = G.build(raw)
    nid = full["node_ids"]
    adj = G.subgraph(full["adj"], nid)
    vals, vecs = S.jacobi(S.laplacian(adj, nid))
    didx = {v: i for i, v in enumerate(nid)}
    seeds = ["con-0006", "judge-0009", "case-0003", "stance-0003"]
    fields = [F.heat(vals, vecs, [1.0 if i == didx[s] else 0.0 for i in range(len(nid))], 2.0)
              for s in seeds]
    views = []
    for s in seeds:
        pr, _i, _d, _c = ppr.personalized_pr(adj, nid, s, 0.15)
        views.append(set(sorted(nid, key=lambda v: (-pr[didx[v]], v))[:8]))
    worst_agree, worst_pc = 1.0, 1.0
    for dup in range(len(seeds)):
        r = D.popularity_immunity_test(fields, views, duplicate_index=dup, copies=100)
        worst_agree = min(worst_agree, r["cos_agree"])
        worst_pc = min(worst_pc, r["cos_pc"])
    if worst_agree < 1.0:
        return False, f"max-min 不免疫：最差 |cos| = {worst_agree:.6f}（应为 1.0）"
    if worst_pc >= 1.0:
        return False, (f"防空转失效：第一主成分竟也免疫（最差 |cos| = {worst_pc:.6f}），"
                       f"那这条检查区分不出两种规则")
    return True, (f"max-min 精确免疫（最差 |cos| = {worst_agree:.6f}）；"
                  f"第一主成分不免疫（最差 {worst_pc:.4f}）")


def bd15_knockout_does_not_touch_the_graph() -> tuple[bool, str]:
    """谱模式敲除**不许动图** —— 这是它替代删节点损伤实验的全部理由。

    防空转：敲除必须真的改变场；如果它是空操作，那"图没变"就没有意义。
    """
    if not C.is_available():
        return None, SKIP_NO_CORPUS
    import graph as G
    import spectral as S
    import field as F
    import knockout as KO
    raw, _ = G.load_nodes()
    full = G.build(raw)
    nid = full["node_ids"]
    adj = G.subgraph(full["adj"], nid)
    vals, vecs = S.jacobi(S.laplacian(adj, nid))
    n = len(nid)
    before = repr(sorted((k, sorted(v.items())) for k, v in adj.items()))
    fv = F.heat(vals, vecs, [1.0 if i == 0 else 0.0 for i in range(n)], 2.0)
    changed = 0
    for k in range(n):
        after_vec = KO.knockout(fv, vecs, k)
        if after_vec != fv:
            changed += 1
        KO.construction_change(adj, nid, fv, after_vec)
    now = repr(sorted((k, sorted(v.items())) for k, v in adj.items()))
    if before != now:
        return False, "敲除过程改了图 —— 混淆又回来了"
    if changed == 0:
        return False, "敲除没有改变场 —— 空操作"
    return True, f"跑遍 {n} 个模式，邻接表逐字节不变；其中 {changed} 个模式确实改变了场"


def bd16_reports_go_through_one_writer() -> tuple[bool, str]:
    """报告必须走 `paths.write`，不许直接 `write_text`。

    为什么这条是硬的不只是风格：`Path.write_text` 在 Windows 上把 `\\n` 写成 `\\r\\n`，
    而 `.gitattributes` 钉的是 LF。于是每次重跑探针，工作区里的报告全部变脏，
    CI 那步「报告与代码同步」就**每次都假红** —— 真信号（判据改了没重跑）
    被假信号（行尾变了）淹没。实测过一次：跑 16 个探针 → 19 份报告全变。

    `paths.write` 在 `newline="\\n"` 上钉死，所以这条检查就是让那个修复**不会退化**。
    """
    offenders = []
    # 要覆盖**所有会写报告的模块**：16 个探针 + curvature_compare，以及种入的临时探针。
    # 早先只 glob `phase*.py`，于是防空转自检种进去的 `_tmp_probe_bd16.py` 扫不到
    # —— 自检当场把这条抓了出来。范围要按"谁可能写报告"定，不是按文件名前缀定。
    cands = [p for p in sorted(ROOT.glob("*.py"))
             if p.name.startswith("phase") or p.name == "curvature_compare.py"
             or p.name.startswith("_tmp_probe")]
    for p in cands:
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "write_text" in line and "paths.write(" not in line:
                offenders.append(f"{p.name}:{i}")
    if offenders:
        return False, f"{len(offenders)} 处绕过了 paths.write：{offenders[:3]}"
    if not any("paths.write(" in q.read_text(encoding="utf-8")
               for q in ROOT.glob("phase*.py")):
        return False, "一个探针都没用 paths.write —— 检查是不是空转了"
    return True, "全部报告都走 paths.write（newline 钉死 LF）"


def test_bd16_fires_on_a_direct_write() -> tuple[bool, str]:
    """防空转：种一个直接 `write_text` 的探针进去，这条必须抓到。"""
    tmp = ROOT / "_tmp_probe_bd16.py"
    try:
        tmp.write_text(
            'import paths\n'
            'paths.report("x.md").write_text("a\\n", encoding="utf-8")\n',
            encoding="utf-8", newline="\n")
        ok, why = bd16_reports_go_through_one_writer()
        if ok is not False:
            return False, f"种入的直接 write_text 没被抓到（返回 {ok!r}）"
        if "_tmp_probe_bd16.py" not in why:
            return False, f"抓到了但没指名道姓：{why}"
    finally:
        if tmp.exists():
            tmp.unlink()
    return True, "种入的直接 write_text 被抓到，且指名了文件"


CHECKS = [
    ("B-D1", "构造必须确定性", bd1_determinism),
    ("B-D2", "意图必须影响构造", bd2_intent_must_matter),
    ("B-D4", "产物里无真值量词汇", bd4_no_truth_vocabulary),
    ("B-D4a", "裸禁词必须抓到（防后门）", test_bd4_fires_on_a_bare_symbol),
    ("B-D4b", "上游 API 白名单恰好两条", test_upstream_symbols_are_exactly_these),
    ("B-D5", "判据输入里无节点正文", bd5_no_content_in_criteria),
    ("B-D6", "判据里无内联数字线", bd6_no_number_literal_comparison),
    ("B-D6a", "种入 0.18 必须抓到（防空转）", test_bd6_fires_on_a_planted_line),
    ("B-D6b", "位移/格式符不许误报", test_bd6_ignores_bitshift_and_format_spec),
    ("B-D6c", "字符串里的比较不许误报", test_bd6_ignores_comparison_inside_a_string),
    ("B-D8", "曲率必须对上三条手算值", bd8_curvature_matches_hand_computed_values),
    ("B-D9", "不变量须在随机图上成立", bd9_invariants_on_random_graphs),
    ("B-D10", "合成图界面必须被分开", bd10_synthetic_interfaces_separated),
    ("B-D11", "sweep cut 对传送参数不敏感", bd11_sweep_cut_parameter_insensitive),
    ("B-D12", "配对检验必须自洽", bd12_paired_test_is_self_consistent),
    ("B-D13", "conformal 覆盖必须兑现", bd13_conformal_coverage_holds),
    ("B-D14", "共识须对复制精确免疫", bd14_consensus_is_popularity_immune),
    ("B-D15", "敲除不许动图", bd15_knockout_does_not_touch_the_graph),
    ("B-D16", "报告须走唯一写入口（行尾钉 LF）", bd16_reports_go_through_one_writer),
    ("B-D16a", "直写 write_text 必须抓到（防空转）", test_bd16_fires_on_a_direct_write),
]

# 靶子在上游数据、不在本模块产物的检查。分开报，不混进产物的绿/红。
DATA_CHECKS = [
    ("B-D7", "留出集里不许有 cue 原文", bd7_heldout_must_not_contain_cues),
]


def _run(name, desc, fn):
    """跑一条检查。返回 True / False / None（None = 跳过，不计入绿也不计入红）。"""
    out = fn()
    if isinstance(out, tuple):
        ok, why = out
    else:
        ok, why = (not out), (f"{len(out)} 处命中" + (f"：{out[:2]}" if out else ""))
    mark = "跳过" if ok is None else ("过" if ok else "**命中**")
    print(f"  {name:<6s} {desc:<26s} {mark}   {why}")
    return ok


def main(argv=None) -> int:
    """跑全部检查。

    `--product-only`：退出码只反映**本仓库产物**的检查，不反映上游数据检查。
    为什么需要它：`B-D7` 的靶子是**语料**（`HELD_OUT` 里混了 cue 原文），
    不是本仓库的产物。拿它当本仓库的门禁，等于让上游的欠账卡住这里的 CI——
    而那一头只有上游能改。所以 CI 用这个开关：**报告照印，退出码不背它。**

    `--expect-skipped N`：断言**恰好跳过 N 条**，不符就退出码 1。
    为什么需要它：跳过态的退出码也是 0，所以"某条检查悄悄退化成永远跳过"
    （例如某个 `is_available()` 判错、或语料换了个目录布局）**不会被任何东西发现**。
    把期望值写成参数，这条不变式才从注释变成可执行的东西。
    """
    import sys as _sys
    args = list(_sys.argv[1:] if argv is None else argv)
    product_only = "--product-only" in args

    expect_skipped = None
    if "--expect-skipped" in args:
        i = args.index("--expect-skipped")
        if i + 1 >= len(args) or not args[i + 1].lstrip("-").isdigit():
            print("⚠️ `--expect-skipped` 后面要跟一个整数。")
            return 2
        expect_skipped = int(args[i + 1])

    print("B-D 否证检查 —— 抓的是**不许出现**的东西。\n")
    if product_only:
        print("（--product-only：退出码只看本仓库产物；上游数据检查照印但不背）\n")
    print("── 本模块产物 ──")
    bad = 0
    skipped = []
    for name, desc, fn in CHECKS:
        r = _run(name, desc, fn)
        if r is None:
            skipped.append(name)
        elif not r:
            bad += 1
    r = _run("----", "豁免集合自我钉住", test_exemption_is_exactly_these_files)
    if r is None:
        skipped.append("----")
    elif not r:
        bad += 1

    print("\n── 上游数据 ──")
    data_bad = 0
    for name, desc, fn in DATA_CHECKS:
        r = _run(name, desc, fn)
        if r is None:
            skipped.append(name)
        elif not r:
            data_bad += 1

    total = len(CHECKS) + 1 + len(DATA_CHECKS)
    print(f"\n共 {total} 条：{total - bad - data_bad - len(skipped)} 条过、"
          f"{bad + data_bad} 条命中、{len(skipped)} 条跳过")
    if skipped:
        print(f"⚠️ 跳过的是 {'、'.join(skipped)} —— **跳过不等于通过**，"
              f"它们什么都没验。要跑全，设 {C.ENV_VAR}=<rl-scaffold 路径>。")
    if data_bad:
        print("⚠️ 数据命中的靶子在上游数据，修法需拍板，不自行放宽、不删数据（见函数注释）。")
        if product_only:
            print("   本轮带 --product-only，所以它不进退出码 —— 但它**仍然是红的**。")

    skip_mismatch = expect_skipped is not None and len(skipped) != expect_skipped
    if skip_mismatch:
        print(f"⚠️ 跳过数不符：期望 {expect_skipped} 条，实际 {len(skipped)} 条"
              f"（{'、'.join(skipped) if skipped else '无'}）。")
        print("   这防的是**某条检查悄悄退化成永远跳过** —— 它退出码照旧是 0，"
              "没有这一条就没人会发现。")

    fail = bool(bad or (data_bad and not product_only) or skip_mismatch)
    print(f"退出码 {1 if fail else 0}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
