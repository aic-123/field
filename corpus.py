"""语料定位：把「读哪个 rl-scaffold」从**硬编码的父目录**变成一个显式参数。

---
为什么要拆出来
--------------

原来的做法是：

    _ROOT = Path(__file__).resolve().parent.parent   # ← 语料就是我的父目录
    sys.path.insert(0, str(_ROOT / "tools"))
    import find_path as fp
    fp.load_all()                                    # 它自己去读 _ROOT/nodes

于是这个模块**只能住在 rl-scaffold 里面**。那是一个三行的路径耦合，
不是架构约束——但它把"实验层"和"产品仓库"焊在了一起，
代价是产品版本线被实验污染（v0.0.5 就是那么来的）。

现在改成：**语料路径显式化，默认仍指向 `../rl-scaffold`，可用 `FIELD_CORPUS` 覆盖。**

    FIELD_CORPUS=/path/to/rl-scaffold python phase1.py

---
不改上游
--------

`tools/find_path.py:64-65` 把 `NODES = ROOT/"nodes"` 写死，而 `ROOT` 是它自己的祖父目录。
**我们不修改上游**（它是 vendor 进来的逐字节拷贝），
只是在 import 之后把 `fp.NODES` 指到别处 —— `load_all()` 在调用时才读这个全局量
（`find_path.py:202` 是全文件唯一用到它的地方），所以这一步是干净且足够的。

---
哪一半不需要语料
----------------

只有**绑定语料**的那几个模块需要 `attach()`：

    graph.py / checks.py / phase1..phase7d（从节点建图的那些）

**机制层完全不碰语料**，可以脱开语料单独跑：

    spectral · geometry · connectivity · curvature · curvature_split · ollivier
    field · ppr · metrics · conformal · knockout · synthetic
    direction · directed · attack · invariants

所以这个仓库有两半：一半是**可复用的机制**，一半是**绑在一份语料上的实验**。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 默认：与 rl-scaffold 并排（`<父目录>/rl-scaffold`）。
DEFAULT_CORPUS = HERE.parent / "rl-scaffold"

ENV_VAR = "FIELD_CORPUS"

_attached: dict = {}


def corpus_root(explicit: str | None = None) -> Path:
    """语料根目录。优先级：显式参数 > `FIELD_CORPUS` > 默认并排。"""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CORPUS.resolve()


def is_available(explicit: str | None = None) -> bool:
    """语料在不在。不抛异常——给"这个探针需不需要语料"用。"""
    return (corpus_root(explicit) / "nodes").is_dir()


def attach(explicit: str | None = None):
    """挂上语料。返回 `(语料根, find_path 模块)`。

    幂等。语料缺失时**直接退出并说清怎么办**，不留下一个跑一半的状态。
    """
    root = corpus_root(explicit)
    key = str(root)
    if key in _attached:
        return root, _attached[key]

    nodes = root / "nodes"
    if not nodes.is_dir():
        raise SystemExit(
            f"找不到语料：{nodes} 不存在。\n"
            f"用环境变量指过去，例如：\n"
            f"    {ENV_VAR}=/path/to/rl-scaffold python {HERE.name}/phase1.py\n"
            f"（默认找的是 {DEFAULT_CORPUS}）"
        )

    tools = root / "tools"
    if not (tools / "find_path.py").is_file():
        raise SystemExit(f"语料里没有 tools/find_path.py：{tools}。这不是一份 rl-scaffold。")

    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))

    import find_path as fp

    # **不修改上游文件**，只改它读哪个目录。
    fp.NODES = nodes
    _attached[key] = fp
    return root, fp


def tool_module(name: str, explicit: str | None = None):
    """挂上语料并 import 上游 tools/ 里的另一个模块（例如 probe_gaps）。"""
    attach(explicit)
    return __import__(name)
