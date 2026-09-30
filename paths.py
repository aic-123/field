"""输出路径。**报告不写在根目录。**

为什么要有这个文件：31 份 .md（19 份生成的报告 + 6 份判决 + 6 份基线）原先全部平铺在
仓库根目录，于是根目录看起来像一堆文件而不是一个仓库。
根目录现在只留**代码**（与 `arena` / `nested-traceable-discussion-graph` 同一形状：
模块平铺在根，测试与文档各自归位）与四份门面文档：

    README.md  README.en.md  DECLARATION.md  CHANGELOG.md

所有报告落在 `docs/` 下**同一个目录里**——不分子目录，是为了让判决之间、
判决与报告之间的相对链接保持有效（换了子目录就要改几十处链接）。

---
为什么写报告必须走 `write()`
----------------------------

**`Path.write_text` 会把 `\\n` 翻译成 `\\r\\n`**（Windows 上的默认行为，叫 universal newlines）。
本仓库把 `.gitattributes` 钉成 `* text=auto eol=lf`，仓库里存的是 LF；
于是每次重跑探针，工作区的 19 份报告都从 LF 变成 CRLF。

**这不只是"看着脏"**：CI 的 `with-corpus` job 里有一步

    if [ -n "$(git status --porcelain -- docs/)" ]; then ... exit 1; fi

它的用意是抓「改了判据却没重跑报告」。而 CRLF 会让**每一次运行都触发它** ——
真信号（判据改了）就被假信号（行尾变了）淹没了。

所以：**报告只能通过 `write()` 写**，它在 `newline="\\n"` 上钉死。
这个仓库原来有一句想当然的注释说"两边归一化一致，所以重跑不会产生假 diff" ——
**实测证伪了它**：git 只在 `add` 时应用行尾归一化，`status` 看的是工作区原始字节。
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"


def report(name: str) -> Path:
    """报告的落点。目录不在就建。

    ⚠️ 拿到的路径**不要直接 `write_text`** —— 用 `write()`。原因见模块 docstring。
    """
    DOCS.mkdir(parents=True, exist_ok=True)
    return DOCS / name


def write(name: str, text: str) -> Path:
    """写一份报告。**这是唯一的写入口。**

    `newline="\\n"` 不是可选项：少了它，Windows 上重跑一次就把 19 份报告全写成 CRLF，
    CI 那条「报告与代码同步」的检查从此每次假红。
    """
    p = report(name)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def baseline(name: str) -> Path:
    """Phase 0 冻结的上游基线原始输出。它与报告放在一起，因为它是同一类证据。"""
    return report(name)
