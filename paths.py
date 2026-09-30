"""输出路径。**报告不写在根目录。**

为什么要有这个文件：31 份 .md（19 份生成的报告 + 6 份判决 + 6 份基线）原先全部平铺在
仓库根目录，于是根目录看起来像一堆文件而不是一个仓库。
根目录现在只留**代码**（与 `arena` / `nested-traceable-discussion-graph` 同一形状：
模块平铺在根，测试与文档各自归位）与四份门面文档：

    README.md  README.en.md  DECLARATION.md  CHANGELOG.md

所有报告落在 `docs/` 下**同一个目录里**——不分子目录，是为了让判决之间、
判决与报告之间的相对链接保持有效（换了子目录就要改几十处链接）。
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"


def report(name: str) -> Path:
    """报告的落点。目录不在就建。"""
    DOCS.mkdir(parents=True, exist_ok=True)
    return DOCS / name


def baseline(name: str) -> Path:
    """Phase 0 冻结的上游基线原始输出。它与报告放在一起，因为它是同一类证据。"""
    return report(name)
