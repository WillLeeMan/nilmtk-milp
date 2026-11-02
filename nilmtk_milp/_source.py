"""定位外部 ``nilmtcmilp`` 包 (TMILP 模型本体)。

模型构建代码不随本包分发, 而是复用用户已有的 ``nilmtcmilp`` 实现。
查找顺序:

1. 已经 ``import nilmtcmilp`` 能成功 (例如 ``pip install -e`` 过, 或在 PYTHONPATH 中);
2. ``NILMTCMILP_HOME`` 环境变量指向的目录 (包的父目录);
3. ``NILMTCMILP_HOME_DEFAULT`` 常量 (本机默认路径)。

找到后把父目录插入 ``sys.path`` 再导入。本模块不会修改 ``nilmtcmilp`` 的任何文件。
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

_CACHE: ModuleType | None = None


def _candidate_homes(home: str | None) -> list[str]:
    homes = [home, os.environ.get("NILMTCMILP_HOME")]
    out: list[str] = []
    for h in homes:
        if h and h not in out:
            out.append(h)
    return out


def import_nilmtcmilp(home: str | None = None) -> ModuleType:
    """导入 ``nilmtcmilp``, 失败时给出可操作的报错信息。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    try:
        _CACHE = importlib.import_module("nilmtcmilp")
        return _CACHE
    except ModuleNotFoundError as exc:
        if exc.name != "nilmtcmilp":
            raise

    tried = []
    for candidate in _candidate_homes(home):
        tried.append(candidate)
        if not os.path.isdir(os.path.join(candidate, "nilmtcmilp")):
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            _CACHE = importlib.import_module("nilmtcmilp")
        except ModuleNotFoundError:
            continue
        return _CACHE

    raise ModuleNotFoundError(
        "找不到 nilmtcmilp 包 (TMILP 模型本体, https://github.com/WillLeeMan/nilmtcmilp)。"
        "请任选一种方式:\n"
        "  1) pip install \"nilmtcmilp @ git+https://github.com/WillLeeMan/nilmtcmilp.git\"\n"
        "  2) pip install -e <nilmtcmilp 仓库目录>, 或把该目录加入 PYTHONPATH\n"
        "  3) 设置环境变量 NILMTCMILP_HOME=<nilmtcmilp 仓库目录>\n"
        "  4) 构造 TMILP 时传入 nilmtcmilp_home=<nilmtcmilp 仓库目录>\n"
        f"已尝试: {tried}"
    )
