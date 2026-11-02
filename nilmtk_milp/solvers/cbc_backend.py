"""CBC 后端 (``pip install pulp[cbc]`` 提供可执行文件, 也可以自己装 CBC)。

工作方式: 写 LP 文件 -> 调用 ``cbc`` 命令行 -> 解析 solution 文件。
CBC 不支持二次目标, 只支持纯线性 MILP。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .base import SolveResult, SolverError, SolverUnavailable, has_quadratic
from .lp import num_vars, write_lp

_SOL_LINE = re.compile(r"^\s*(\d+)\s+(\S+)\s+([-+0-9.eE]+)")


def find_cbc() -> str:
    """定位 cbc 可执行文件。"""
    env = os.environ.get("CBC_PATH")
    if env and os.path.isfile(env):
        return env
    try:
        import pulp

        path = pulp.PULP_CBC_CMD().path
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    found = shutil.which("cbc")
    if found:
        return found
    raise SolverUnavailable(
        "找不到 cbc 可执行文件。请执行 pip install 'pulp[cbc]', "
        "或设置环境变量 CBC_PATH 指向 cbc(.exe)。"
    )


def solve(model, gap: float = 1e-3, time_limit: float | None = None,
          verbose: bool = False, **options) -> SolveResult:
    if has_quadratic(model):
        raise SolverError("CBC 不支持二次目标 (Q/Qc); 请改用 scip 后端。")

    cbc = find_cbc()
    n = num_vars(model)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nilmtk_milp_cbc_") as tmp:
        lp_file = os.path.join(tmp, "model.lp")
        sol_file = os.path.join(tmp, "model.sol")
        write_lp(model, lp_file)

        cmd = [
            cbc, lp_file,
            "-sec", str(float(time_limit) if time_limit else 3600.0),
            "-ratioGap", str(float(gap)),
            "-solve",
            "-solu", sol_file,
            "-quit",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        runtime = time.perf_counter() - t0
        log = (proc.stdout or "") + (proc.stderr or "")
        if verbose:
            print(log)

        x = None
        obj = float("nan")
        if os.path.isfile(sol_file):
            with open(sol_file, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            # CBC 的 solution 文件只列出非零变量, 其余按 0 (或下界) 补
            lb = getattr(model, "lb", None)
            base = np.zeros(n, dtype=float)
            if lb is not None:
                lb = np.asarray(lb, float).ravel()
                finite = np.isfinite(lb)
                base[finite] = lb[finite]
            values = 0
            for line in text.splitlines():
                m = _SOL_LINE.match(line)
                if m:
                    idx = int(m.group(1))
                    if 0 <= idx < n:
                        base[idx] = float(m.group(3))
                        values += 1
                else:
                    mo = re.search(r"objective value\s+([-+0-9.eE]+)", line)
                    if mo and not np.isfinite(obj):
                        obj = float(mo.group(1))
            if values > 0:
                x = base

        status = "unknown"
        lowered = log.lower()
        for token, label in (("optimal", "Optimal"), ("infeasible", "Infeasible"),
                             ("unbounded", "Unbounded"), ("stopped", "Stopped")):
            if token in lowered:
                status = label
                break
        if x is None and status == "Optimal":
            raise SolverError(f"CBC 报告最优但解文件无法解析:\n{log[-2000:]}")

    return SolveResult(x=x, objval=obj, status=status, runtime=runtime, solver="cbc",
                       message=log[-2000:] if x is None else "")
