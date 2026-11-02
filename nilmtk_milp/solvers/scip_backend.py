"""SCIP 后端 (``pip install pyscipopt``)。

SCIP 通过 ``pyscipopt`` 读取 LP 文件建模, 再调用 API 优化, 避免逐变量建模的开销。
SCIP >= 8 起使用 Apache-2.0 许可证, 可用于开源/商用项目。
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from .base import SolveResult, SolverError, SolverUnavailable, has_quadratic
from .lp import num_vars, write_lp


def solve(model, gap: float = 1e-3, time_limit: float | None = None,
          verbose: bool = False, **options) -> SolveResult:
    try:
        from pyscipopt import Model as ScipModel
    except ImportError as exc:  # pragma: no cover
        raise SolverUnavailable("未安装 pyscipopt。请执行: pip install pyscipopt") from exc

    if has_quadratic(model):
        raise SolverError("SCIP 后端的二次目标尚未实现; 当前模型请用纯线性版本。")

    n = num_vars(model)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nilmtk_milp_scip_") as tmp:
        lp_file = os.path.join(tmp, "model.lp")
        write_lp(model, lp_file)

        m = ScipModel("nilmtk_milp")
        m.hideOutput(not verbose)
        m.readProblem(lp_file)
        m.setParam("limits/gap", float(gap))
        if time_limit is not None:
            m.setParam("limits/time", float(time_limit))
        m.optimize()
        runtime = time.perf_counter() - t0

        status = m.getStatus()
        x = None
        obj = float("nan")
        if m.getNSols() > 0:
            best = m.getBestSol()
            # SCIP 读 LP 时会丢弃"未出现在任何约束/目标中"的变量, 因此按变量名回填
            x = np.zeros(n, dtype=float)
            for var in m.getVars():
                name = var.name
                if not name.startswith("x") or not name[1:].isdigit():
                    raise SolverError(f"无法识别的变量名: {name}")
                idx = int(name[1:])
                if idx >= n:
                    raise SolverError(f"变量索引越界: {name}")
                x[idx] = m.getSolVal(best, var)
            obj = float(m.getObjVal())
        gap_value = None
        try:
            gap_value = float(m.getGap())
        except Exception:
            pass

    return SolveResult(x=x, objval=obj, status=str(status), runtime=runtime,
                       solver="scip", gap=gap_value)
