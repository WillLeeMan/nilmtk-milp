"""HiGHS 后端 (``pip install highspy``)。

HiGHS 是目前综合性能最好的开源 MIP 求解器之一，纯 C++ 内核加官方 Python 绑定，
在 Windows 上开箱即用。后端直接传稀疏矩阵，不经过文件系统。
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from .base import SolveResult, SolverError, SolverUnavailable, has_quadratic
from .lp import num_vars


def _status_name(highspy, status) -> str:
    try:
        return highspy.modelStatusToString(status)
    except Exception:  # pragma: no cover - 不同版本 API 差异
        return str(status)


def solve(model, gap: float = 1e-3, time_limit: float | None = None,
          verbose: bool = False, **options) -> SolveResult:
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover
        raise SolverUnavailable("未安装 highspy。请执行: pip install highspy") from exc

    if has_quadratic(model):
        raise SolverError("HiGHS 后端尚未实现二次目标 (Q/Qc); 请改用 scip 后端。")

    t0 = time.perf_counter()
    n = num_vars(model)
    f = np.zeros(n) if getattr(model, "f", None) is None else np.asarray(model.f, float).ravel()
    lb = np.full(n, -np.inf) if getattr(model, "lb", None) is None else np.asarray(model.lb, float).ravel()
    ub = np.full(n, np.inf) if getattr(model, "ub", None) is None else np.asarray(model.ub, float).ravel()
    ctype = getattr(model, "ctype", None) or "C" * n

    blocks = []
    row_lower: list[float] = []
    row_upper: list[float] = []
    if getattr(model, "Aeq", None) is not None and model.Aeq.shape[0] > 0:
        blocks.append(sparse.csr_matrix(model.Aeq))
        beq = np.asarray(model.beq, float).ravel()
        row_lower.extend(beq.tolist())
        row_upper.extend(beq.tolist())
    if getattr(model, "Aineq", None) is not None and model.Aineq.shape[0] > 0:
        blocks.append(sparse.csr_matrix(model.Aineq))
        bineq = np.asarray(model.bineq, float).ravel()
        row_lower.extend([-np.inf] * len(bineq))
        row_upper.extend(bineq.tolist())
    mat = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, n))
    csc = mat.tocsc()

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = mat.shape[0]
    lp.col_cost_ = f
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.asarray(row_lower, float)
    lp.row_upper_ = np.asarray(row_upper, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data.astype(float)
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if c in ("B", "I") else highspy.HighsVarType.kContinuous
        for c in ctype
    ]

    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    h.passModel(lp)
    h.setOptionValue("mip_rel_gap", float(gap))
    h.setOptionValue("mip_abs_gap", float(options.get("abs_gap", 0.0)))
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    for key, value in options.items():
        if key in ("abs_gap", "threads", "seed", "random_seed"):
            continue
        try:
            h.setOptionValue(key, value)
        except Exception:
            pass

    h.run()
    runtime = time.perf_counter() - t0
    status = _status_name(highspy, h.getModelStatus())

    sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=float) if len(sol.col_value) == n else None
    if x is not None and not np.all(np.isfinite(x)):
        x = None
    info = h.getInfo()
    obj = float(info.objective_function_value)
    if not np.isfinite(obj):
        obj = float("nan")
    mip_gap = float(getattr(info, "mip_gap", float("nan")))

    return SolveResult(
        x=x,
        objval=obj,
        status=status,
        runtime=runtime,
        solver="highs",
        gap=mip_gap if np.isfinite(mip_gap) else None,
        extra={"nodes": int(getattr(info, "mip_node_count", -1))},
    )
