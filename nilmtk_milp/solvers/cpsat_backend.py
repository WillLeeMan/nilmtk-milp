"""OR-Tools CP-SAT 后端 (``pip install ortools``)。

CP-SAT 只接受**整数**变量与系数, 而 NILM 模型里的功率/归一化变量是连续的,
因此这里分两类处理:

* 0/1 变量 (ctype='B') 与整数变量 (ctype='I') 直接作为 CP-SAT 的整数变量, 不缩放;
* 连续变量 (ctype='C') 放大 ``scale`` 倍取整 (``X = round(x*scale)``), 返回时再除回来。

约束 ``A x <= b`` 统一放大 ``scale^2`` 倍写成整数:
二进制/整数变量的系数为 ``round(a*scale^2)``, 连续变量为 ``round(a*scale)``。
默认 ``scale=1000`` 即 1 mW 分辨率。
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse

from .base import SolveResult, SolverError, SolverUnavailable, has_quadratic
from .lp import num_vars


def _row_terms(csr: sparse.csr_matrix, i: int):
    start, end = csr.indptr[i], csr.indptr[i + 1]
    return csr.indices[start:end], csr.data[start:end]


def solve(model, gap: float = 1e-2, time_limit: float | None = None,
          scale: int = 1000, ub_cap: float = 1e5, eq_tol: int = 0,
          verbose: bool = False, **options) -> SolveResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover
        raise SolverUnavailable("未安装 ortools。请执行: pip install ortools") from exc

    if has_quadratic(model):
        raise SolverError("CP-SAT 不支持二次目标 (Q/Qc); 请改用 scip 后端。")

    scale = int(scale)
    total = scale * scale
    t0 = time.perf_counter()
    n = num_vars(model)
    f = np.zeros(n) if getattr(model, "f", None) is None else np.asarray(model.f, float).ravel()
    lb = np.full(n, -np.inf) if getattr(model, "lb", None) is None else np.asarray(model.lb, float).ravel()
    ub = np.full(n, np.inf) if getattr(model, "ub", None) is None else np.asarray(model.ub, float).ravel()
    ctype = getattr(model, "ctype", None) or "C" * n

    m = cp_model.CpModel()
    variables = []
    # mult[j]: 把"约束整体放大 total 倍"时, 变量 V_j 对应的系数倍率
    mult = np.zeros(n, dtype=np.int64)
    out_scale = np.ones(n, dtype=float)  # V -> x 的还原系数

    for j in range(n):
        kind = ctype[j]
        if kind == "B":
            variables.append(m.NewBoolVar(f"x{j}"))
            mult[j] = total
            out_scale[j] = 1.0
        elif kind == "I":
            lo = math.ceil(lb[j]) if np.isfinite(lb[j]) else math.ceil(-ub_cap)
            hi = math.floor(ub[j]) if np.isfinite(ub[j]) else math.floor(ub_cap)
            if lo > hi:
                raise SolverError(f"变量 x{j} 的界为空: [{lb[j]}, {ub[j]}]")
            variables.append(m.NewIntVar(int(lo), int(hi), f"x{j}"))
            mult[j] = total
            out_scale[j] = 1.0
        else:
            lo = math.ceil(lb[j] * scale) if np.isfinite(lb[j]) else math.ceil(-ub_cap * scale)
            hi = math.floor(ub[j] * scale) if np.isfinite(ub[j]) else math.floor(ub_cap * scale)
            if lo > hi:
                raise SolverError(f"变量 x{j} 的界在缩放后为空: [{lb[j]}, {ub[j]}]")
            variables.append(m.NewIntVar(int(lo), int(hi), f"x{j}"))
            mult[j] = scale
            out_scale[j] = 1.0 / scale

    def add_rows(mat: sparse.csr_matrix, rhs: np.ndarray, is_eq: bool) -> None:
        mat = sparse.csr_matrix(mat)
        for i in range(mat.shape[0]):
            idx, val = _row_terms(mat, i)
            if len(idx) == 0:
                continue
            terms = [int(round(val[k] * mult[idx[k]])) * variables[idx[k]]
                     for k in range(len(idx))]
            target = int(round(float(rhs[i]) * total))
            expr = sum(terms[1:], terms[0]) if len(terms) > 1 else terms[0]
            if is_eq:
                m.AddLinearConstraint(expr, target - eq_tol, target + eq_tol)
            else:
                m.Add(expr <= target)

    if getattr(model, "Aeq", None) is not None and model.Aeq.shape[0] > 0:
        add_rows(model.Aeq, np.asarray(model.beq, float).ravel(), True)
    if getattr(model, "Aineq", None) is not None and model.Aineq.shape[0] > 0:
        add_rows(model.Aineq, np.asarray(model.bineq, float).ravel(), False)

    obj_terms = [int(round(f[j] * mult[j])) * variables[j]
                 for j in range(n) if f[j] != 0.0]
    if obj_terms:
        m.Minimize(sum(obj_terms[1:], obj_terms[0]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit) if time_limit else 3600.0
    solver.parameters.relative_gap_limit = float(gap)
    solver.parameters.log_search_progress = bool(verbose)
    if options.get("threads"):
        solver.parameters.num_search_workers = int(options["threads"])

    status_code = solver.Solve(m)
    runtime = time.perf_counter() - t0
    status = solver.StatusName(status_code)

    x = None
    obj = float("nan")
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x = np.array([solver.Value(v) * out_scale[j] for j, v in enumerate(variables)],
                     dtype=float)
        obj = float(solver.ObjectiveValue()) / total

    return SolveResult(x=x, objval=obj, status=status, runtime=runtime, solver="cpsat",
                       extra={"objective_scaled": obj * total})
