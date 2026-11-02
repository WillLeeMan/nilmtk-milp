"""求解器后端注册表。

用法::

    from nilmtk_milp.solvers import solve
    result = solve(model, solver="highs", gap=1e-3, time_limit=60)

支持 ``highs`` / ``cbc`` / ``scip`` / ``cpsat`` 四种, 也可以直接用对应模块。
"""

from __future__ import annotations

from .base import SolveResult, SolverError, SolverUnavailable, has_quadratic

_BACKENDS = {
    "highs": "nilmtk_milp.solvers.highs_backend",
    "cbc": "nilmtk_milp.solvers.cbc_backend",
    "scip": "nilmtk_milp.solvers.scip_backend",
    "cpsat": "nilmtk_milp.solvers.cpsat_backend",
    "cp-sat": "nilmtk_milp.solvers.cpsat_backend",
}

ALIASES = {
    "highs": "highs",
    "highspy": "highs",
    "cbc": "cbc",
    "coin": "cbc",
    "scip": "scip",
    "pyscipopt": "scip",
    "cpsat": "cpsat",
    "cp-sat": "cpsat",
    "ortools": "cpsat",
    "cp_sat": "cpsat",
}


def normalize(name: str) -> str:
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(
            f"未知求解器 {name!r}; 可选: {', '.join(sorted(set(ALIASES.values())))}"
        )
    return ALIASES[key]


def solve(model, solver: str = "highs", **options) -> SolveResult:
    """用指定后端求解 ``model``。

    Parameters
    ----------
    model
        任意带 ``f / Aineq / bineq / Aeq / beq / lb / ub / ctype`` 属性的对象,
        通常是 ``nilmtcmilp.milp_model.MilpModel`` 或
        :class:`nilmtk_milp.core.model.MilpModel`。
    solver
        ``highs`` / ``cbc`` / ``scip`` / ``cpsat``。
    **options
        透传给后端的参数 (gap / time_limit / verbose / scale / ub_cap ...)。
    """
    import importlib

    key = normalize(solver)
    module = importlib.import_module(_BACKENDS[key])
    return module.solve(model, **options)


def available_solvers() -> dict[str, bool]:
    """探测各后端依赖是否就绪 (能 import 即为 True)。"""
    import importlib.util

    checks = {
        "highs": "highspy",
        "cbc": "pulp",
        "scip": "pyscipopt",
        "cpsat": "ortools",
    }
    return {name: importlib.util.find_spec(mod) is not None for name, mod in checks.items()}
