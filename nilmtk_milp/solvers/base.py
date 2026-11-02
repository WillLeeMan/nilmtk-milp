"""求解器后端共用的数据结构和异常。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SolverUnavailable(RuntimeError):
    """后端所需的第三方包未安装。"""


class SolverError(RuntimeError):
    """求解器运行失败 (例如模型不可行且拿不到解)。"""


@dataclass
class SolveResult:
    """与 ``nilmtcmilp.copt_solver.solve`` 返回结构对齐的结果。

    Attributes
    ----------
    x : ndarray or None
        最优解向量 (长度为变量数); 无可行解时为 None。
    objval : float
        目标函数值。
    status : str
        求解器状态字符串 (Optimal / Time limit / Infeasible ...)。
    runtime : float
        求解耗时 (秒)。
    solver : str
        实际使用的后端名称。
    gap : float or None
        相对 MIP gap (若求解器提供)。
    message : str
        附加信息 (日志摘要、错误原因等)。
    """

    x: np.ndarray | None
    objval: float = float("nan")
    status: str = ""
    runtime: float = float("nan")
    solver: str = ""
    gap: float | None = None
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.x is not None


def has_quadratic(model) -> bool:
    """模型是否带二次目标项 ``x'Qx``。"""
    q = getattr(model, "Q", None)
    if q is None:
        q = getattr(model, "H", None)
    if q is None:
        return False
    try:
        return q.shape[0] > 0 and q.nnz > 0
    except AttributeError:
        return False
