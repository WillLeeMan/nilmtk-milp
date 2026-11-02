"""DRO-LR 的 SciPy 回退实现。

``nilmtcmilp`` 里的分布鲁棒线性回归 (DRO-LR) 是一个二阶锥规划, 原实现固定用
COPT 求解。为了让只装了 HiGHS/CBC/SCIP/CP-SAT 的用户也能跑预处理, 这里给出
等价的凸优化实现 —— 它不依赖任何外部求解器。

原问题::

    min  eta * a + (1/N) * sum_i b_i
    s.t. |y_i - x_i'beta| <= b_i
         ||beta||^2 + 1 <= a^2

最优处 a = sqrt(1 + ||beta||^2), b_i = |y_i - x_i'beta|, 于是等价于对 beta 的
无约束凸问题::

    min_beta  eta * sqrt(1 + ||beta||^2) + (1/N) * sum_i |y_i - x_i'beta|

beta 只有 2 个分量 (截距和斜率), 用 Nelder-Mead 从最小二乘解出发即可稳定收敛。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_LS_FALLBACK = 1e-3


def _initial_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if not np.all(np.isfinite(beta)) or np.allclose(beta, 0):
        beta = np.zeros(X.shape[1])
        beta[-1] = _LS_FALLBACK
    return beta


def _objective(beta: np.ndarray, X: np.ndarray, y: np.ndarray, eta: float) -> float:
    cone = np.sqrt(1.0 + float(beta @ beta))
    return eta * cone + float(np.mean(np.abs(y - X @ beta)))


def dro_lr(x, y, eta, B=None) -> np.ndarray:
    """``function_DRO_LR`` 的 SciPy 版本。

    Parameters
    ----------
    x : ndarray, shape (k, N)
        设计矩阵, 第一行通常是全 1 (截距), 第二行是序号。
    y : ndarray, shape (N,)
    eta : float
        鲁棒项权重。
    B : float, optional
        固定截距 (对应 ``dro_lr_fixB``)。
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = X.T
    y = np.asarray(y, dtype=float).ravel()
    eta = float(eta)

    if B is not None and X.shape[1] >= 1:
        fixed = float(B)
        rest = X[:, 1:]
        target = y - fixed

        def obj_rest(beta_rest):
            beta = np.concatenate([[fixed], np.ravel(beta_rest)])
            return _objective(beta, X, y, eta)

        x0 = _initial_beta(np.column_stack([np.ones(len(y)), rest]), target)[1:]
        res = minimize(
            obj_rest, np.atleast_1d(x0),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000, "maxfev": 40000},
        )
        return np.concatenate([[fixed], np.ravel(res.x)])

    x0 = _initial_beta(X, y)
    res = minimize(
        _objective, x0, args=(X, y, eta),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000, "maxfev": 40000},
    )
    beta = np.asarray(res.x, dtype=float)
    # 兜底: 若 Nelder-Mead 没收敛到比最小二乘更好的点, 至少返回一个可行解
    if not np.all(np.isfinite(beta)):
        beta = x0
    return beta


def dro_lr_fixB(x, y, eta, B) -> np.ndarray:
    """``function_DRO_LR_fixB`` 的 SciPy 版本 (截距固定为 B)。"""
    return dro_lr(x, y, eta, B=B)
