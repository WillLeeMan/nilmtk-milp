"""调用外部 ``nilmtcmilp`` 的构造函数, 生成 TMILP-1bin-project 模型。

模型本体 (变量、约束、目标函数) 完全由 ``nilmtcmilp`` 提供, 本模块只负责:

* 把 NILMTK 的总表读数作为 ``P_Sum`` 传进去 (原实现用的是"所选电器功率之和",
  在真实 NILM 场景下应当换成电表实测总有功功率);
* 统一 Q 分支 (默认只做有功 P, ``Q_Sum`` 传空数组即可关闭无功)。
"""

from __future__ import annotations

import numpy as np

from .._source import import_nilmtcmilp


def build_tmilp_model(dp, mains, S, T: int, test_day_num: int,
                      C_start: float, C_run: float):
    """构造一个窗口的 MILP 模型。

    Parameters
    ----------
    dp
        ``nilmtcmilp.data_processing.EmilpData``, ``partial_fit`` 阶段得到。
    mains : ndarray, shape (T,)
        该窗口的总有功功率 (W), 作为功率平衡的右端项。
    S : array-like
        每个电器的状态数。
    T : int
        窗口采样点数, 必须等于 ``test_day_num * per_day``。
    test_day_num : int
        窗口包含的天数。
    C_start, C_run : float
        目标函数中的启动/运行惩罚系数。
    """
    nilmtcmilp = import_nilmtcmilp()
    mains = np.asarray(mains, dtype=float).ravel()
    if mains.size != T:
        raise ValueError(f"mains 长度 {mains.size} 与 T={T} 不一致")

    model = nilmtcmilp.construct_tmilp_1bin_project(
        dp.P_top, dp.P_bottom,
        np.asarray(dp.Q_top), np.asarray(dp.Q_bottom),
        dp.State_Limit,
        mains,
        np.array([], dtype=float),      # Q_Sum 留空 => 只解有功
        dp.P_up, dp.P_down,
        np.asarray(dp.Q_up), np.asarray(dp.Q_down),
        dp.Transition_model,
        dp.Omega,
        T, np.asarray(S).ravel(), test_day_num,
        dp.probabilities_On, dp.proportion_used_applicance,
        C_start, C_run,
    )
    return model
