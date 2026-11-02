"""nilmtk-milp -- MILP 负荷分解算法接入 NILMTK.

布局参考 nilmtk-contrib::

    nilmtk_milp/
        disaggregate/   # NILMTK Disaggregator 子类
        core/           # 数据预处理、模型构建、功率分解
        solvers/        # HiGHS / CBC / SCIP / CP-SAT 后端

算法本体 (TMILP-1bin-project) 来自外部 ``nilmtcmilp`` 包, 本包只做三件事:

1. 把 NILMTK 的 MeterGroup/DataFrame 数据整理成该模型需要的数组;
2. 用统一的 :class:`~nilmtk_milp.solvers.base.MilpModel` 接口调用不同求解器;
3. 把求解结果还原成 NILMTK 需要的 ``List[DataFrame]``。

适用版本: NILMTK >= 0.4 (新 API: ``partial_fit`` / ``disaggregate_chunk``)。
"""

from .version import __version__

__all__ = ["__version__"]
