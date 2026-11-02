"""把 NILMTK 的数据整理成 ``nilmtcmilp`` 需要的数组, 并调用其预处理流程。"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd

from .._source import import_nilmtcmilp

_DRO_PATCHED = False


def _copt_available() -> bool:
    return importlib.util.find_spec("coptpy") is not None


def ensure_dro_backend(preprocessing: str = "auto") -> str:
    """确保 DRO 回归有可用的求解后端, 返回实际使用的后端名。

    * ``copt``: 用 ``nilmtcmilp`` 原生的 COPT 版 (需要 coptpy);
    * ``scipy``: 用本包 :mod:`nilmtk_milp.core.dro` 的凸优化实现 (无外部依赖);
    * ``auto``: 有 coptpy 用 copt, 否则回退 scipy。

    回退方式是给 ``nilmtcmilp`` 的模块动态替换 ``dro_lr`` / ``dro_lr_fixB``;
    只改内存中的函数引用, 不会修改磁盘上的任何文件。
    """
    global _DRO_PATCHED
    choice = str(preprocessing).lower()

    if choice == "copt":
        if not _copt_available():
            raise ImportError(
                "preprocessing='copt' 需要 coptpy, 但当前环境没有安装; 请改用 'auto' 或 'scipy'。"
            )
        return "copt"

    if choice == "auto":
        choice = "copt" if _copt_available() else "scipy"

    if choice == "scipy" and not _DRO_PATCHED:
        from . import dro as _dro

        nilmtcmilp = import_nilmtcmilp()
        dro_module = importlib.import_module("nilmtcmilp.dro_lr")
        data_module = importlib.import_module("nilmtcmilp.data_processing")

        dro_module.dro_lr = _dro.dro_lr
        dro_module.dro_lr_fixB = _dro.dro_lr_fixB
        data_module.dro_lr = _dro.dro_lr
        data_module.dro_lr_fixB = _dro.dro_lr_fixB
        _DRO_PATCHED = True
        warnings.warn(
            "未检测到 coptpy, 预处理阶段改用 SciPy 版 DRO-LR (结果与原 COPT 版在数值上等价)。",
            RuntimeWarning,
            stacklevel=2,
        )
    return choice


def _as_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            raise ValueError("空的 DataFrame")
        series = obj.iloc[:, 0]
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        series = pd.Series(np.asarray(obj).ravel())
    return series


def _concat_frames(objs) -> pd.Series:
    if isinstance(objs, (pd.DataFrame, pd.Series)):
        objs = [objs]
    series = pd.concat([_as_series(o) for o in objs])
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series.astype(float)


def infer_sample_period(index) -> float:
    """从时间索引推断采样周期 (秒)。"""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        diffs = np.diff(index.asi8) / 1e9
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))
    raise ValueError("无法从索引推断采样周期, 请在参数中显式给出 sample_period")


def align_training_data(train_mains, train_appliances, sample_period: float | None = None):
    """把总表和各电器读数对齐成矩阵。

    Returns
    -------
    index : DatetimeIndex
    mains : ndarray, shape (rows,)
    powers : ndarray, shape (rows, n_appliances)
    names : list[str]
    per_day : int
        每个"天"的采样点数 (用于模型的分日惩罚/状态转换统计)。
    """
    mains = _concat_frames(train_mains)
    names: list[str] = []
    app_series: list[pd.Series] = []
    for name, dfs in train_appliances:
        names.append(str(name))
        app_series.append(_concat_frames(dfs).rename(name))

    frame = pd.concat([mains.rename("__mains__"), *app_series], axis=1, join="inner")
    if frame.empty:
        raise ValueError("总表与电器读数的索引没有交集, 无法训练")
    frame = frame.interpolate(limit_direction="both").fillna(0.0)

    if sample_period is None:
        sample_period = infer_sample_period(frame.index)
    per_day = int(round(86400.0 / float(sample_period)))
    if per_day < 1:
        raise ValueError("采样周期过大, 无法构成一天的数据")

    days = len(frame) // per_day
    if days < 1:
        raise ValueError(
            f"训练数据只有 {len(frame)} 个采样点, 不足一天 ({per_day} 点); "
            "请提供更长的训练区间或调大 sample_period。"
        )
    frame = frame.iloc[: days * per_day]

    return (
        frame.index,
        frame["__mains__"].to_numpy(dtype=float),
        frame[names].to_numpy(dtype=float),
        names,
        per_day,
    )


def fit_states(powers: np.ndarray, names: list[str], n_states) -> tuple[np.ndarray, np.ndarray]:
    """对每个电器做 K-means 离散化, 返回 (S, cluster_id)。"""
    nilmtcmilp = import_nilmtcmilp()
    if isinstance(n_states, dict):
        requested = [int(n_states.get(name, n_states.get("*", 2))) for name in names]
    elif np.isscalar(n_states):
        requested = [int(n_states)] * len(names)
    else:
        requested = [int(v) for v in n_states]

    S: list[int] = []
    labels: list[np.ndarray] = []
    for i, name in enumerate(names):
        column = powers[:, i]
        k = max(1, min(requested[i], int(np.unique(column).size)))
        if k == 1:
            S.append(1)
            labels.append(np.ones(len(column), dtype=int))
            continue
        cluster = nilmtcmilp.get_state_num_by_kmean(column, name, -1, k)
        S.append(k)
        labels.append(np.asarray(cluster).ravel())
    return np.asarray(S), np.column_stack(labels)


def run_data_processing(powers, S, cluster_id, names, per_day, always_on,
                        dataset_name: str = "nilmtk", preprocessing: str = "auto"):
    """调用 ``nilmtcmilp.data_processing_emilp`` 得到 :class:`EmilpData`。

    注意: 原函数会用 ``test_data`` 计算 ``P_Sum`` (所选电器功率之和)。真实 NILM
    场景下这个值没有意义 —— 求解时我们会把电表实测总功率直接传给模型,
    因此这里允许把训练数据同时当作 test 传入, 只为满足函数签名。
    """
    ensure_dro_backend(preprocessing)
    nilmtcmilp = import_nilmtcmilp()
    train_data = {dataset_name: {"appliance_P": powers}}
    test_data = {dataset_name: {"appliance_P": powers[:per_day]}}
    flags = np.asarray([1 if always_on.get(name, False) else 0 for name in names], dtype=int)

    return nilmtcmilp.data_processing_emilp(
        train_data, test_data, list(names), ["P"], [dataset_name],
        np.asarray(S).ravel(), np.asarray(cluster_id),
        powers.shape[0], per_day, powers.shape[0] // per_day, flags,
    )
