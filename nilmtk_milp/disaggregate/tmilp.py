"""TMILP-1bin-project 的 NILMTK 适配器。

模型本体来自 ``nilmtcmilp`` (MATLAB ``function_construction_TMILP_1bin_project.m``
的 Python 移植), 本类只负责:

* ``partial_fit``: 用 NILMTK 的电器级真值学习状态、功率上下界、最短运行时间、
  状态转移矩阵等参数;
* ``disaggregate_chunk``: 按窗口构造 MILP 并调用所选求解器, 输出各电器功率;
* ``save_model`` / ``load_model``: 参数持久化。
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from nilmtk.disaggregate import Disaggregator

from .._source import import_nilmtcmilp
from ..core.build import build_tmilp_model
from ..core.preprocess import align_training_data, fit_states, run_data_processing
from ..solvers import normalize, solve as solve_model


class TMILP(Disaggregator):
    """基于混合整数线性规划的非侵入式负荷分解。

    Parameters
    ----------
    solver : {"highs", "cbc", "scip", "cpsat"}
        求解器后端, 默认 ``highs``。
    n_states : int | dict
        每个电器的状态数 (K-means 聚类数); 可以是统一整数, 也可以是
        ``{"fridge": 2, "kettle": 3}`` 这样的字典, 缺省 2。
    always_on : dict
        ``{"fridge": True}``, 表示该电器不存在"全关"状态。
    window_days : int
        每次求解的窗口长度 (天), 默认 1。
    gap : float
        MIP 相对间隙, 默认 1e-3。
    time_limit : float
        单窗口求解时间上限 (秒), 默认 60。
    C_start, C_run : float
        目标函数中的启动/运行惩罚系数 (对应 MATLAB 的 C_start / C_run)。
    sample_period : float, optional
        采样周期 (秒)。默认从索引自动推断。
    save-model-path / pretrained-model-path : str
        模型保存 / 加载目录 (与 nilmtk-contrib 命名保持一致)。
    solver_options : dict
        透传给求解器的其它参数 (如 ``{"ub_cap": 20000, "threads": 4}``)。
    verbose : bool
        是否打印求解器日志。
    """

    def __init__(self, params=None):
        params = dict(params or {})
        self.model = None
        self.MODEL_NAME = str(params.get("model_name", "TMILP"))

        self.solver = normalize(params.get("solver", "highs"))
        self.solver_options = dict(params.get("solver_options", {}))
        self.gap = float(params.get("gap", 1e-3))
        self.time_limit = float(params.get("time_limit", 60.0))
        self.verbose = bool(params.get("verbose", False))

        self.n_states = params.get("n_states", 2)
        self.always_on = dict(params.get("always_on", {}))
        self.window_days = int(params.get("window_days", 1))
        self.C_start = float(params.get("C_start", 0.892))
        self.C_run = float(params.get("C_run", 1.0419))
        self.sample_period = params.get("sample_period")
        self.preprocessing = params.get("preprocessing", "auto")
        self.nilmtcmilp_home = params.get("nilmtcmilp_home")

        self.save_model_path = params.get("save-model-path")
        self.load_model_path = params.get("pretrained-model-path")

        # 训练需要完整的多日数据, 不支持 NILMTK 的 chunk-wise 训练
        self.chunk_wise_training = False

        if self.nilmtcmilp_home:
            import_nilmtcmilp(self.nilmtcmilp_home)
        if self.load_model_path:
            self.load_model(self.load_model_path)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        """从电器级真值中学习模型参数。

        Parameters
        ----------
        train_main : list[pd.DataFrame]
            总表读数 (每个元素是一个 chunk)。
        train_appliances : list[tuple[str, list[pd.DataFrame]]]
            ``[(电器名, [chunk, ...]), ...]``。
        """
        _, _, powers, names, per_day = align_training_data(
            train_main, train_appliances, self.sample_period
        )
        S, cluster_id = fit_states(powers, names, self.n_states)
        dp = run_data_processing(
            powers, S, cluster_id, names, per_day, self.always_on,
            preprocessing=self.preprocessing,
        )

        self.model = {
            "dp": dp,
            "S": np.asarray(S).ravel(),
            "names": list(names),
            "per_day": int(per_day),
            "window_days": int(self.window_days),
            "C_start": float(self.C_start),
            "C_run": float(self.C_run),
        }
        if self.save_model_path:
            self.save_model(self.save_model_path)
        return self

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def disaggregate_chunk(self, test_mains):
        """对每个 chunk 做分解, 返回与输入等长的 ``List[pd.DataFrame]``。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练: 请先调用 partial_fit() 或提供 pretrained-model-path")

        results = []
        for frame in test_mains:
            results.append(self._disaggregate_frame(frame))
        return results

    def _disaggregate_frame(self, frame) -> pd.DataFrame:
        if isinstance(frame, pd.DataFrame):
            series = frame.iloc[:, 0] if frame.shape[1] else None
        elif isinstance(frame, pd.Series):
            series = frame
        else:
            series = pd.Series(np.asarray(frame).ravel())
        if series is None:
            raise ValueError("总表 DataFrame 至少需要一列功率")

        series = pd.to_numeric(series, errors="coerce").astype(float)
        if series.isna().any():
            series = series.interpolate(limit_direction="both").fillna(0.0)

        index = series.index
        values = series.to_numpy(dtype=float)

        per_day = int(self.model["per_day"])
        window_days = max(1, int(self.model["window_days"]))
        window = per_day * window_days
        n_app = len(self.model["names"])

        n_windows = int(np.ceil(len(values) / window)) if len(values) else 0
        predictions = np.zeros((max(n_windows * window, 0), n_app), dtype=float)

        for w in range(n_windows):
            seg = values[w * window:(w + 1) * window]
            if len(seg) < window:  # 末尾不足一个窗口: 用最后一个值补齐
                seg = np.concatenate([seg, np.full(window - len(seg), seg[-1])])
            try:
                block = self._solve_window(seg, window_days)
            except Exception as exc:  # 单个窗口失败不应中断整段
                warnings.warn(f"{self.MODEL_NAME}: 窗口 {w} 求解失败 ({exc}); 该段输出为 0")
                block = np.zeros((window, n_app), dtype=float)
            predictions[w * window:(w + 1) * window, :] = block

        return pd.DataFrame(
            predictions[: len(values)], index=index, columns=self.model["names"]
        )

    def _solve_window(self, mains: np.ndarray, window_days: int) -> np.ndarray:
        dp = self.model["dp"]
        S = self.model["S"]
        T = int(len(mains))

        model = build_tmilp_model(
            dp, mains, S, T, window_days,
            self.model["C_start"], self.model["C_run"],
        )
        result = solve_model(
            model,
            solver=self.solver,
            gap=self.gap,
            time_limit=self.time_limit,
            verbose=self.verbose,
            **self.solver_options,
        )
        if not result.ok:
            raise RuntimeError(f"求解器 {self.solver} 未返回解 (status={result.status})")
        return self._decode(result.x, dp, S, T)

    @staticmethod
    def _decode(x, dp, S, T: int) -> np.ndarray:
        """把解向量还原成 (T, n_appliances) 的功率矩阵。"""
        nilmtcmilp = import_nilmtcmilp()
        x = np.asarray(x, dtype=float).ravel()
        state_size = int(np.sum(S))
        ssT = state_size * T
        if x.size < 4 * ssT:
            raise ValueError(f"解向量长度 {x.size} 小于模型变量数 {4 * ssT}")

        x_vec = x[:ssT]
        p_range = np.tile(np.asarray(dp.P_top, float) - np.asarray(dp.P_bottom, float), T)
        p_base = np.tile(np.asarray(dp.P_bottom, float), T)
        power = p_range * x[2 * ssT:3 * ssT] + x_vec * p_base

        parts = nilmtcmilp.decompose_power(x_vec, power, np.asarray(S), T)
        return np.column_stack([np.asarray(p, float) for p in parts])

    # ------------------------------------------------------------------
    # 模型持久化
    # ------------------------------------------------------------------
    def save_model(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        path = os.path.join(folder_name, "model.pkl")
        payload = dict(self.model or {})
        payload["MODEL_NAME"] = self.MODEL_NAME
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    def load_model(self, folder_name):
        # 反序列化 EmilpData 前, 必须先让 nilmtcmilp 可导入 (pickle 按模块名找类)
        import_nilmtcmilp()
        path = os.path.join(folder_name, "model.pkl")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload.pop("MODEL_NAME", None)
        self.model = payload
        return self.model
