"""最小可运行示例：合成一天的用电数据，跑 TMILP 并对比四种求解器。

用法::

    python examples/run_demo.py --solver highs
    python examples/run_demo.py --all
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from nilmtk_milp.disaggregate import TMILP


def make_data(seed: int = 0, per_day: int = 24, days: int = 3, start: str = "2024-01-01"):
    """两电器 + 基础负载的合成数据 (小时粒度)。"""
    rng = np.random.default_rng(seed)
    n = per_day * days
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")

    fridge = 60.0 + 40.0 * (np.sin(np.arange(n) / 3.0) > 0)   # 60/100 W 两态
    kettle = np.where(rng.random(n) < 0.08, 1500.0, 0.0)       # 偶发 1500 W
    base = 20.0 + rng.normal(0, 1.0, n)
    mains = fridge + kettle + base

    apps = pd.DataFrame({"fridge": fridge, "kettle": kettle}, index=index)
    return pd.DataFrame({"power": mains}, index=index), apps


def run(solver: str, save_model: bool = False):
    train_mains, train_apps = make_data(seed=0, days=3)
    test_mains, test_apps = make_data(seed=1, days=1, start="2024-02-01")

    clf = TMILP({
        "solver": solver,
        "n_states": {"fridge": 2, "kettle": 2},
        "always_on": {"fridge": True},
        "window_days": 1,
        "gap": 1e-2,
        "time_limit": 60,
    })
    clf.partial_fit(
        [train_mains],
        [("fridge", [train_apps[["fridge"]]]), ("kettle", [train_apps[["kettle"]]])],
    )

    if save_model:
        clf.save_model("model_demo")
        print("模型已保存到 model_demo/")

    pred = clf.disaggregate_chunk([test_mains])[0]
    for name in pred.columns:
        mae = float(np.mean(np.abs(pred[name].to_numpy() - test_apps[name].to_numpy())))
        print(f"  {name:8s} MAE = {mae:8.2f} W")
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="highs",
                        choices=["highs", "cbc", "scip", "cpsat"])
    parser.add_argument("--all", action="store_true", help="依次跑四种求解器")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    solvers = ["highs", "cbc", "scip", "cpsat"] if args.all else [args.solver]
    for name in solvers:
        print(f"== solver = {name} ==")
        run(name, save_model=args.save_model)


if __name__ == "__main__":
    main()
