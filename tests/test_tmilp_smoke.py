"""TMILP 适配器的端到端冒烟测试 (合成数据, 小时粒度, 不依赖任何数据集文件)。"""

import numpy as np
import pandas as pd

from nilmtk_milp.disaggregate import TMILP


def make_data(seed: int = 0, per_day: int = 24, days: int = 3, start: str = "2024-01-01"):
    rng = np.random.default_rng(seed)
    n = per_day * days
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")

    # 冰箱: 两态循环 (60W / 100W)
    fridge = 60.0 + 40.0 * (np.sin(np.arange(n) / 3.0) > 0)
    # 水壶: 偶发 1500W
    kettle = np.where(rng.random(n) < 0.08, 1500.0, 0.0)
    # 基础负载
    base = 20.0 + rng.normal(0, 1.0, n)

    mains = fridge + kettle + base
    apps = pd.DataFrame({"fridge": fridge + 0.0, "kettle": kettle}, index=index)
    mains_df = pd.DataFrame({"power": mains}, index=index)
    return mains_df, apps


def test_partial_fit_and_disaggregate():
    train_mains, train_apps = make_data(seed=0, days=3)
    test_mains, test_apps = make_data(seed=1, days=1, start="2024-02-01")

    clf = TMILP({
        "solver": "highs",
        "n_states": {"fridge": 2, "kettle": 2},
        "always_on": {"fridge": True},
        "window_days": 1,
        "gap": 1e-2,
        "time_limit": 30,
    })
    clf.partial_fit(
        [train_mains],
        [("fridge", [train_apps[["fridge"]]]), ("kettle", [train_apps[["kettle"]]])],
    )
    assert clf.model is not None
    print("S =", clf.model["S"], "per_day =", clf.model["per_day"])

    preds = clf.disaggregate_chunk([test_mains])
    assert isinstance(preds, list) and len(preds) == 1
    pred = preds[0]
    assert pred.shape == (len(test_mains), 2), pred.shape
    assert list(pred.columns) == ["fridge", "kettle"]
    assert not pred.isna().any().any()

    mae_fridge = float(np.mean(np.abs(pred["fridge"].to_numpy() - test_apps["fridge"].to_numpy())))
    print("fridge MAE =", mae_fridge)
    print(pred.head(3))


if __name__ == "__main__":
    test_partial_fit_and_disaggregate()
    print("OK")
