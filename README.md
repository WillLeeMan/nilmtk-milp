# nilmtk-milp

把 **TMILP-1bin-project**（混合整数线性规划负荷分解模型）接入
[NILMTK](https://github.com/nilmtk/nilmtk) 的算法包，布局参照
[nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib)。

- 模型本体（变量、约束、目标函数）复用你已有的 `nilmtcmilp` 包，**只读引用，不修改其源码**；
- 本包负责三件事：把 NILMTK 的 MeterGroup/DataFrame 整理成模型输入、调度求解器、把解还原成各电器功率；
- 求解器可在 **HiGHS / CBC / SCIP / CP-SAT** 之间切换，用户按自己环境任选其一。

```
NMtk/
├── pyproject.toml
├── README.md
├── nilmtk_milp/
│   ├── __init__.py
│   ├── _source.py              # 定位外部 nilmtcmilp 包
│   ├── disaggregate/
│   │   ├── __init__.py         # 懒加载导出 TMILP
│   │   └── tmilp.py            # Disaggregator 子类（partial_fit / disaggregate_chunk / save/load）
│   ├── core/
│   │   ├── model.py            # 与 MATLAB model 结构体等价的 MilpModel
│   │   ├── preprocess.py       # NILMTK 数据 -> 数组 -> EmilpData
│   │   ├── build.py            # 调用 nilmtcmilp 的构造函数
│   │   └── dro.py              # DRO-LR 的 SciPy 回退实现
│   └── solvers/
│       ├── __init__.py         # solve(model, solver=...) 统一入口
│       ├── lp.py               # MilpModel -> CPLEX LP 文件
│       ├── highs_backend.py
│       ├── cbc_backend.py
│       ├── scip_backend.py
│       └── cpsat_backend.py
├── examples/run_demo.py
└── tests/
    ├── test_solvers.py
    └── test_tmilp_smoke.py
```

## 安装

NILMTK 目前不发布到 PyPI，需要先从源码安装（要求 Python 3.11+）：

```bash
pip install "nilmtk @ git+https://github.com/nilmtk/nilmtk.git"
```

然后安装本包和至少一个求解器：

```bash
pip install -e .
pip install highspy              # HiGHS
pip install "pulp[cbc]"          # CBC
pip install pyscipopt            # SCIP
pip install ortools              # CP-SAT
```

模型本体在独立仓库 [nilmtcmilp](https://github.com/WillLeeMan/nilmtcmilp)，先安装它：

```bash
pip install "nilmtcmilp @ git+https://github.com/WillLeeMan/nilmtcmilp.git"
```

如果只是本地开发，也可以直接 `pip install -e <nilmtcmilp 仓库目录>`，或用环境变量指向源码：

```bash
set NILMTCMILP_HOME=<nilmtcmilp 仓库目录>
```

也可以在构造 `TMILP` 时显式传入：

```python
clf = TMILP({"nilmtcmilp_home": r"D:\path\to\nilmtcmilp", "solver": "highs"})
```

## 快速开始

```python
import pandas as pd
from nilmtk_milp.disaggregate import TMILP

clf = TMILP({
    "solver": "highs",                 # highs / cbc / scip / cpsat
    "n_states": {"fridge": 2, "kettle": 2},
    "always_on": {"fridge": True},
    "gap": 1e-2,
    "time_limit": 60,
})

# train_main: list[DataFrame]；train_appliances: [(名字, [DataFrame, ...]), ...]
clf.partial_fit([train_mains], [("fridge", [fridge_df]), ("kettle", [kettle_df])])

predictions = clf.disaggregate_chunk([test_mains])   # -> list[DataFrame]
```

在 NILMTK 的 `API` 里使用：

```python
from nilmtk.api import API
from nilmtk_milp.disaggregate import TMILP

params = {
    "power": {"mains": "active", "appliance": "active"},
    "appliances": ["fridge", "kettle"],
    "train": {"datasets": {"redd": {"path": "redd.h5", "buildings": {1: {...}}}}},
    "test":  {"datasets": {"redd": {"path": "redd.h5", "buildings": {2: {...}}}},
              "metrics": ["rmse"]},
    "methods": {"TMILP": TMILP({"solver": "highs", "n_states": 2})},
}
API(params)
```

也可以直接跑自带示例（合成数据，无需任何数据集文件）：

```bash
python examples/run_demo.py --all
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `solver` | `highs` | 求解器后端 |
| `n_states` | `2` | 每个电器的状态数，整数或 `{电器名: k}` |
| `always_on` | `{}` | `{电器名: True}`，表示没有全关状态 |
| `window_days` | `1` | 每次求解覆盖的天数（窗口长度） |
| `gap` | `1e-3` | MIP 相对间隙 |
| `time_limit` | `60` | 单窗口求解时间上限（秒） |
| `C_start`, `C_run` | `0.892`, `1.0419` | 目标函数里的启动 / 运行惩罚 |
| `sample_period` | 自动推断 | 采样周期（秒） |
| `preprocessing` | `auto` | `auto` / `copt` / `scipy`，DRO 回归用哪个后端 |
| `save-model-path` | – | 训练完把模型存到该目录 |
| `pretrained-model-path` | – | 直接从该目录加载模型（跳过训练） |
| `solver_options` | `{}` | 透传给求解器的参数（如 `{"ub_cap": 20000}`） |
| `verbose` | `False` | 打印求解器日志 |

## 求解器怎么选

| 后端 | 安装 | 特点 |
|---|---|---|
| `highs` | `pip install highspy` | 综合最快的开源 MIP，直接传稀疏矩阵，推荐默认 |
| `cbc` | `pip install "pulp[cbc]"` | 老牌免费求解器，通过 LP 文件调用，规模大时略慢 |
| `scip` | `pip install pyscipopt` | Apache-2.0，约束规划能力强，通过 LP 文件调用 |
| `cpsat` | `pip install ortools` | 二值变量密集型问题很快，但只吃整数，连续功率按 `scale` 放大取整 |

四者在同一个窗口上的目标值应当基本一致（CP-SAT 因取整会有 1e-2 W 量级的差异）。

## 实现说明与限制

- **训练需要整天数据**：状态统计、转移矩阵、最短运行时间都按"天"计算，`partial_fit`
  会自动截断到整天数；因此 `chunk_wise_training` 固定为 `False`。
- **窗口化推理**：`disaggregate_chunk` 按 `window_days * per_day` 切窗，末窗不足时用最后一个采样值补齐，输出再截断回原长度。
- **总表读数**：原 MATLAB/Python 流程用"所选电器功率之和"充当总表；这里改为传入电表实测总有功功率，因此残差项（`kP`）在真实数据上非零，这是正常现象。
- **DRO 回归**：预处理中的分布鲁棒回归原本依赖 COPT（二阶锥规划）。当环境里没有 `coptpy` 时，
  自动切换到 `core/dro.py` 的 SciPy 凸优化实现（数学等价，已验证与网格搜索最优一致）。
  需要严格复现原结果时，安装 `coptpy` 即可走原路径。
- **二次目标**（`Q` / `Qc`）暂未实现，HiGHS/CBC/SCIP/CP-SAT 后端只支持纯线性 MILP；
  TMILP-1bin-project 只做有功时正好是纯线性模型。
- `partial_fit` 要求电器名与 NILMTK 传进来的名字一致，输出列名即这些名字。

## 与上游项目的关系

- 模型构建、状态聚类、预处理逻辑全部来自配套仓库 [nilmtcmilp](https://github.com/WillLeeMan/nilmtcmilp)，本包只在运行时 import 它；
- 本仓库不包含 `nilmtcmilp` 的模型代码，使用前需要单独安装它（见上文"安装"）；
- DRO 回退只是在内存中替换函数引用，不会修改 `nilmtcmilp` 的任何文件。

## 测试

```bash
# 四后端单元测试（只需要求解器，不需要模型本体）
python tests/test_solvers.py

# 端到端冒烟测试（需要先安装 nilmtcmilp）
python tests/test_tmilp_smoke.py
```

## 引用

本包封装的是下面这篇论文中的紧凑 MILP 形式（启动变量连续化 + 归一化功率变量）：

> W. Li, X. Yan, and C. Gao, "A Compact Mixed-Integer Linear Programming Formulation for
> Non-Intrusive Load Monitoring With Startup Linking and Normalized Power Variables,"
> in *2026 IEEE/IAS Industrial and Commercial Power System Asia (I&CPS Asia)*, Nanjing, China, 2026.
> DOI: [10.1109/ICPSASIA70813.2026.11692312](https://doi.org/10.1109/ICPSASIA70813.2026.11692312)

```bibtex
@inproceedings{li2026compact,
  title     = {A Compact Mixed-Integer Linear Programming Formulation for Non-Intrusive
               Load Monitoring With Startup Linking and Normalized Power Variables},
  author    = {Li, Wei and Yan, Xingyu and Gao, Ciwei},
  booktitle = {2026 IEEE/IAS Industrial and Commercial Power System Asia (I\&CPS Asia)},
  address   = {Nanjing, China},
  year      = {2026},
  doi       = {10.1109/ICPSASIA70813.2026.11692312}
}
```

GitHub 会读取根目录的 [CITATION.cff](CITATION.cff)。
