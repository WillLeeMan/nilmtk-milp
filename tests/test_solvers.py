"""四个求解器后端的最小可用性测试 (背包问题)。"""

import numpy as np
from scipy import sparse

from nilmtk_milp.core.model import MilpModel
from nilmtk_milp.solvers import available_solvers, solve

EXPECTED_X = np.array([1.0, 0.0, 1.0])


def knapsack() -> MilpModel:
    n = 3
    return MilpModel(
        f=np.array([-5.0, -4.0, -3.0]),
        Aineq=sparse.csr_matrix([[2.0, 3.0, 1.0]]),
        bineq=np.array([4.0]),
        lb=np.zeros(n),
        ub=np.ones(n),
        ctype="BBB",
    )


def test_all_backends():
    status = available_solvers()
    print("可用后端:", status)
    for name in ("highs", "cbc", "scip", "cpsat"):
        if not status[name]:
            print(f"[skip] {name} 未安装")
            continue
        result = solve(knapsack(), solver=name, gap=1e-6, time_limit=30)
        assert result.ok, f"{name} 未返回解: {result.status}"
        assert abs(result.objval + 8.0) < 1e-5, f"{name} 目标值错误: {result.objval}"
        np.testing.assert_allclose(result.x, EXPECTED_X, atol=1e-5)
        print(f"[ok] {name}: obj={result.objval:.6f} status={result.status} "
              f"runtime={result.runtime:.3f}s")


if __name__ == "__main__":
    test_all_backends()
