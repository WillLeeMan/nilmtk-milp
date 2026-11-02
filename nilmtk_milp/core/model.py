"""与 MATLAB ``model`` 结构体对应的求解器无关模型容器。

字段语义与 ``nilmtcmilp.milp_model.MilpModel`` 完全一致,
方便在没有安装 ``nilmtcmilp`` 的环境里单独构造/测试模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse


@dataclass
class MilpModel:
    f: Optional[np.ndarray] = None
    Aineq: Optional[sparse.csr_matrix] = None
    bineq: Optional[np.ndarray] = None
    Aeq: Optional[sparse.csr_matrix] = None
    beq: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    ctype: Optional[str] = None
    Q: Optional[sparse.csr_matrix] = None
    Qc: Optional[List[sparse.csr_matrix]] = None
    l: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    H: Optional[sparse.csr_matrix] = None
    variable_location: Dict[str, np.ndarray] = field(default_factory=dict)

    def num_vars(self) -> int:
        if self.f is not None:
            return int(self.f.shape[0])
        if self.lb is not None:
            return int(self.lb.shape[0])
        if self.ctype is not None:
            return len(self.ctype)
        return 0

    def model_size(self):
        """[变量数, 约束行数, 非零元个数]"""
        rows = 0
        nnz = 0
        for A in (self.Aineq, self.Aeq):
            if A is not None and A.shape[0] > 0:
                rows += A.shape[0]
                nnz += A.nnz
        return [self.num_vars(), rows, nnz]
