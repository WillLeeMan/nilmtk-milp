"""把 :class:`~nilmtk_milp.core.model.MilpModel` 写成 CPLEX LP 格式。

CBC 与 SCIP 都能直接读取 LP 文件; 相比逐条 Python API 建模, 写文件再读入
在几万个变量的规模下快一到两个数量级 (CBC 2.10 的 MPS 读取器只认定长格式,
LP 是自由格式, 更稳)。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from .base import has_quadratic


def num_vars(model) -> int:
    for attr in ("f", "lb", "ub", "ctype"):
        value = getattr(model, attr, None)
        if value is not None:
            try:
                return int(value.shape[0])
            except AttributeError:
                return len(value)
    raise ValueError("模型为空: f/lb/ub/ctype 全部为 None")


def _num(value: float) -> str:
    return repr(float(value))


def _terms(indices, values, names, scale: float = 1.0) -> str:
    parts = []
    for idx, val in zip(indices, values):
        coef = float(val) * scale
        if coef == 0.0:
            continue
        sign = "+" if coef > 0 else "-"
        mag = abs(coef)
        body = names[int(idx)] if mag == 1.0 else f"{_num(mag)} {names[int(idx)]}"
        parts.append(f"{sign} {body}")
    return " ".join(parts) if parts else "0"


def write_lp(model, path: str, name: str = "nilmtk_milp") -> str:
    """写出 LP 文件, 返回文件路径。"""
    if has_quadratic(model):
        raise ValueError("LP 格式不支持二次目标; 请改用 highs 后端, 或关闭 Q/Qc")

    n = num_vars(model)
    f = np.zeros(n) if getattr(model, "f", None) is None else np.asarray(model.f, float).ravel()
    lb = np.full(n, -np.inf) if getattr(model, "lb", None) is None else np.asarray(model.lb, float).ravel()
    ub = np.full(n, np.inf) if getattr(model, "ub", None) is None else np.asarray(model.ub, float).ravel()
    ctype = getattr(model, "ctype", None) or "C" * n
    if len(ctype) != n:
        raise ValueError(f"ctype 长度 {len(ctype)} 与变量数 {n} 不一致")

    names = [f"x{j}" for j in range(n)]
    binaries = [names[j] for j in range(n) if ctype[j] == "B"]
    generals = [names[j] for j in range(n) if ctype[j] == "I"]

    lines: list[str] = [f"\\* {name} *\\", "Minimize"]
    obj_line = _terms(np.arange(n), f, names)
    lines.append(f" obj: {obj_line}" if obj_line else " obj: 0")

    lines.append("Subject To")
    row = 0

    def emit_rows(mat, rhs, relation: str) -> int:
        nonlocal row
        if mat is None or mat.shape[0] == 0:
            return row
        csr = sparse.csr_matrix(mat)
        for i in range(csr.shape[0]):
            start, end = csr.indptr[i], csr.indptr[i + 1]
            if start == end:
                continue
            body = _terms(csr.indices[start:end], csr.data[start:end], names)
            lines.append(f" r{row}: {body} {relation} {_num(rhs[i])}")
            row += 1
        return row

    emit_rows(model.Aeq, np.asarray(model.beq, float).ravel(), "=") if getattr(model, "Aeq", None) is not None else None
    emit_rows(model.Aineq, np.asarray(model.bineq, float).ravel(), "<=") if getattr(model, "Aineq", None) is not None else None

    lines.append("Bounds")
    for j in range(n):
        if ctype[j] in ("B", "I"):
            continue
        lo, hi = float(lb[j]), float(ub[j])
        if math.isinf(lo) and math.isinf(hi):
            lines.append(f" {names[j]} free")
        elif lo == 0.0 and math.isinf(hi):
            continue
        elif math.isinf(lo):
            lines.append(f" {names[j]} <= {_num(hi)}")
        elif math.isinf(hi):
            lines.append(f" {names[j]} >= {_num(lo)}")
        elif lo == hi:
            lines.append(f" {names[j]} = {_num(lo)}")
        else:
            lines.append(f" {_num(lo)} <= {names[j]} <= {_num(hi)}")

    if binaries:
        lines.append("Binaries")
        for j in binaries:
            lines.append(f" {j}")
    if generals:
        lines.append("Generals")
        for j in generals:
            lines.append(f" {j}")
        for j, nm in enumerate(names):
            if ctype[j] != "I":
                continue
            lo, hi = float(lb[j]), float(ub[j])
            if not math.isinf(hi):
                lines.append(f" {nm} <= {_num(hi)}")
            if not math.isinf(lo) and lo != 0.0:
                lines.append(f" {nm} >= {_num(lo)}")

    lines.append("End")
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    return path
