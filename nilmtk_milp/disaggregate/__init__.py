"""NILMTK 的算法入口, 风格与 ``nilmtk_contrib.disaggregate`` 一致 (懒加载)。

用法::

    from nilmtk_milp.disaggregate import TMILP
    clf = TMILP({"solver": "highs", "n_states": 3})
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TMILP": ("nilmtk_milp.disaggregate.tmilp", "TMILP"),
    "Disaggregator": ("nilmtk.disaggregate", "Disaggregator"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, class_name)
    globals()[name] = value
    return value
