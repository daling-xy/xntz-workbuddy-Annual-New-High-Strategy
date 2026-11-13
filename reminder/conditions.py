# -*- coding: utf-8 -*-
"""条件求值：把 (field, op, value) 作用到一条行情快照上。

行情快照就是普通的 dict，例如：
    {"symbol": "600519", "close": 1500.0, "vol_ratio": 2.3,
     "kline": {"ma20": 1480.0}}

字段名支持点号取嵌套：`kline.ma20`。
本模块不关心字段从哪来 —— 只要 dict 里有这个键就能算。
"""
from __future__ import annotations

import string
from typing import Any, Mapping, Optional

_MISSING = object()


def get_path(snapshot: Mapping[str, Any], path: str) -> Any:
    """按点号路径取值，取不到返回 None。"""
    cur: Any = snapshot
    for part in str(path).split("."):
        if isinstance(cur, Mapping):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _to_number(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def compare(left: Any, op: str, right: Any) -> Optional[bool]:
    """返回 True/False；无法比较（缺值、类型不符）返回 None。

    None 与 False 的区别很重要：
      - False = 确实不满足
      - None  = 这一轮算不出来（数据还没到），不该触发也不该记状态
    """
    if left is None or right is None:
        return None

    ln, rn = _to_number(left), _to_number(right)
    if ln is not None and rn is not None:
        a, b = ln, rn
    elif isinstance(left, str) and isinstance(right, str):
        a, b = left, right
    elif isinstance(left, bool) and isinstance(right, bool):
        a, b = left, right
    else:
        return None

    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    return None


def cross(left_now: Any, left_prev: Any, op: str, right: Any) -> Optional[bool]:
    """穿越判定。需要上一轮的值；没有上一轮一律返回 None。

    cross_above: 上一轮 <= 阈值，本轮 > 阈值
    cross_below: 上一轮 >= 阈值，本轮 < 阈值
    """
    if left_prev is None or left_now is None or right is None:
        return None
    ln, lp, rn = _to_number(left_now), _to_number(left_prev), _to_number(right)
    if ln is None or lp is None or rn is None:
        return None
    if op == "cross_above":
        return lp <= rn and ln > rn
    if op == "cross_below":
        return lp >= rn and ln < rn
    return None


def eval_condition(
    cond: Any,
    snapshot: Mapping[str, Any],
    prev_value: Any = None,
) -> Optional[bool]:
    """求值单个条件。cond 需要有 .field / .op / .value / .ref 属性。"""
    left = get_path(snapshot, cond.field)
    right = get_path(snapshot, cond.ref) if getattr(cond, "ref", None) else cond.value

    if cond.op in ("cross_above", "cross_below"):
        return cross(left, prev_value, cond.op, right)
    return compare(left, cond.op, right)


class _SafeDict(dict):
    """模板里出现未知占位符时原样保留，不抛 KeyError。"""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


class _DottedFormatter(string.Formatter):
    """让 `{kline.ma20}` 按"整个字段名"查字典，而不是取 kline 的属性。

    Python 自带的 str.format 会把 `a.b` 解释成"先取 a，再取属性 b"，
    而我们这里 `kline.ma20` 是一个扁平化的键名，必须优先按整串匹配。
    """

    def __init__(self, mapping: Mapping[str, Any]):
        super().__init__()
        self.mapping = mapping

    def get_field(self, field_name, args, kwargs):  # type: ignore[override]
        if field_name in self.mapping:
            return self.mapping[field_name], field_name
        return super().get_field(field_name, args, kwargs)


def render(template: str, context: Mapping[str, Any]) -> str:
    """渲染消息模板。`{symbol}` `{close}` `{kline.ma20}` 都能用。"""
    if not template:
        return ""
    flat = dict(context)
    # 允许 {kline.ma20} 这种点号写法
    for key in list(context.keys()):
        value = context[key]
        if isinstance(value, Mapping):
            for k2, v2 in value.items():
                flat.setdefault(f"{key}.{k2}", v2)
    try:
        return _DottedFormatter(_SafeDict(flat)).vformat(str(template), (), {})
    except Exception:
        return str(template)


__all__ = ["get_path", "compare", "cross", "eval_condition", "render"]
