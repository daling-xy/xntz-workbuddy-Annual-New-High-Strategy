# -*- coding: utf-8 -*-
"""reminder —— 规则化实时提醒引擎（数据源与推送渠道都可插拔）。

最小用法：

    from reminder import Rule, AlertEngine, ConsoleChannel, StateStore

    rule = Rule.from_dict({
        "id": "stop_loss",
        "name": "跌破止损",
        "symbols": ["600519"],
        "conditions": [{"field": "close", "op": "<=", "value": 1400}],
        "cooldown_seconds": 3600,
        "severity": "critical",
        "message": "{symbol} 现价 {close}，跌破止损",
    })

    engine = AlertEngine(
        rules=[rule],
        channels={"console": ConsoleChannel()},
        severity_routes={"critical": ["console"]},
        state=StateStore("state.json"),
    )
    engine.evaluate({"600519": {"close": 1390, "name": "贵州茅台"}})
"""
from __future__ import annotations

from .channels import (
    Alert, CallableChannel, Channel, ConsoleChannel, FileChannel,
    MultiChannel, WebhookChannel, WindowsToastChannel,
)
from .conditions import compare, cross, eval_condition, get_path, render
from .engine import AlertEngine, StateStore, run_once, run_poll
from .models import Condition, Rule, RuleError, describe, load_rules, save_rule
from .sources import CallableSource, CSVSource, JSONLSource, MappingSource, QuoteSource, TencentSource

__version__ = "1.0.0"

__all__ = [
    "Alert", "Channel", "ConsoleChannel", "FileChannel", "WindowsToastChannel",
    "WebhookChannel", "CallableChannel", "MultiChannel",
    "Condition", "Rule", "RuleError", "describe", "load_rules", "save_rule",
    "AlertEngine", "StateStore", "run_once", "run_poll",
    "QuoteSource", "MappingSource", "CallableSource", "JSONLSource", "CSVSource",
    "TencentSource", "compare", "cross", "eval_condition", "get_path", "render",
]
