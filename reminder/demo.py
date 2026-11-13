# -*- coding: utf-8 -*-
"""内置演示：不连任何行情，用假的三个价格走完整条链路。

    python -m reminder demo

会依次展示：
  1. 第一次看到价格 -> 不报（没有"上一轮"可比）
  2. 上穿阈值 -> 报
  3. 继续在上方 -> 不报（cross 只报穿越那一次）
  4. 跌破止损 -> 报
  5. 冷却期内继续跌 -> 不报
  6. 进程重启后 -> 仍然不报（状态落盘了）
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .channels import CallableChannel, ConsoleChannel
from .engine import AlertEngine, StateStore
from .models import Rule

RULES = [
    {
        "id": "demo_cross",
        "name": "上穿 100",
        "symbols": ["AAAA"],
        "conditions": [{"field": "close", "op": "cross_above", "value": 100}],
        "cooldown_seconds": 0,
        "severity": "info",
        "message": "AAAA 上穿 100，现价 {close}",
    },
    {
        "id": "demo_stop",
        "name": "跌破 95 止损",
        "symbols": ["AAAA"],
        "conditions": [{"field": "close", "op": "<=", "value": 95}],
        "cooldown_seconds": 3600,
        "severity": "critical",
        "message": "AAAA 跌破止损，现价 {close}",
    },
]

TICKS = [
    ("第 1 轮 开在 99（首次观察，不报穿越）", 99.0),
    ("第 2 轮 涨到 101（上穿 100 → 报）", 101.0),
    ("第 3 轮 涨到 103（仍在 100 上方，不重复报）", 103.0),
    ("第 4 轮 跌到 94（跌破止损 → 报）", 94.0),
    ("第 5 轮 继续跌到 92（冷却期内，不报）", 92.0),
]


def _make_engine(state_file: Path, log_file: Path, sink: list) -> AlertEngine:
    return AlertEngine(
        rules=[Rule.from_dict(r) for r in RULES],
        channels={
            "console": ConsoleChannel(color=False),
            "capture": CallableChannel(lambda a: sink.append(a)),
        },
        severity_routes={"info": ["console", "capture"], "critical": ["console", "capture"]},
        state=StateStore(state_file),
        alert_log=log_file,
    )


def run_demo() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="reminder_demo_"))
    print("=" * 64)
    print("reminder 演示（数据是假的，只看链路）")
    print("=" * 64)
    try:
        sink: list = []
        engine = _make_engine(tmp / "state.json", tmp / "alerts.jsonl", sink)

        for label, price in TICKS:
            print(f"\n▶ {label}")
            fired = engine.evaluate({"AAAA": {"name": "演示股", "close": price}})
            if not fired:
                print("   → 无告警")

        engine.state.save()

        print("\n" + "-" * 64)
        print("模拟进程重启（重新加载 state.json）")
        print("-" * 64)
        sink2: list = []
        engine2 = _make_engine(tmp / "state.json", tmp / "alerts.jsonl", sink2)
        print("\n▶ 第 6 轮 跌到 90（重启后仍在冷却期）")
        fired = engine2.evaluate({"AAAA": {"name": "演示股", "close": 90.0}})
        if not fired:
            print("   → 无告警（冷却状态已从磁盘恢复）")

        log_lines = (tmp / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
        print("\n" + "=" * 64)
        print(f"结果：共 {len(sink)} 条告警，日志 {len(log_lines)} 行")
        print("=" * 64)

        ok = len(sink) == 2 and len(sink2) == 0 and len(log_lines) == 2
        print("✅ 符合预期" if ok else "❌ 不符合预期，请把上面的输出发给我")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(run_demo())
