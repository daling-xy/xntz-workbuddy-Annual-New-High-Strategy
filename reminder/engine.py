# -*- coding: utf-8 -*-
"""告警引擎：把规则 + 行情 + 渠道串起来。

核心循环只有四步：
    取数 -> 逐条规则求值 -> 冷却判断 -> 推送并记日志

两条铁律：
  1. 冷却按 (规则, 标的) 记，不是按规则 —— 一只票触发不该压住另一只
  2. 状态必须落盘 —— 否则重启一次就把冷却清零，一天内重复轰炸
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .channels import Alert, Channel
from .conditions import eval_condition, get_path, render
from .models import Rule

DEFAULT_MESSAGE = "{rule_name}｜{symbol} 触发"


class StateStore:
    """冷却表与上一轮取值，落盘为 JSON。

    last_fired: {"rule_id|symbol": unix_ts}
    prev:       {"rule_id|symbol|field": 上一轮的数值}
    """

    def __init__(self, path: Path | str, autosave: bool = True):
        self.path = Path(path)
        self.autosave = autosave
        self.last_fired: Dict[str, float] = {}
        self.prev: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] 状态文件损坏，按空状态启动：{e}", file=sys.stderr)
            return
        self.last_fired = {str(k): float(v) for k, v in (raw.get("last_fired") or {}).items()}
        self.prev = dict(raw.get("prev") or {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"last_fired": self.last_fired, "prev": self.prev},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    # --- 便捷读写 ---
    def fired_at(self, rule_id: str, symbol: str) -> float:
        return float(self.last_fired.get(f"{rule_id}|{symbol}", 0.0))

    def mark_fired(self, rule_id: str, symbol: str, ts: float) -> None:
        self.last_fired[f"{rule_id}|{symbol}"] = float(ts)

    def prev_value(self, rule_id: str, symbol: str, field_name: str) -> Any:
        return self.prev.get(f"{rule_id}|{symbol}|{field_name}")

    def set_prev(self, rule_id: str, symbol: str, field_name: str, value: Any) -> None:
        self.prev[f"{rule_id}|{symbol}|{field_name}"] = value


class AlertEngine:
    def __init__(
        self,
        rules: Sequence[Rule],
        channels: Mapping[str, Channel],
        severity_routes: Optional[Mapping[str, Sequence[str]]] = None,
        state: Optional[StateStore] = None,
        alert_log: Optional[Path | str] = None,
        clock=time.time,
    ):
        self.rules = list(rules)
        self.channels = dict(channels)
        self.severity_routes = {
            k: list(v) for k, v in (severity_routes or {
                "info": ["console"],
                "warn": ["console"],
                "critical": ["console"],
            }).items()
        }
        self.state = state or StateStore("reminder_state.json", autosave=False)
        self.alert_log = Path(alert_log) if alert_log else None
        self.clock = clock
        self._warned_channels: set[str] = set()

    # ------------------------------------------------------------ 取数范围
    def required_symbols(self) -> List[str]:
        """需要向数据源请求的标的。返回空列表 = 让数据源给全部。"""
        if any(r.scope == "all" for r in self.rules if r.enabled):
            return []
        syms: List[str] = []
        seen = set()
        for r in self.rules:
            if not r.enabled:
                continue
            for s in r.symbols:
                if s not in seen:
                    seen.add(s)
                    syms.append(s)
        return syms

    # ------------------------------------------------------------ 求值
    def _targets(self, rule: Rule, snapshot: Mapping[str, Mapping[str, Any]]) -> List[str]:
        if rule.scope == "all":
            return list(snapshot.keys())
        return [s for s in rule.symbols if s in snapshot]

    def _eval_rule(self, rule: Rule, symbol: str, quote: Mapping[str, Any]) -> tuple[bool, Dict[str, Any]]:
        ctx = {"symbol": symbol, "rule_id": rule.id, "rule_name": rule.name,
               "severity": rule.severity}
        ctx.update(dict(quote))

        results: List[Optional[bool]] = []
        for cond in rule.conditions:
            prev = self.state.prev_value(rule.id, symbol, cond.field)
            results.append(eval_condition(cond, ctx, prev))
            # 记录本轮取值，供下一次 cross 判定。
            # 只在拿到值时才记录：缺数据把 prev 写成 None，会让下一次真穿越被吞掉。
            current = get_path(ctx, cond.field)
            if current is not None:
                self.state.set_prev(rule.id, symbol, cond.field, current)

        if rule.logic == "and":
            hit = all(r is True for r in results)
        else:
            hit = any(r is True for r in results)
        return hit, ctx

    # ------------------------------------------------------------ 推送
    def _route(self, rule: Rule) -> List[str]:
        if rule.channels:
            return rule.channels
        return self.severity_routes.get(rule.severity, ["console"])

    def _dispatch(self, alert: Alert, rule: Rule) -> None:
        for name in self._route(rule):
            ch = self.channels.get(name)
            if ch is None:
                if name not in self._warned_channels:
                    print(f"[warn] 渠道未定义：{name}", file=sys.stderr)
                    self._warned_channels.add(name)
                continue
            try:
                ch.send(alert)
            except Exception as e:
                print(f"[warn] 渠道 {name} 发送失败：{e}", file=sys.stderr)

        if self.alert_log:
            self.alert_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alert_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 主入口
    def evaluate(self, snapshot: Mapping[str, Mapping[str, Any]],
                 now: Optional[float] = None) -> List[Alert]:
        """对一份快照跑全部规则，返回本轮触发的告警。"""
        now = self.clock() if now is None else now
        fired: List[Alert] = []

        for rule in self.rules:
            if not rule.enabled:
                continue
            for symbol in self._targets(rule, snapshot):
                quote = snapshot[symbol]
                if not isinstance(quote, Mapping):
                    continue
                hit, ctx = self._eval_rule(rule, symbol, quote)

                if not hit:
                    continue
                if now - self.state.fired_at(rule.id, symbol) < rule.cooldown_seconds:
                    continue

                msg = render(rule.message or DEFAULT_MESSAGE, ctx)
                alert = Alert(
                    rule_id=rule.id, rule_name=rule.name, symbol=symbol,
                    severity=rule.severity, message=msg,
                    fields=dict(quote),
                )
                self.state.mark_fired(rule.id, symbol, now)
                self._dispatch(alert, rule)
                fired.append(alert)

        if fired and getattr(self.state, "autosave", True):
            self.state.save()
        return fired


# ---------------------------------------------------------------- 运行方式
def _safe_fetch(source: Any, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
    """取数失败只警告，不中断服务 —— 网络抖动不该把常驻进程打死。"""
    try:
        return source.fetch(list(symbols)) or {}
    except Exception as e:
        print(f"[warn] 取数失败（{type(source).__name__}）：{e}", file=sys.stderr, flush=True)
        return {}


def run_once(engine: AlertEngine, source: Any) -> List[Alert]:
    """跑一轮就返回。适合放进 cron（收盘后跑一次）。"""
    return engine.evaluate(_safe_fetch(source, engine.required_symbols()))


def run_poll(
    engine: AlertEngine,
    source: Any,
    interval: float = 30.0,
    max_ticks: Optional[int] = None,
    on_tick: Optional[Any] = None,
) -> int:
    """轮询循环。max_ticks 主要用于测试；返回实际跑了多少轮。"""
    ticks = 0
    while True:
        snapshot = _safe_fetch(source, engine.required_symbols())
        engine.evaluate(snapshot)
        ticks += 1
        if on_tick:
            on_tick(ticks, snapshot)
        if max_ticks is not None and ticks >= max_ticks:
            return ticks
        # 数据源已经放完（回放场景）就退出，别空转
        if not snapshot and getattr(source, "exhausted", False):
            return ticks
        time.sleep(interval)


__all__ = ["StateStore", "AlertEngine", "run_once", "run_poll"]
