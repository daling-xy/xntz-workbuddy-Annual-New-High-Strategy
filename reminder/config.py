# -*- coding: utf-8 -*-
"""配置加载：用 "模块路径:类名" 的方式把数据源和渠道接进来。

配置文件长这样：

{
  "poll_interval": 30,
  "rules_dir": "rules",
  "state_file": "state.json",
  "alert_log": "alerts.jsonl",
  "source": {
    "factory": "reminder.sources:TencentSource",
    "args": {}
  },
  "channels": {
    "console": {"factory": "reminder.channels:ConsoleChannel"},
    "toast":   {"factory": "reminder.channels:WindowsToastChannel"},
    "wechat":  {"factory": "reminder.channels:WebhookChannel",
                "args": {"url": "https://qyapi.weixin.qq.com/...", "format": "wecom"}}
  },
  "severity_routes": {
    "info": ["console"],
    "warn": ["console", "toast"],
    "critical": ["console", "toast", "wechat"]
  }
}

别人要接自己的数据源，只要写一个带 fetch(symbols) 的类，
然后在 source.factory 里填 "我的模块:我的类" 即可，不用改这个项目的任何代码。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple


class ConfigError(ValueError):
    pass


def load_json(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p} JSON 语法错误 第{e.lineno}行：{e.msg}") from e


def resolve_factory(spec: str) -> Any:
    """"module.path:Attr" -> 实际对象。"""
    if not isinstance(spec, str) or ":" not in spec:
        raise ConfigError(f'factory 必须写成 "模块:类名"，收到 {spec!r}')
    mod_name, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ConfigError(f"导入模块失败 {mod_name}：{e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ConfigError(f"模块 {mod_name} 里没有 {attr}") from e


def build(spec: Dict[str, Any]) -> Any:
    """{"factory": "...", "args": {...}} -> 实例"""
    if not isinstance(spec, dict) or "factory" not in spec:
        raise ConfigError(f"需要 factory 字段：{spec!r}")
    factory = resolve_factory(spec["factory"])
    args = spec.get("args") or {}
    if not isinstance(args, dict):
        raise ConfigError("args 必须是对象")
    return factory(**args)


def build_source(cfg: Dict[str, Any]) -> Any:
    if "source" not in cfg:
        raise ConfigError("配置缺少 source")
    src = build(cfg["source"])
    if not hasattr(src, "fetch"):
        raise ConfigError("数据源必须有 fetch(symbols) 方法")
    return src


def build_channels(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("channels") or {"console": {"factory": "reminder.channels:ConsoleChannel"}}
    out: Dict[str, Any] = {}
    for name, spec in raw.items():
        ch = build(spec)
        if not hasattr(ch, "send"):
            raise ConfigError(f"渠道 {name} 必须有 send(alert) 方法")
        out[name] = ch
    return out


def resolve_paths(cfg: Dict[str, Any], base_dir: Path) -> Tuple[Path, Path, Path]:
    base = Path(base_dir)
    rules_dir = base / cfg.get("rules_dir", "rules")
    state_file = base / cfg.get("state_file", "state.json")
    alert_log = base / cfg.get("alert_log", "alerts.jsonl")
    return rules_dir, state_file, alert_log


__all__ = ["ConfigError", "load_json", "resolve_factory", "build",
           "build_source", "build_channels", "resolve_paths"]
