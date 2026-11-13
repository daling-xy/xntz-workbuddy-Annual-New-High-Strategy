# -*- coding: utf-8 -*-
"""规则与告警的数据模型。

设计参考 tick-stock-panel 的 monitor_rules.py：
  - 一条规则一个 JSON 文件，纯函数校验，不碰任何数据源
  - 条件写成 (field, op, value) 三元组，加新指标不用改代码

本模块不知道行情从哪来，也不知道告警往哪发。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")

# 比较运算。cross_* 需要上一轮的值，由引擎维护。
OPS = {">", ">=", "<", "<=", "==", "!=", "cross_above", "cross_below"}
LOGICS = {"and", "or"}
SEVERITIES = {"info", "warn", "critical"}
SCOPES = {"symbols", "all"}


class RuleError(ValueError):
    """规则非法。消息是中文，直接给人看。"""


def _as_condition(raw: Any) -> "Condition":
    if not isinstance(raw, Mapping):
        raise RuleError(f"condition 必须是对象，收到 {type(raw).__name__}")
    field_name = raw.get("field")
    if not isinstance(field_name, str) or not field_name.strip():
        raise RuleError("condition.field 不能为空")
    op = raw.get("op")
    if op not in OPS:
        raise RuleError(f"condition.op 必须是 {sorted(OPS)} 之一，收到 {op!r}")

    ref = raw.get("ref")
    value = raw.get("value")
    if ref is not None:
        if not isinstance(ref, str) or not ref.strip():
            raise RuleError("condition.ref 必须是非空字段名")
        if value is not None:
            raise RuleError(f"condition 不能同时给 value 和 ref（field={field_name}）")
    elif value is None:
        raise RuleError(f"condition 必须给 value 或 ref（field={field_name}）")
    elif isinstance(value, (dict, list)):
        raise RuleError(f"condition.value 只支持数字/字符串/布尔（field={field_name}）")

    return Condition(field=field_name.strip(), op=str(op), value=value, ref=ref)


@dataclass
class Condition:
    field: str
    op: str
    value: Any = None
    ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"field": self.field, "op": self.op}
        if self.ref is not None:
            d["ref"] = self.ref
        else:
            d["value"] = self.value
        return d


@dataclass
class Rule:
    id: str
    name: str
    conditions: List[Condition]
    enabled: bool = True
    type: str = "price"
    scope: str = "symbols"
    symbols: List[str] = field(default_factory=list)
    logic: str = "and"
    cooldown_seconds: int = 300
    severity: str = "info"
    # 空 = 按 severity 走全局路由；非空 = 只发这些渠道
    channels: List[str] = field(default_factory=list)
    message: str = ""
    # 任意附加说明，引擎不读，给人看
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "type": self.type,
            "scope": self.scope,
            "symbols": list(self.symbols),
            "conditions": [c.to_dict() for c in self.conditions],
            "logic": self.logic,
            "cooldown_seconds": self.cooldown_seconds,
            "severity": self.severity,
            "channels": list(self.channels),
            "message": self.message,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Rule":
        if not isinstance(raw, Mapping):
            raise RuleError("规则必须是 JSON 对象")
        rid = raw.get("id")
        if not isinstance(rid, str) or not ID_RE.match(rid):
            raise RuleError(f"规则 id 非法（小写字母/数字/下划线，1-40 字符）：{rid!r}")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuleError(f"{rid}: name 不能为空")

        conds_raw = raw.get("conditions")
        if not isinstance(conds_raw, list) or not conds_raw:
            raise RuleError(f"{rid}: conditions 必须是非空数组")
        conditions = [_as_condition(c) for c in conds_raw]

        logic = raw.get("logic", "and")
        if logic not in LOGICS:
            raise RuleError(f"{rid}: logic 必须是 {sorted(LOGICS)} 之一")

        scope = raw.get("scope", "symbols")
        if scope not in SCOPES:
            raise RuleError(f"{rid}: scope 必须是 {sorted(SCOPES)} 之一")

        symbols_raw = raw.get("symbols", [])
        if not isinstance(symbols_raw, list) or any(not isinstance(s, str) for s in symbols_raw):
            raise RuleError(f"{rid}: symbols 必须是字符串数组")
        symbols = [s.strip() for s in symbols_raw if s.strip()]
        if scope == "symbols" and not symbols:
            raise RuleError(f"{rid}: scope=symbols 时必须给 symbols")

        severity = raw.get("severity", "info")
        if severity not in SEVERITIES:
            raise RuleError(f"{rid}: severity 必须是 {sorted(SEVERITIES)} 之一")

        cooldown = raw.get("cooldown_seconds", 300)
        if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or cooldown < 0:
            raise RuleError(f"{rid}: cooldown_seconds 必须是非负数字")

        channels = raw.get("channels", [])
        if not isinstance(channels, list) or any(not isinstance(c, str) for c in channels):
            raise RuleError(f"{rid}: channels 必须是字符串数组")

        return cls(
            id=rid,
            name=name.strip(),
            conditions=conditions,
            enabled=bool(raw.get("enabled", True)),
            type=str(raw.get("type", "price")),
            scope=scope,
            symbols=symbols,
            logic=logic,
            cooldown_seconds=int(cooldown),
            severity=severity,
            channels=[c.strip() for c in channels if c.strip()],
            message=str(raw.get("message", "") or ""),
            note=str(raw.get("note", "") or ""),
        )


# ---------------------------------------------------------------- 文件存储
def load_rule_file(path: Path) -> Rule:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuleError(f"{path}: JSON 语法错误 第{e.lineno}行 第{e.colno}列：{e.msg}") from e
    rule = Rule.from_dict(raw)
    # 文件名与 id 不一致时给出提醒，但不强制改
    if Path(path).stem != rule.id:
        rule.note = (rule.note + f" [文件名 {Path(path).stem} 与 id 不一致]").strip()
    return rule


def load_rules(rules_dir: Path) -> List[Rule]:
    """读取目录下全部 *.json。任一文件非法即抛错，不静默跳过。"""
    d = Path(rules_dir)
    if not d.exists():
        raise RuleError(f"规则目录不存在：{d}")
    rules: List[Rule] = []
    seen: Dict[str, Path] = {}
    for p in sorted(d.glob("*.json")):
        rule = load_rule_file(p)
        if rule.id in seen:
            raise RuleError(f"规则 id 重复：{rule.id}（{seen[rule.id]} 与 {p}）")
        seen[rule.id] = p
        rules.append(rule)
    return rules


def save_rule(rule: Rule, rules_dir: Path) -> Path:
    d = Path(rules_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{rule.id}.json"
    p.write_text(
        json.dumps(rule.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return p


def delete_rule(rule_id: str, rules_dir: Path) -> bool:
    p = Path(rules_dir) / f"{rule_id}.json"
    if p.exists():
        p.unlink()
        return True
    return False


def describe(rule: Rule) -> str:
    """人话描述一条规则，用于 validate / list 输出。"""
    parts = []
    for c in rule.conditions:
        rhs = f"字段[{c.ref}]" if c.ref else repr(c.value)
        parts.append(f"{c.field} {c.op} {rhs}")
    joiner = " 且 " if rule.logic == "and" else " 或 "
    scope = "全部标的" if rule.scope == "all" else ",".join(rule.symbols)
    return f"[{rule.severity}] {rule.name}（{rule.id}）｜{scope}｜{joiner.join(parts)}｜冷却 {rule.cooldown_seconds}s"


__all__ = [
    "ID_RE", "OPS", "LOGICS", "SEVERITIES", "SCOPES",
    "RuleError", "Condition", "Rule",
    "load_rule_file", "load_rules", "save_rule", "delete_rule", "describe",
]
