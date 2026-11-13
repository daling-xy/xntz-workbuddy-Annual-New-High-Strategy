# -*- coding: utf-8 -*-
"""自检：不依赖 pytest，直接 python -m reminder.selftest 跑。

覆盖：条件运算、穿越语义、模板、规则校验、状态持久化、
      冷却、AND/OR、作用域、渠道路由、配置工厂、JSONL 回放。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from .channels import Alert, CallableChannel, ConsoleChannel
from .conditions import compare, cross, eval_condition, get_path, render
from .config import build, resolve_factory
from .engine import AlertEngine, StateStore
from .models import Condition, Rule, RuleError, load_rules, save_rule
from .sources import CallableSource, JSONLSource, MappingSource

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def section(title: str) -> None:
    print(f"\n[{title}]")


def rule(**over) -> Rule:
    base = {
        "id": "r1", "name": "测试", "symbols": ["A"],
        "conditions": [{"field": "close", "op": ">", "value": 10}],
    }
    base.update(over)
    return Rule.from_dict(base)


def main() -> int:
    section("条件运算")
    check("> 成立", compare(11, ">", 10) is True)
    check("> 不成立", compare(9, ">", 10) is False)
    check("== 处理字符串数字", compare("10.0", "==", 10) is True)
    check("缺值返回 None", compare(None, ">", 10) is None)
    check("类型不符返回 None", compare("abc", ">", 10) is None)

    section("穿越语义")
    check("首次观察不报", cross(101, None, "cross_above", 100) is None)
    check("有效上穿", cross(101, 99, "cross_above", 100) is True)
    check("持续在上方不重复", cross(102, 101, "cross_above", 100) is False)
    check("有效下穿", cross(99, 101, "cross_below", 100) is True)

    section("字段路径与模板")
    snap = {"close": 12.5, "kline": {"ma20": 11.0}}
    check("点号取值", get_path(snap, "kline.ma20") == 11.0)
    check("取值缺失返回 None", get_path(snap, "kline.ma60") is None)
    check("模板变量替换", render("{close}/{kline.ma20}", snap) == "12.5/11.0")
    check("未知占位符原样保留", render("{unknown}", snap) == "{unknown}")

    section("规则校验")
    try:
        Rule.from_dict({"id": "Bad ID", "name": "x", "conditions": [{"field": "a", "op": ">", "value": 1}]})
        check("非法 id 应报错", False)
    except RuleError:
        check("非法 id 报错", True)

    try:
        rule(conditions=[])
        check("空条件应报错", False)
    except RuleError:
        check("空条件报错", True)

    try:
        rule(conditions=[{"field": "a", "op": "≈", "value": 1}])
        check("非法运算符应报错", False)
    except RuleError:
        check("非法运算符报错", True)

    try:
        rule(symbols=[])
        check("scope=symbols 无标的应报错", False)
    except RuleError:
        check("scope=symbols 无标的报错", True)

    check("scope=all 不要求 symbols", rule(scope="all", symbols=[]).scope == "all")

    section("状态持久化")
    tmp = Path(tempfile.mkdtemp(prefix="reminder_test_"))
    try:
        sp = tmp / "state.json"
        s1 = StateStore(sp)
        s1.mark_fired("r1", "A", 1000.0)
        s1.set_prev("r1", "A", "close", 12.5)
        s1.save()

        s2 = StateStore(sp)
        check("冷却时间持久化", s2.fired_at("r1", "A") == 1000.0)
        check("上轮取值持久化", s2.prev_value("r1", "A", "close") == 12.5)

        sp.write_text("{ 坏掉的 json", encoding="utf-8")
        s3 = StateStore(sp)  # 不应抛异常
        check("损坏状态文件降级为空", s3.fired_at("r1", "A") == 0.0)

        section("引擎：AND / OR")
        sink: list = []
        eng = AlertEngine(
            rules=[rule(id="and_rule", logic="and", conditions=[
                {"field": "close", "op": ">", "value": 10},
                {"field": "vol", "op": ">", "value": 100},
            ])],
            channels={"cap": CallableChannel(sink.append)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_and.json"),
        )
        eng.evaluate({"A": {"close": 11, "vol": 50}})
        check("AND 缺一条件不报", len(sink) == 0)
        eng.evaluate({"A": {"close": 11, "vol": 150}})
        check("AND 全满足才报", len(sink) == 1)

        sink.clear()
        eng_or = AlertEngine(
            rules=[rule(id="or_rule", logic="or", conditions=[
                {"field": "close", "op": ">", "value": 10},
                {"field": "vol", "op": ">", "value": 100},
            ])],
            channels={"cap": CallableChannel(sink.append)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_or.json"),
        )
        eng_or.evaluate({"A": {"close": 5, "vol": 150}})
        check("OR 满足其一即报", len(sink) == 1)

        section("引擎：冷却与作用域")
        sink.clear()
        eng2 = AlertEngine(
            rules=[rule(id="cd", cooldown_seconds=100)],
            channels={"cap": CallableChannel(sink.append)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_cd.json"),
        )
        eng2.evaluate({"A": {"close": 11}}, now=1000)
        eng2.evaluate({"A": {"close": 11}}, now=1050)
        check("冷却期内不再报", len(sink) == 1)
        eng2.evaluate({"A": {"close": 11}}, now=1101)
        check("冷却结束后重新报", len(sink) == 2)

        sink.clear()
        eng3 = AlertEngine(
            rules=[rule(id="all_scope", scope="all", symbols=[], conditions=[
                {"field": "close", "op": ">", "value": 10}])],
            channels={"cap": CallableChannel(sink.append)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_all.json"),
        )
        eng3.evaluate({"A": {"close": 11}, "B": {"close": 11}, "C": {"close": 5}})
        check("scope=all 遍历全部标的", len(sink) == 2)
        check("required_symbols 为空表示全量", eng3.required_symbols() == [])

        section("渠道路由")
        sink_a, sink_b = [], []
        eng4 = AlertEngine(
            rules=[rule(id="route", severity="critical", channels=["b"])],
            channels={"a": CallableChannel(sink_a.append), "b": CallableChannel(sink_b.append)},
            severity_routes={"critical": ["a"]},
            state=StateStore(tmp / "s_route.json"),
        )
        eng4.evaluate({"A": {"close": 11}})
        check("规则自带的 channels 覆盖全局路由", len(sink_a) == 0 and len(sink_b) == 1)

        section("配置工厂")
        check("resolve_factory 可用",
              resolve_factory("reminder.channels:ConsoleChannel") is ConsoleChannel)
        ch = build({"factory": "reminder.channels:CallableChannel",
                    "args": {"fn": lambda a: None}})
        check("带参数构建渠道", isinstance(ch, CallableChannel))
        try:
            build({"factory": "reminder.channels:不存在的类"})
            check("非法 factory 应报错", False)
        except Exception:
            check("非法 factory 报错", True)

        section("数据源")
        ms = MappingSource({"A": {"close": 1}, "B": {"close": 2}})
        check("MappingSource 过滤标的", list(ms.fetch(["A"])) == ["A"])
        check("MappingSource 空列表给全部", len(ms.fetch([])) == 2)

        cs = CallableSource(lambda syms: {"A": {"close": 1}})
        check("CallableSource 可用", cs.fetch(["A"])["A"]["close"] == 1)

        jl = tmp / "replay.jsonl"
        jl.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in [
                {"A": {"close": 1}}, {"A": {"close": 2}},
            ]), encoding="utf-8")
        js = JSONLSource(jl)
        check("JSONL 第一轮", js.fetch([])["A"]["close"] == 1)
        check("JSONL 第二轮", js.fetch([])["A"]["close"] == 2)
        check("JSONL 放完后返回空", js.fetch([]) == {})
        check("JSONL exhausted 标记", js.exhausted is True)

        section("缺数据不污染穿越状态")
        sink.clear()
        eng_miss = AlertEngine(
            rules=[rule(id="miss", conditions=[
                {"field": "close", "op": "cross_above", "value": 100}])],
            channels={"cap": CallableChannel(sink.append)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_miss.json"),
        )
        eng_miss.evaluate({"A": {"close": 99}})
        eng_miss.evaluate({"A": {}})              # 这一轮字段缺失
        eng_miss.evaluate({"A": {"close": 101}})  # 仍应与 99 比较，识别为穿越
        check("中间缺一轮数据仍能识别穿越", len(sink) == 1)

        section("取数失败不中断")
        def boom(_symbols):
            raise RuntimeError("模拟网络故障")

        from .engine import run_once
        eng_boom = AlertEngine(
            rules=[rule(id="boom")], channels={"cap": CallableChannel(lambda a: None)},
            severity_routes={"info": ["cap"]}, state=StateStore(tmp / "s_boom.json"))
        fired = run_once(eng_boom, CallableSource(boom))
        check("数据源抛异常时返回空结果", fired == [])

        section("规则文件读写")
        rd = tmp / "rules"
        save_rule(rule(id="file_rule"), rd)
        loaded = load_rules(rd)
        check("落盘后能读回", len(loaded) == 1 and loaded[0].id == "file_rule")

        dup = tmp / "dup"
        dup.mkdir()
        (dup / "a.json").write_text(json.dumps({
            "id": "same", "name": "x", "symbols": ["A"],
            "conditions": [{"field": "c", "op": ">", "value": 1}]}), encoding="utf-8")
        (dup / "b.json").write_text(json.dumps({
            "id": "same", "name": "y", "symbols": ["A"],
            "conditions": [{"field": "c", "op": ">", "value": 1}]}), encoding="utf-8")
        try:
            load_rules(dup)
            check("重复 id 应报错", False)
        except RuleError:
            check("重复 id 报错", True)

        section("告警日志")
        log = tmp / "alerts.jsonl"
        eng5 = AlertEngine(
            rules=[rule(id="log_rule")],
            channels={"cap": CallableChannel(lambda a: None)},
            severity_routes={"info": ["cap"]},
            state=StateStore(tmp / "s_log.json"),
            alert_log=log,
        )
        eng5.evaluate({"A": {"close": 11}})
        line = json.loads(log.read_text(encoding="utf-8").strip())
        check("日志字段完整",
              line["rule_id"] == "log_rule" and line["symbol"] == "A" and "ts" in line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 50)
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    print("=" * 50)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
