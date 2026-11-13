# -*- coding: utf-8 -*-
"""命令行入口。

    python -m reminder run       --config reminder\\config.json
    python -m reminder once      --config reminder\\config.json
    python -m reminder validate  --rules  reminder\\rules
    python -m reminder demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import build_channels, build_source, load_json, resolve_paths
from .engine import AlertEngine, StateStore, run_once, run_poll
from .models import RuleError, describe, load_rules


def _build_engine(cfg_path: Path):
    cfg = load_json(cfg_path)
    base = cfg_path.parent
    rules_dir, state_file, alert_log = resolve_paths(cfg, base)

    rules = load_rules(rules_dir)
    channels = build_channels(cfg)
    state = StateStore(state_file)
    engine = AlertEngine(
        rules=rules,
        channels=channels,
        severity_routes=cfg.get("severity_routes"),
        state=state,
        alert_log=alert_log,
    )
    return cfg, engine


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg, engine = _build_engine(cfg_path)
    source = build_source(cfg)
    interval = float(args.interval if args.interval is not None else cfg.get("poll_interval", 30))
    print(f"规则 {len(engine.rules)} 条｜渠道 {list(engine.channels)}｜间隔 {interval}s")
    print("按 Ctrl+C 停止。")
    try:
        run_poll(engine, source, interval=interval, max_ticks=args.ticks)
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        engine.state.save()
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg, engine = _build_engine(cfg_path)
    source = build_source(cfg)
    fired = run_once(engine, source)
    print(f"本轮触发 {len(fired)} 条告警。")
    engine.state.save()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    rules_dir = Path(args.rules)
    try:
        rules = load_rules(rules_dir)
    except RuleError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    if not rules:
        print(f"（{rules_dir} 下没有规则文件）")
        return 0
    for r in rules:
        flag = "启用" if r.enabled else "停用"
        print(f"[{flag}] {describe(r)}")
    print(f"\n共 {len(rules)} 条规则，全部通过校验。")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    return run_demo()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reminder", description="规则化实时提醒（数据源可插拔）")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, help_text in (
        ("run", cmd_run, "常驻轮询，触发即推送"),
        ("once", cmd_once, "只跑一轮（适合 cron / 计划任务）"),
        ("validate", cmd_validate, "校验规则文件"),
        ("demo", cmd_demo, "用内置假数据跑一遍，验证整条链路"),
    ):
        sp = sub.add_parser(name, help=help_text)
        if name in ("run", "once"):
            sp.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
            sp.add_argument("--interval", type=float, default=None, help="轮询间隔秒，覆盖配置")
            sp.add_argument("--ticks", type=int, default=None, help="跑多少轮后退出（调试用）")
        if name == "validate":
            sp.add_argument("--rules", default=str(Path(__file__).with_name("rules")))
        sp.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
