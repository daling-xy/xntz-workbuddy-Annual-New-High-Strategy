# -*- coding: utf-8 -*-
"""年度新高策略 v3 · 每日入口
================================================================
用法：
  python run_daily.py --check          # 只探测数据源最新交易日（不刷新）
  python run_daily.py --fresh          # 刷新行情/快照/board → 跑 v3 选股
  python run_daily.py                  # 不刷新，直接用本地缓存跑 v3
  python run_daily.py --fresh --asof 0 # 收盘后（当天 bar 已定型）；默认 asof=1 盘中安全
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
sys.path.insert(0, BASE)
from yh_datasource import fetch_klines, fetch_snapshots, fetch_fundamentals, fetch_kline  # noqa: E402

REPORT_DATES = ("2026-09-30", "2026-06-30", "2025-12-31", "2025-09-30",
                "2025-06-30", "2026-03-31", "2025-03-31")
PROBE_CODES = ("600030", "000001", "600519", "688018", "300209")


def probe() -> int:
    print("=" * 80)
    print("数据源探测（只探测，不刷新）")
    print("=" * 80)
    latest = []
    for c in PROBE_CODES:
        try:
            b = fetch_kline(c, days=6)
            if b:
                latest.append(b[-1]["date"])
                print(f"  {c}  最新交易={b[-1]['date']}  收盘={b[-1]['close']}")
        except Exception as e:  # noqa: BLE001
            print(f"  {c}  失败 {type(e).__name__}")
    if not latest:
        print("  ⚠ 全部失败，无法判定")
        return 1
    src = max(latest)
    try:
        kl = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))
        local = max(v[-1]["date"] for v in kl.values() if v)
    except Exception:  # noqa: BLE001
        local = "(未知)"
    print(f"  数据源最新交易日 = {src}")
    print(f"  本地日线最新     = {local}")
    if local != "(未知)" and src <= local:
        print("  → 数据源未超出本地缓存，稍后再跑或直接用本地缓存")
        return 1
    print("  → 数据源已有新数据，可以 --fresh")
    return 0


def build_board_map(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("财报数据为空，无法生成 board 映射")
    df = df.sort_values(["SECURITY_CODE", "REPORTDATE"])
    latest = df.groupby("SECURITY_CODE").tail(1)
    out = latest[["SECURITY_CODE", "BOARD_NAME"]].rename(
        columns={"SECURITY_CODE": "code", "BOARD_NAME": "board"})
    out["code"] = out["code"].astype(str)
    out = out.drop_duplicates("code")
    os.makedirs(CACHE, exist_ok=True)
    out.to_csv(os.path.join(CACHE, "board_map.csv"), index=False, encoding="utf-8-sig")
    return out


def refresh() -> None:
    t0 = time.time()
    print("[1/3] 拉取业绩报表（股票清单 + board）...")
    rows = fetch_fundamentals(REPORT_DATES)
    df = pd.DataFrame(rows)
    codes = sorted(df["SECURITY_CODE"].astype(str).unique())
    bm = build_board_map(rows)
    print(f"  股票 {len(codes)} 只；board 映射 {len(bm)} 行 -> cache/board_map.csv")

    print("[2/3] 拉取日线（320 日）...")
    kmap = fetch_klines(codes, days=320, workers=32, cache_tag="320")
    latest = max(v[-1]["date"] for v in kmap.values() if v)
    print(f"  日线 {len(kmap)} 只，最新交易日 = {latest}")

    print("[3/3] 拉取实时快照 ...")
    try:
        fetch_snapshots(codes)
    except Exception as e:  # noqa: BLE001
        print(f"  [降级] 快照拉取失败（{type(e).__name__}），v3 选股不依赖快照")
    print(f"刷新完成 [{time.time() - t0:.0f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description="年度新高策略 v3 · 每日入口")
    ap.add_argument("--fresh", action="store_true", help="先刷新行情/财报/快照，再跑选股")
    ap.add_argument("--check", action="store_true", help="只探测数据源最新交易日")
    ap.add_argument("--asof", type=int, default=1, choices=[0, 1],
                    help="基准日偏移：1=倒数第2根K线（盘中安全，默认）；0=最新一根（收盘后用）")
    args = ap.parse_args()

    print("=" * 80)
    print("年度新高策略 v3 · 每日选股")
    print("=" * 80)
    print(f"运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.check:
        raise SystemExit(probe())

    if args.fresh:
        refresh()
    else:
        print("※ 未加 --fresh：直接使用本地缓存（要跑今天行情请加 --fresh）")

    os.environ["YH_ASOF"] = str(args.asof)
    sys.path.insert(0, os.path.join(BASE, "v3"))
    import daily_rank_v3  # noqa: E402
    daily_rank_v3.main()


if __name__ == "__main__":
    main()
