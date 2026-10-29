# -*- coding: utf-8 -*-
"""
【每日选股一键入口】

    python run_daily.py                 只用本地最新缓存跑（秒级出结果，不联网）
    python run_daily.py --fresh         先刷新行情/快照/指数/宽度，再跑（联网，约 3~8 分钟）
    python run_daily.py --fresh --fin   强制连财报一起刷新（通常无需手动加，见下）

它做的事按顺序是：
    1) 检查各项缓存的新鲜度，打印出来，让你知道现在算的是哪一天的数据
    2) --fresh 时按依赖顺序清缓存并重建：
         财报（视披露窗口自动增量）→ 行情/快照 → 市场宽度 breadth
         → 周期标签 regime → 截面面板 panel
       注：处于定期报告【陆续披露窗口】（截至法定披露截止日 +3 天）时，每天都有新公司
           披露，此时财报会被【自动强制增量重拉】，无需加 --fin，
           以免个股口径随时间机械性翻转。
    3) 调用 run_screen_v2.main() 输出当日候选池、观察池、持仓诊断、严格度阶梯

不刷新的东西（以及为什么）：
    · kline_long2.pkl（2018 至今全历史）：重拉一次要很久，每季度跑一次 yh_fetch_long2.py 即可
    · fin_*.pkl（财报）：默认自动判断——披露窗口内每日增量刷新，窗口结束则沿用缓存
    · events_*.pkl（回测事件表）：只在重新做参数研究时才需要重建

数据来源：腾讯前复权日线与换手率／通达信五大指数日线／东方财富财报与快照。
以上均为统计回测结论，不构成投资建议。
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


def _p(name: str) -> str:
    return os.path.join(CACHE, name)


def _last_date(name: str) -> str:
    """读一个缓存文件里最新数据的日期，用于新鲜度检查"""
    path = _p(name)
    if not os.path.exists(path):
        return "（无缓存）"
    try:
        with open(path, "rb") as f:
            o = pickle.load(f)
    except Exception as e:
        return f"（读取失败 {e}）"
    if isinstance(o, pd.DataFrame):
        col = "date" if "date" in o.columns else None
        if col is None:
            return f"DataFrame {len(o)} 行（无 date 列）"
        return f"{o[col].min()} ~ {o[col].max()}"
    if isinstance(o, dict):
        for v in o.values():
            if isinstance(v, list) and v and isinstance(v[-1], dict) and "date" in v[-1]:
                return f"{v[-1]['date']}（{len(o)} 只票）"
        return f"dict {len(o)} 项"
    if isinstance(o, list):
        return f"list {len(o)} 条"
    return str(type(o))


def fin_items() -> tuple:
    """当前 REPORT_DATES 对应的财报缓存文件名。

    从 yh_panel.REPORT_DATES 派生，不再写死 —— 之前写死导致改动报告期后
    --fin 删不到真正的缓存，刷新形同虚设。
    """
    from yh_panel import REPORT_DATES
    from yh_datasource import fin_cache_name, fin_still_disclosing
    name = fin_cache_name(REPORT_DATES)
    rolling = fin_still_disclosing(REPORT_DATES)
    return name, rolling


def freshness() -> pd.DataFrame:
    from yh_panel import REPORT_DATES
    from yh_datasource import fin_cache_name, fin_still_disclosing
    items = [
        ("日线（320日）", "kline_320.pkl"),
        ("实时快照", "snapshot.pkl"),
        ("截面面板", "panel_320_1.pkl"),
        ("市场宽度", "breadth_long_60.pkl"),
        ("周期标签", "regime_state.pkl"),
        ("指数面板", "index_panel.pkl"),
        ("财报", fin_cache_name(REPORT_DATES)),
        ("全历史日线", "kline_long2.pkl"),
    ]
    rows = []
    for tag, f in items:
        path = _p(f)
        mt = (time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
              if os.path.exists(path) else "—")
        rows.append({"数据": tag, "文件": f, "覆盖范围": _last_date(f), "本地更新于": mt})
    return pd.DataFrame(rows)


def _rm(*names: str) -> None:
    for n in names:
        p = _p(n)
        if os.path.exists(p):
            os.remove(p)
            print(f"  已清除缓存 {n}")


def refresh(fin: bool = False, args_idx: bool = False, asof: int = 1) -> None:
    print("=" * 100)
    print("刷新数据")
    print("=" * 100)

    # ---- 0. 财报（必须在[1]之前清缓存，否则面板会被重复构建两次）----
    # 处于定期报告的【陆续披露窗口】时，每天都有新公司披露，
    # 不每天增量重拉，个股口径会随时间机械性翻转 → 此处自动强制刷新。
    from yh_panel import REPORT_DATES
    from yh_datasource import fin_cache_name, fin_still_disclosing
    fin_name = fin_cache_name(REPORT_DATES)
    rolling = fin_still_disclosing(REPORT_DATES)
    if rolling and not fin:
        print(f"[0/4] {', '.join(rolling)} 仍处披露窗口内（截止日前后）"
              f"→ 自动增量刷新财报")
        fin = True
    if fin:
        print(f"[0/4] 清除财报缓存 {fin_name} ...")
        _rm(fin_name)
    else:
        print(f"[0/4] 财报沿用缓存（{fin_name}），无报告期处披露窗口")

    # ---- 1. 行情与快照（决定 panel 里的价格、箱宽、换手、成交额）----
    print("[1/4] 重拉全 A 日线 + 快照 ...")
    _rm("kline_320.pkl", "snapshot.pkl", "panel_320_1.pkl")
    from yh_datasource import fetch_klines, fetch_snapshots
    from yh_panel import build_fundamentals_frame
    fin_df = build_fundamentals_frame()
    codes = sorted(fin_df.index.tolist())
    kmap = fetch_klines(codes, days=320, workers=32, cache_tag="320")
    print(f"  日线 {len(kmap)} 只；最新交易日 = "
          f"{max(v[-1]['date'] for v in kmap.values() if v)}")
    fetch_snapshots(codes)

    # ---- 2. 指数面板（通达信五大指数，周期判定的观察字段）----
    # 注意：周期判定（UP/CHOP/DOWN）只依赖市场宽度 breadth，指数只是观察字段，
    # 所以这里拉不到指数时要能降级，不能中断整个刷新流程。
    print("[2/4] 重拉通达信指数 ...")
    from yh_regime_state import index_panel, build_breadth, assign_state
    ipx_old_path = _p("index_panel.pkl")
    ipx_old = None
    if os.path.exists(ipx_old_path):
        try:
            with open(ipx_old_path, "rb") as f:
                ipx_old = pickle.load(f)
        except Exception:
            ipx_old = None
    if args_idx:
        _rm("tdx_idx_000001.pkl", "tdx_idx_000300.pkl", "tdx_idx_000852.pkl",
            "tdx_idx_000905.pkl", "tdx_idx_399006.pkl")
    try:
        ipx = index_panel(force=True)
        print(f"  指数面板 {len(ipx)} 行，最新 {ipx['date'].max()}")
    except Exception as e:
        print(f"  [降级] 通达信指数拉取失败：{e}")
        if ipx_old is not None:
            ipx = ipx_old
            print(f"  已沿用旧指数面板，最新 {ipx['date'].max()}"
                  f"（指数仅作观察字段，不影响周期判定）")
        else:
            print("  无旧缓存可用，本轮周期标签不含指数观察字段。")
            ipx = None

    # ---- 3. 市场宽度 + 周期标签 ----
    # 长历史（2018 至今）不重拉，只用新拉的 320 天把尾部覆盖掉，省时间。
    print("[3/4] 重算市场宽度与周期标签 ...")
    br_new = build_breadth("320", 60, force=True)
    old_path = _p("breadth_long_60.pkl")
    if os.path.exists(old_path):
        with open(old_path, "rb") as f:
            br_old = pickle.load(f)
        cut = br_new["date"].min()
        br = pd.concat([br_old[br_old["date"] < cut], br_new], ignore_index=True)
        print(f"  宽度：长历史保留到 {cut} 之前，之后用新数据覆盖；合计 {len(br)} 天")
    else:
        br = br_new
    merged = br if ipx is None else br.merge(ipx, on="date", how="left")
    # 指数面板可能滞后一天（通达信接口偶发不可用），观察字段向前填充即可；
    # 周期标签本身只用 breadth，不受影响。
    for c in merged.columns:
        if c.endswith(("_dd250", "_pos", "_slope", "_ret60")):
            merged[c] = merged[c].ffill()
    st = assign_state(merged)
    st.to_pickle(_p("regime_state.pkl"))
    with open(old_path, "wb") as f:
        pickle.dump(br, f)
    print(f"  周期标签最新：{st['date'].max()} → {st['regime'].iloc[-1]}")

    # ---- 4. 重建截面面板 ----
    print("[4/4] 重建截面面板 ...")
    from yh_panel import build_panel
    panel = build_panel(asof_offset=asof, days=320, force=True)
    print(f"  面板 {len(panel)} 只，评估基准日 {panel['date'].max()}，"
          f"快照日 {panel['snapshot_time'].max()}")

    if rolling:
        print(f"  注：{', '.join(rolling)} 仍在披露窗口，财报已按最新口径刷新。")
    print()


PROBE_CODES = ("600030", "000001", "600519", "688018", "300209")


def probe() -> int:
    """只探测数据源最新交易日，不刷新不重算。

    用途：收盘后先确认当天数据是否已经出来，避免白跑一次 --fresh（约 2.5 分钟）。
    返回：数据源最新交易日相对本地缓存是否更新（0=有更新或未知，1=无更新）
    """
    import datetime as _dt
    from yh_datasource import fetch_kline

    print("=" * 90)
    print("数据源探测（只探测，不刷新）")
    print("=" * 90)
    print(f"当前系统时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    latest = []
    for c in PROBE_CODES:
        try:
            b = fetch_kline(c, days=6)
            if b:
                latest.append(b[-1]["date"])
                print(f"  {c}  最新交易日 = {b[-1]['date']}  收盘 = {b[-1]['close']}")
            else:
                print(f"  {c}  拉取失败")
        except Exception as e:
            print(f"  {c}  异常 {type(e).__name__}: {e}")
    print()

    if not latest:
        print("  ⚠ 全部拉取失败，无法判定数据源状态")
        return 1

    src = max(latest)
    # 本地面板缓存的最新日期
    try:
        pn = pd.read_pickle(_p("panel_320_1.pkl"))
        local = str(pd.to_datetime(pn["date"]).max().date())
    except Exception:
        local = "(未知)"

    print(f"  数据源最新交易日 = {src}")
    print(f"  本地面板最新     = {local}")
    print()
    if local != "(未知)" and src <= local:
        print(f"  → 数据源尚未超出本地缓存（{src} ≤ {local}），当天数据可能还没出，")
        print("    建议稍等几分钟再跑 --fresh，或直接跑（会用最新可用数据）。")
        return 1
    print(f"  → 数据源已有新数据（{src} > {local}），可以跑 --fresh 刷新。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="年度新高策略 · 每日选股一键入口")
    ap.add_argument("--fresh", action="store_true", help="先刷新行情/快照/指数/宽度再跑")
    ap.add_argument("--fin", action="store_true", help="连同财报一起刷新（需配合 --fresh）")
    ap.add_argument("--idx", action="store_true",
                    help="强制重拉通达信五大指数（默认沿用缓存；指数仅作观察字段）")
    ap.add_argument("--check", action="store_true",
                    help="只探测数据源最新交易日，不刷新不重算")
    ap.add_argument("--asof", type=int, default=1, choices=[0, 1],
                    help="结构评估基准日偏移：1=倒数第2根K线（盘中安全，默认）；"
                         "0=最新一根K线（收盘后用，当天bar已定型）")
    args = ap.parse_args()

    print("=" * 100)
    print("年度新高策略 · 每日选股")
    print("=" * 100)
    print(f"运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    if args.check:
        raise SystemExit(probe())

    print("【刷新前的数据新鲜度】")
    print(freshness().to_string(index=False))
    print()

    if args.fresh:
        refresh(fin=args.fin, args_idx=args.idx, asof=args.asof)
        print("【刷新后的数据新鲜度】")
        print(freshness().to_string(index=False))
        print()
    else:
        print("※ 未加 --fresh，直接使用本地缓存。若要跑今天的行情，请改用：")
        print("     python run_daily.py --fresh")
        print()

    # 把基准日偏移透传给筛选脚本，确保「刷新」与「筛选」用的是同一个评估日
    os.environ["YH_ASOF"] = str(args.asof)

    import run_screen_v2
    run_screen_v2.main()


if __name__ == "__main__":
    main()
