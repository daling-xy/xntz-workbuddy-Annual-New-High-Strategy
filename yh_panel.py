# -*- coding: utf-8 -*-
"""
构建「截面面板」：把 K 线、快照、基本面拼成一张表，供筛选 / 校准 / 回测复用。
"""
from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import fetch_fundamentals, fetch_klines, fetch_snapshots  # noqa: E402
from yh_factors import features_at  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# 2026-10-27 起纳入三季报（2026-09-30，法定披露日 10-31 前陆续出齐）。
# 注意：latest/prev 取的是「最新一期」与「紧邻上一期」（非去年同期），
#       未披露三季报的公司会自动回落到中报 / 一季报，语义保持一致。
#
# 2026-10-27 二改：补入 2025-09-30 / 2025-12-31，用于构造 TTM（最近四个季度）营收。
#   原因：TOTAL_OPERATE_INCOME 是【年初至今累计】，三季报=3个季度、中报=2个季度，
#         跨报告期不可比（详见 README 3.4）。TTM = 本期累计 + 上年全年 − 上年同期累计，
#         无论哪一期都覆盖最近 12 个月，口径恒定。
#   说明：补入的两期排在 latest 之前，不改变 latest / prev 的选取语义。
REPORT_DATES = ("2026-09-30", "2026-06-30", "2025-12-31", "2025-09-30",
                "2025-06-30", "2026-03-31", "2025-03-31")

DEFAULT_PARAMS = {
    "high_days": 250,     # 一年（250 交易日）
    "box_days": 30,       # 箱体窗口
    "tight_days": 10,     # 末端收缩窗口
}


def _num(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _ttm_revenue(df: pd.DataFrame, codes: pd.Index) -> pd.Series:
    """构造 TTM（最近四个季度/最近一年）营收。

    TTM(期 t) = 累计(t) + 累计(上年 12-31) − 累计(上年同期)
    年报(12-31)本身即全年，直接取用。
    任一项缺失 → NaN（不猜测、不回退到累计口径，避免悄悄引入污染）。
    """
    d = df[["SECURITY_CODE", "REPORTDATE", "TOTAL_OPERATE_INCOME"]].copy()
    d["REPORTDATE"] = pd.to_datetime(d["REPORTDATE"])
    d["v"] = pd.to_numeric(d["TOTAL_OPERATE_INCOME"], errors="coerce")
    d = d.dropna(subset=["v"])
    cum = d.set_index(["SECURITY_CODE", "REPORTDATE"])["v"]
    # 2026-10-28 修复：财报源偶发「同票同报告期」重复行，会让 cum.get((code, date))
    # 返回 Series 而非标量，下游 np.isnan(Series) 直接抛 ValueError。
    # 按索引去重（保留最后一行=最新披露口径），并保证索引有序。
    cum = cum[~cum.index.duplicated(keep="last")].sort_index()

    out = {}
    for code in codes:
        try:
            sub = cum.loc[code]
        except KeyError:
            out[code] = np.nan
            continue
        if isinstance(sub, pd.Series):
            sub = sub.sort_index()
            rd = sub.index[-1]
        else:                       # 只有一期
            out[code] = np.nan
            continue
        if rd.month == 12 and rd.day == 31:
            out[code] = float(sub.iloc[-1])
            continue
        prev_same = pd.Timestamp(year=rd.year - 1, month=rd.month, day=rd.day)
        prev_q4 = pd.Timestamp(year=rd.year - 1, month=12, day=31)
        a = cum.get((code, prev_same), np.nan)
        b = cum.get((code, prev_q4), np.nan)
        out[code] = float(sub.iloc[-1] + b - a) if (not np.isnan(a) and not np.isnan(b)) else np.nan
    return pd.Series(out, index=codes, dtype=float)


def build_fundamentals_frame() -> pd.DataFrame:
    """把多期财报压平成每票一行的最新 + 同比多数 saat"""
    rows = fetch_fundamentals(report_dates=REPORT_DATES)
    df = pd.DataFrame(rows)
    df["REPORTDATE"] = pd.to_datetime(df["REPORTDATE"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["SECURITY_CODE", "REPORTDATE"])
    latest = df.groupby("SECURITY_CODE").tail(1).copy()
    latest = latest.set_index("SECURITY_CODE")

    # 最近两期（用于判断增长是否连续）
    prev = df.groupby("SECURITY_CODE").nth(-2).set_index("SECURITY_CODE")

    out = pd.DataFrame(index=latest.index)
    out["name_fin"] = latest["SECURITY_NAME_ABBR"]
    out["board"] = latest["BOARD_NAME"]
    out["market"] = latest["TRADE_MARKET"]
    out["report_date"] = latest["REPORTDATE"]
    out["revenue"] = latest["TOTAL_OPERATE_INCOME"].map(_num)
    # TTM（最近四个季度）营收：口径恒定，消除「累计数跨报告期不可比」的污染。
    # TTM = 本期累计 + 上年全年累计 − 上年同期累计；若本期就是年报，则累计即全年。
    out["revenue_ttm"] = _ttm_revenue(df, latest.index)
    out["rev_yoy"] = latest["YSTZ"].map(_num)
    out["profit"] = latest["PARENT_NETPROFIT"].map(_num)
    out["profit_yoy"] = latest["SJLTZ"].map(_num)
    out["roe_w"] = latest["WEIGHTAVG_ROE"].map(_num)
    out["gross_margin"] = latest["XSMLL"].map(_num)
    out["mgjyxjje"] = latest["MGJYXJJE"].map(_num)   # 每股经营现金流

    out["rev_yoy_prev"] = prev["YSTZ"].map(_num)
    out["profit_yoy_prev"] = prev["SJLTZ"].map(_num)
    out["growth_persist"] = ((out["rev_yoy"] > 0) & (out["rev_yoy_prev"] > 0)).astype(int)
    out["grow_accel"] = out["rev_yoy"] - out["rev_yoy_prev"]
    # 营收绝对规模（元）与是否亏损
    out["is_loss"] = (out["profit"] < 0).astype(int)
    return out


def build_panel(asof_offset: int = 1, days: int = 320,
                params: Optional[Dict] = None, force: bool = False) -> pd.DataFrame:
    """
    asof_offset=1 表示以「倒数第 2 根 K 线」为评估点，即跳过当日盘中未完成的那根 bar。
    若想用最新（含当日盘中），传 asof_offset=0。
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    tag = f"panel_{days}_{asof_offset}.pkl"
    path = os.path.join(CACHE, tag)
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    fin = build_fundamentals_frame()
    codes = sorted(fin.index.tolist())
    kmap = fetch_klines(codes, days=days, workers=32, cache_tag=str(days))
    snap = fetch_snapshots(codes)

    recs = []
    for code in codes:
        bars = kmap.get(code)
        if not bars:
            continue
        idx = len(bars) - 1 - asof_offset
        if idx < p["high_days"]:
            continue
        feats = features_at(bars, p, idx=idx)
        if not feats:
            continue
        # 停牌过滤：近 5 日有成交额为 0 的
        recent_amt = [b.get("amount", 0) for b in bars[idx - 4: idx + 1]]
        if min(recent_amt) <= 0:
            continue
        # 今日（最后一根）用于触发检查
        last_bar = bars[-1]
        feats.update({
            "code": code,
            "bars": len(bars),
            "snapshot_time": last_bar["date"],
            "last_price": last_bar["close"],
            "today_pct": last_bar["close"] / bars[-2]["close"] - 1.0 if len(bars) > 1 else np.nan,
        })
        s = snap.get(code, {})
        feats.update({
            "name": s.get("name") or fin.at[code, "name_fin"] if code in fin.index else s.get("name"),
            "float_mv": s.get("float_mv", np.nan),
            "total_mv": s.get("total_mv", np.nan),
            "pe_ttm": s.get("pe_ttm", np.nan),
            "pb": s.get("pb", np.nan),
        })
        recs.append(feats)

    panel = pd.DataFrame(recs).set_index("code")
    panel = panel.join(fin, how="left")
    panel["name"] = panel["name"].fillna(panel.get("name_fin"))
    if "name_fin" in panel:
        panel = panel.drop(columns=["name_fin"])

    with open(path, "wb") as f:
        pickle.dump(panel, f)
    return panel


if __name__ == "__main__":
    p = build_panel(asof_offset=1, days=320)
    print("面板股票数:", len(p))
    print("评估基准日:", p["date"].max(), " 快照日:", p["snapshot_time"].max())
    cols = ["width30", "prox", "dd_from_high", "amt30", "turn30", "contraction", "width_pct", "atr_pct"]
    print(p[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.to_string())
