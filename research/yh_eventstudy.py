# -*- coding: utf-8 -*-
"""
事件回放.getAll stocks × 所有历史时点 → 事件表，用于实证校准因子阈值。

核心方法：
  对每只股票的每个交易日 t（t 有完整 250 日前视窗口且保留足够后视天数），
  计算当日结构特征 + 未来 20/40/60 日收益 + 持有期内最大不利/有利幅度。

为避免时序区间整体牛熊的影响，所有收益都减去「同一交易日的全市场横截面均值」，
得到超额收益后再评估 —— 这才是因子真正的选股能力。
"""
from __future__ import annotations

import os
import pickle
import sys
from typing import Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import fetch_fundamentals, fetch_klines  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

HIGH_DAYS = 250
BOX_DAYS = 30
TIGHT_DAYS = 10
STEP = 5                      # 每 5 个交易日采样一次事件，降低重叠
MIN_BARS = 450                # 参与回放所需的最短历史长度


def roll_max(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.maximum.accumulate(x)
    sw = sliding_window_view(x, w).max(axis=1)
    out = np.empty(len(x), dtype=float)
    out[:w - 1] = np.maximum.accumulate(x)[:w - 1]
    out[w - 1:] = sw
    return out


def roll_min(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.minimum.accumulate(x)
    sw = sliding_window_view(x, w).min(axis=1)
    out = np.empty(len(x), dtype=float)
    out[:w - 1] = np.minimum.accumulate(x)[:w - 1]
    out[w - 1:] = sw
    return out


def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    s = np.maximum(0, idx - w + 1)
    return (csum[idx + 1] - csum[s]) / (idx - s + 1)


def build_event_table(days: int = 1100, step: int = STEP, force: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE, f"events_{days}_{step}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    fin_rows = fetch_fundamentals(report_dates=("2026-06-30", "2025-06-30", "2026-03-31", "2025-03-31"))
    codes = sorted({r["SECURITY_CODE"] for r in fin_rows})
    kmap = fetch_klines(codes, days=days, workers=32, cache_tag=str(days))

    recs = []
    for code in codes:
        bars = kmap.get(code)
        if not bars:
            continue
        n = len(bars)
        if n < MIN_BARS:
            continue
        dates = np.array([b["date"] for b in bars])
        c = np.array([b["close"] for b in bars], dtype=float)
        h = np.array([b["high"] for b in bars], dtype=float)
        lo = np.array([b["low"] for b in bars], dtype=float)
        amt = np.array([b.get("amount", 0.0) for b in bars], dtype=float)
        turn = np.array([b.get("turnover", 0.0) for b in bars], dtype=float)
        n = len(c)

        hi250 = roll_max(c, HIGH_DAYS)
        hi30 = roll_max(c, BOX_DAYS)
        lo30 = roll_min(c, BOX_DAYS)
        hi10 = roll_max(c, TIGHT_DAYS)
        lo10 = roll_min(c, TIGHT_DAYS)

        with np.errstate(divide="ignore", invalid="ignore"):
            width30 = hi30 / lo30 - 1.0
            width10 = hi10 / lo10 - 1.0
            prox = hi30 / hi250
            dd_from_high = c / hi250 - 1.0
            pos_in_box = (c - lo30) / np.where(hi30 > lo30, hi30 - lo30, np.nan)
            contraction = width10 / np.where(width30 > 0, width30, np.nan)

        amt30 = roll_mean(amt, BOX_DAYS)
        amt10 = roll_mean(amt, TIGHT_DAYS)
        turn30 = roll_mean(turn, BOX_DAYS)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_dry = amt10 / np.where(amt30 > 0, amt30, np.nan)

        ma20 = roll_mean(c, 20); ma60 = roll_mean(c, 60); ma120 = roll_mean(c, 120)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret20 = c / np.roll(c, 20) - 1.0
            ret60 = c / np.roll(c, 60) - 1.0
            ret250 = c / np.roll(c, 250) - 1.0
        ret20[:20] = np.nan; ret60[:60] = np.nan; ret250[:250] = np.nan

        # 自身过去一年的 30 日宽度分布 → 当日在其中的分位（只对采样点计算）
        width_pct = np.full(n, np.nan)
        _samp = np.arange(HIGH_DAYS, max(HIGH_DAYS, n - 60))
        _samp = _samp[_samp % step == 0]
        for i in _samp:
            j = max(0, i - HIGH_DAYS + 1)
            hist = width30[j: i + 1]
            hist = hist[np.isfinite(hist)]
            if len(hist) >= 60:
                width_pct[i] = float((hist >= width30[i]).mean())

        def fwd(k):
            out = np.full(n, np.nan)
            if n > k:
                out[:n - k] = c[k:] / c[:n - k] - 1.0
            return out

        f20, f40, f60 = fwd(20), fwd(40), fwd(60)
        mae = np.full(n, np.nan); mfe = np.full(n, np.nan)
        rmin_lo20 = roll_min(lo, 20); rmax_hi20 = roll_max(h, 20)
        rmin_lo60 = roll_min(lo, 60); rmax_hi60 = roll_max(h, 60)
        mae20 = np.full(n, np.nan); mfe60 = np.full(n, np.nan)
        mae20[:n - 20] = rmin_lo20[20:] / c[:n - 20] - 1.0
        mfe60[:n - 60] = rmax_hi60[60:] / c[:n - 60] - 1.0

        idxs = np.arange(HIGH_DAYS, n - 60)
        idxs = idxs[idxs % step == 0]
        for i in idxs:
            if not np.isfinite(f20[i]):
                continue
            recs.append((
                code, dates[i], c[i],
                width30[i], width10[i], prox[i], dd_from_high[i], pos_in_box[i],
                contraction[i], width_pct[i], amt30[i], turn30[i], vol_dry[i],
                ret20[i], ret60[i], ret250[i], ma20[i], ma60[i], ma120[i],
                f20[i], f40[i], f60[i], mae20[i], mfe60[i],
            ))

    cols = ["code", "date", "close", "width30", "width10", "prox", "dd_from_high",
            "pos_in_box", "contraction", "width_pct", "amt30", "turn30", "vol_dry",
            "ret20", "ret60", "ret250", "ma20", "ma60", "ma120",
            "fwd20", "fwd40", "fwd60", "mae20", "mfe60"]
    df = pd.DataFrame(recs, columns=cols)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 减去当日横截面均值 → 超额收益
    for k in ["fwd20", "fwd40", "fwd60"]:
        df[f"ex_{k}"] = df[k] - df.groupby("date")[k].transform("mean")
    df["MA20_above"] = (df["close"] > df["ma20"]).astype(float)

    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


if __name__ == "__main__":
    df = build_event_table()
    print("事件数:", len(df), " 股票数:", df["code"].nunique(),
          " 时间跨度:", df["date"].min(), "~", df["date"].max())
    print(df[["width30", "prox", "amt30", "turn30", "width_pct", "contraction"]].describe(
        percentiles=[.1, .5, .9]).T.to_string())
    print()
    for k in ["fwd20", "fwd40", "fwd60"]:
        print(k, "均值 %.4f 中位 %.4f 胜率 %.3f" %
              (df[k].mean(), df[k].median(), (df[k] > 0).mean()))
