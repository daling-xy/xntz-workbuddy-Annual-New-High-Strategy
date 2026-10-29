# -*- coding: utf-8 -*-
"""
长历史事件回放（2018-2026，腾讯按时段分页拼接的全历史）。

相比原 yh_eventstudy（有效样本被 641 根上限压缩在 2025-2026 两年），
本表覆盖 2020 疫情、2021 结构牛、2022 熊市、2024 两次极端行情、2026 当前下跌，
使得「按中级市场周期分层」的统计第一次成为可能。

因子定义与 v1 完全一致，便于对照；成交额/换手率均为原值。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import CACHE_DIR  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

HIGH_DAYS = 250
BOX_DAYS = 30
TIGHT_DAYS = 10
STEP = 5
MIN_BARS = 450


def roll_max(x, w):
    if len(x) < w:
        return np.maximum.accumulate(x)
    sw = sliding_window_view(x, w).max(axis=1)
    out = np.empty(len(x), dtype=float)
    out[:w - 1] = np.maximum.accumulate(x)[:w - 1]
    out[w - 1:] = sw
    return out


def roll_min(x, w):
    if len(x) < w:
        return np.minimum.accumulate(x)
    sw = sliding_window_view(x, w).min(axis=1)
    out = np.empty(len(x), dtype=float)
    out[:w - 1] = np.minimum.accumulate(x)[:w - 1]
    out[w - 1:] = sw
    return out


def roll_mean(x, w):
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    s = np.maximum(0, idx - w + 1)
    return (csum[idx + 1] - csum[s]) / (idx - s + 1)


def load_long():
    p = os.path.join(CACHE_DIR, "kline_long2.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError("先运行 yh_fetch_long2.py")
    with open(p, "rb") as f:
        return pickle.load(f)


def build_event_table(step: int = STEP, force: bool = False,
                      codes: list | None = None) -> pd.DataFrame:
    path = os.path.join(CACHE, f"events_long_{step}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    kmap = load_long()
    if codes:
        kmap = {c: kmap[c] for c in codes if c in kmap}

    recs = []
    for code, bars in kmap.items():
        n = len(bars)
        if n < MIN_BARS:
            continue
        dates = np.array([b["date"] for b in bars])
        c = np.array([b["close"] for b in bars], dtype=float)
        h = np.array([b["high"] for b in bars], dtype=float)
        lo = np.array([b["low"] for b in bars], dtype=float)
        amt = np.array([b.get("amount", 0.0) for b in bars], dtype=float)
        turn = np.array([b.get("turnover", 0.0) for b in bars], dtype=float)

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
            base250 = roll_mean(amt, 250)
            amt_act = amt30 / np.where(base250 > 0, base250, np.nan)

        ma20 = roll_mean(c, 20); ma60 = roll_mean(c, 60); ma120 = roll_mean(c, 120)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret20 = c / np.roll(c, 20) - 1.0
            ret60 = c / np.roll(c, 60) - 1.0
            ret250 = c / np.roll(c, 250) - 1.0
        ret20[:20] = np.nan; ret60[:60] = np.nan; ret250[:250] = np.nan

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
        mae20 = np.full(n, np.nan); mfe60 = np.full(n, np.nan)
        mae20[:n - 20] = roll_min(lo, 20)[20:] / c[:n - 20] - 1.0
        mfe60[:n - 60] = roll_max(h, 60)[60:] / c[:n - 60] - 1.0

        idxs = np.arange(HIGH_DAYS, n - 60)
        idxs = idxs[idxs % step == 0]
        for i in idxs:
            if not np.isfinite(f20[i]):
                continue
            recs.append((
                code, dates[i], c[i],
                width30[i], width10[i], prox[i], dd_from_high[i], pos_in_box[i],
                contraction[i], width_pct[i], amt30[i], turn30[i], vol_dry[i], amt_act[i],
                ret20[i], ret60[i], ret250[i], ma20[i], ma60[i], ma120[i],
                f20[i], f40[i], f60[i], mae20[i], mfe60[i],
            ))

    cols = ["code", "date", "close", "width30", "width10", "prox", "dd_from_high",
            "pos_in_box", "contraction", "width_pct", "amt30", "turn30", "vol_dry", "amt_act",
            "ret20", "ret60", "ret250", "ma20", "ma60", "ma120",
            "fwd20", "fwd40", "fwd60", "mae20", "mfe60"]
    df = pd.DataFrame(recs, columns=cols)
    df = df.replace([np.inf, -np.inf], np.nan)

    for k in ["fwd20", "fwd40", "fwd60"]:
        df[f"ex_{k}"] = df[k] - df.groupby("date")[k].transform("mean")

    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


if __name__ == "__main__":
    df = build_event_table()
    print("事件数:", len(df), " 股票数:", df["code"].nunique(),
          " 时间:", df["date"].min(), "~", df["date"].max())
    y = df.copy(); y["year"] = y["date"].str[:4]
    print(y.groupby("year").agg(事件数=("code", "size"), 股票数=("code", "nunique"),
                                交易日=("date", "nunique")).to_string())
