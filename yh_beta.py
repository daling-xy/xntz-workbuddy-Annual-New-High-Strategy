# -*- coding: utf-8 -*-
"""
给事件表加上「事前 beta」，用于把「真选股能力」和「高 beta 放大」区分开。

动机：下跌市（尤其修复期）里深跌股票涨得猛，很可能只是 beta 放大，
      而不是真的选股能力强。用 fwd60 - beta * 市场同期收益 重算超额，
      才能判断「放宽 prox」到底是引入了 alpha 还是引入了 beta。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import CACHE_DIR  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BETA_W = 250


def build_market_ret(path=None) -> pd.Series:
    """等权全A日收益序列（用作市场基准）"""
    path = path or os.path.join(CACHE, "mkt_ret.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with open(os.path.join(CACHE_DIR, "kline_long2.pkl"), "rb") as f:
        kmap = pickle.load(f)

    axis = sorted({b["date"] for bars in kmap.values() for b in bars})
    aidx = pd.Index(axis)
    L = len(axis)
    s = np.zeros(L); c = np.zeros(L)

    for bars in kmap.values():
        if not bars or len(bars) < 30:
            continue
        pos = aidx.get_indexer([b["date"] for b in bars])
        if np.any(pos < 0):
            continue
        cl = np.array([float(b["close"]) for b in bars])
        r = np.empty(len(cl)); r[0] = np.nan
        r[1:] = cl[1:] / np.maximum(cl[:-1], 1e-9) - 1.0
        ok = ~np.isnan(r)
        if ok.sum() < 30:
            continue
        s[pos[ok]] += r[ok]
        c[pos[ok]] += 1
    nz = c > 0
    ret = np.full(L, np.nan)
    ret[nz] = s[nz] / c[nz]
    ser = pd.Series(ret, index=axis).sort_index()
    with open(path, "wb") as f:
        pickle.dump(ser, f)
    return ser


def add_beta(ev: pd.DataFrame) -> pd.DataFrame:
    """按事件所在位置，用过去 250 日日收益估计 beta"""
    path = os.path.join(CACHE, "events_long_beta.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    mkt = build_market_ret()
    mkt_map = {d: i for i, d in enumerate(mkt.index)}
    mv = mkt.values

    with open(os.path.join(CACHE_DIR, "kline_long2.pkl"), "rb") as f:
        kmap = pickle.load(f)

    out = []
    for code, sub in ev.groupby("code", sort=False):
        bars = kmap.get(code)
        if not bars:
            continue
        dates = [b["date"] for b in bars]
        pos = {}
        for i, d in enumerate(dates):
            pos[d] = i
        cl = np.array([float(b["close"]) for b in bars])
        r = np.empty(len(cl)); r[0] = np.nan
        r[1:] = cl[1:] / np.maximum(cl[:-1], 1e-9) - 1.0

        # 滚动 cov / var（与等权市场收益对齐）
        mi = np.array([mkt_map.get(d, -1) for d in dates])
        aligned = np.where(mi >= 0, mv[np.clip(mi, 0, len(mv) - 1)], np.nan)

        n = len(cl)
        valid = np.isfinite(r) & np.isfinite(aligned)
        rr = np.where(valid, r, 0.0)
        mm = np.where(valid, aligned, 0.0)
        cs_r = np.concatenate([[0.0], np.cumsum(rr)])
        cs_m = np.concatenate([[0.0], np.cumsum(mm)])
        cs_rm = np.concatenate([[0.0], np.cumsum(rr * mm)])
        cs_m2 = np.concatenate([[0.0], np.cumsum(mm * mm)])
        cs_r2 = np.concatenate([[0.0], np.cumsum(rr * rr)])

        beta = np.full(n, np.nan)
        for i in range(BETA_W - 1, n):
            j = i - BETA_W + 1
            k = i - j + 1
            s_r = cs_r[i + 1] - cs_r[j]
            s_m = cs_m[i + 1] - cs_m[j]
            s_rm = cs_rm[i + 1] - cs_rm[j]
            s_m2 = cs_m2[i + 1] - cs_m2[j]
            var_m = (s_m2 - s_m * s_m / k) / k
            if var_m > 1e-12:
                beta[i] = ((s_rm - s_r * s_m / k) / k) / var_m

        idx = np.array([pos[d] for d in sub["date"]], dtype=int)
        out.append(pd.DataFrame({"code": code, "date": sub["date"].values,
                                 "beta": beta[idx],
                                 "ret_std": np.nan}))

    bdf = pd.concat(out, ignore_index=True)
    res = ev.merge(bdf, on=["code", "date"], how="left")

    # 市场未来 60 日收益（按交易日）
    mkt_fwd60 = (mkt.shift(-60) / mkt - 1.0) if False else None
    cum = np.cumprod(1.0 + np.nan_to_num(mv, nan=0.0))
    fwd_map = {}
    for t in range(len(mv)):
        if t + 60 < len(mv):
            fwd_map[mkt.index[t]] = cum[t + 60] / cum[t] - 1.0
    res["mkt_fwd60"] = res["date"].map(fwd_map)
    res["ab_fwd60"] = res["fwd60"] - res["beta"] * res["mkt_fwd60"]
    res = res.replace([np.inf, -np.inf], np.nan)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


if __name__ == "__main__":
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_5.pkl"))
    st = pd.read_pickle(os.path.join(CACHE, "regime_state.pkl"))[["date", "regime"]]
    ev = ev.merge(st, on="date", how="inner")
    res = add_beta(ev)
    print("带 beta 的事件数:", len(res), " beta 缺失:", res["beta"].isna().sum())
    print(res["beta"].describe(percentiles=[.1, .5, .9]).round(3).to_string())
