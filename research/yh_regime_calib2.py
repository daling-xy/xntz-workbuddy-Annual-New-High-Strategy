# -*- coding: utf-8 -*-
"""
周期分层校准 II：把「真 alpha」与「beta 放大」分开，产出最终可直接落地的决策表。

上一版发现：下跌市里放宽 prox 似乎更好。但深跌股天然高 beta，
必须扣掉 beta 之后再看，否则会把「赌beta」误当成「选股能力」。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1200
_rng = np.random.default_rng(20261022)


def date_mean(sub, key):
    return sub.groupby("date")[key].mean()


def boot_ci(sub, key, n=BOOT):
    v = date_mean(sub, key).dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def row(label, sub, ex_key="ex_fwd60", ab_key="ab_fwd60", boot=True):
    if sub is None or len(sub) == 0:
        return None
    e, lo, hi = boot_ci(sub, ex_key) if boot else (np.nan,) * 3
    ae, alo, ahi = boot_ci(sub, ab_key) if boot else (np.nan,) * 3
    return {
        "分组": label,
        "事件数": int(len(sub)),
        "交易日": int(sub["date"].nunique()),
        "去均值超额%": round(e * 100, 2),
        "90%CI": f"[{lo*100:.1f},{hi*100:.1f}]",
        "beta调整超额%": round(ae * 100, 2),
        "beta调整CI": f"[{alo*100:.1f},{ahi*100:.1f}]",
        "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
        "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
        "大亏<=-20%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
        "大赢>=+20%": round((sub["fwd60"] >= 0.20).mean() * 100, 1),
        "mae20中位%": round(sub["mae20"].median() * 100, 1),
        "日均票": round(len(sub) / max(sub["date"].nunique(), 1), 1),
    }


COLS = ["分组", "事件数", "交易日", "日均票", "去均值超额%", "90%CI",
        "beta调整超额%", "beta调整CI", "绝对收益%", "胜率%",
        "大亏<=-20%", "大赢>=+20%", "mae20中位%"]


def show(rows, title):
    rs = [r for r in rows if r]
    if not rs:
        print(f"--- {title} --- 无样本\n")
        return
    df = pd.DataFrame(rs)[COLS]
    print(f"--- {title} ---")
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(df.to_string(index=False))
    print()


def load():
    p = os.path.join(CACHE, "events_long_beta.pkl")
    ev = pd.read_pickle(p)
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta_raw"] = ev["beta"]
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    st = pd.read_pickle(os.path.join(CACHE, "regime_state.pkl"))
    want = ["date", "regime", "breadth", "沪深300_dd250", "沪深300_ret60"]
    keep = [c for c in want if c in st.columns and c not in ev.columns]
    keep = ["date"] + [c for c in keep if c != "date"]
    if len(keep) > 1:
        ev = ev.merge(st[keep], on="date", how="left")
    return ev


def main(ev):
    print("=" * 118)
    print("A. 核心决策表：距年高要求(prox) × 中级周期 —— 同时看去均值超额与 beta 调整超额")
    print("=" * 118)
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows = []
        for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            b = (s["width30"] <= 0.25) & (s["amt30"] >= 2e8)
            rows.append(row(f"[{r}] prox>={v:.2f} 全套基础条件",
                            s[(s["prox"] >= v) & b]))
        show(rows, f"[{r}] 逐级放宽 prox（基础条件：箱宽<=25%、日均额>=2亿）")

    print("=" * 118)
    print("B. prox 分档（互斥） —— 补全了「创新高 >=1.00」这一档")
    print("=" * 118)
    edges = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0, np.inf]
    names = ["<0.70", "0.70~0.80", "0.80~0.85", "0.85~0.90",
             "0.90~0.95", "0.95~1.00", "创新高>=1.00"]
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows = []
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            rows.append(row(f"[{r}] {nm}", s[(s["prox"] > lo) & (s["prox"] <= hi)]))
        show(rows, f"[{r}] prox 分档")

    print("=" * 118)
    print("C. 关键补充：市场处于深度回撤（沪深300距年高>=15%）时")
    print("=" * 118)
    deep = ev[ev["沪深300_dd250"] <= -0.15]
    rows = []
    for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        bb = (deep["width30"] <= 0.25) & (deep["amt30"] >= 2e8)
        rows.append(row(f"深度回撤 prox>={v:.2f}", deep[(deep["prox"] >= v) & bb]))
    show(rows, "市场深度回撤时逐级放宽 prox")

    print("=" * 118)
    print("D. 「资金流活跃」这条底线：绝对换手率 turn30 × 周期")
    print("=" * 118)
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[(ev["regime"] == r) & (ev["prox"] >= 0.85)]
        rows = []
        for lo, hi, nm in [(0, 1.0, "<1.0%(极度低迷)"), (1.0, 1.5, "1.0~1.5%"),
                           (1.5, 3.0, "1.5~3.0%"), (3.0, 6.0, "3.0~6.0%"),
                           (6.0, 10.0, "6.0~10.0%"), (10.0, 1e9, ">10%(过热)")]:
            rows.append(row(f"[{r}] turn30 {nm}", s[(s["turn30"] >= lo) & (s["turn30"] < hi)]))
        show(rows, f"[{r}] 30 日日均换手率（固定 prox>=0.85）")

    print("=" * 118)
    print("E. 「涨幅已经走了一大截」vs「还在起步」：近 60 日收益 × 周期")
    print("=" * 118)
    for r in ["UP", "DOWN"]:
        s = ev[(ev["regime"] == r) & (ev["prox"] >= 0.85)]
        q = s["ret60"].quantile([.2, .4, .6, .8]).values
        bins = [-np.inf, *q, np.inf]
        nms = ["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"]
        rows = []
        for lo, hi, nm in zip(bins[:-1], bins[1:], nms):
            rows.append(row(f"[{r}] {nm}", s[(s["ret60"] > lo) & (s["ret60"] <= hi)]))
        show(rows, f"[{r}] 近60日动量")

    print("=" * 118)
    print("F. 落地方案对比：v1 参数 vs 下跌市放宽版")
    print("=" * 118)
    down = ev[ev["regime"] == "DOWN"]
    combos = [
        ("v1原版 prox>=0.90 箱宽<=25% 额>=4亿",
         (down["prox"] >= 0.90) & (down["width30"] <= 0.25) & (down["amt30"] >= 4e8)),
        ("放宽A prox>=0.80 箱宽<=25% 额>=2亿",
         (down["prox"] >= 0.80) & (down["width30"] <= 0.25) & (down["amt30"] >= 2e8)),
        ("放宽B prox>=0.75 箱宽<=30% 额>=2亿 且 ret60 不太强",
         (down["prox"] >= 0.75) & (down["width30"] <= 0.30) & (down["amt30"] >= 2e8)
         & (down["ret60"] <= down["ret60"].quantile(0.8))),
        ("放宽C prox>=0.70 箱宽<=30% 额>=2亿 且 缩量(amt_act<=1.3)",
         (down["prox"] >= 0.70) & (down["width30"] <= 0.30) & (down["amt30"] >= 2e8)
         & (down["amt_act"] <= 1.3)),
        ("下跌市直接停手（不建仓）", None),
    ]
    rows = []
    for nm, mask in combos:
        if mask is None:
            rows.append(row("下跌市停手(0仓位)", down.iloc[:1].assign(
                fwd60=0.0, ex_fwd60=0.0, ab_fwd60=0.0, mae20=0.0)))
            continue
        rows.append(row(nm, down[mask]))
    show(rows, "[DOWN 周期] 候选方案")


if __name__ == "__main__":
    ev = load()
    print("事件数:", len(ev), " 区间:", ev["date"].min(), "~", ev["date"].max())
    print("beta 缺失:", int(ev["beta"].isna().sum()), " beta中位:", round(float(ev["beta"].median()), 3))
    print()
    main(ev)
