# -*- coding: utf-8 -*-
"""
F 段重做：events 表里的 ret_std 是【全空死列】（0/150万 非空），不能直接比较。
这里从 kline_long2 原始 K 线重算 30 日已实现波动率（日收益标准差，年化口径另给），
再回答「箱宽<=15% 是不是就是波动率<=15%」。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_rng = np.random.default_rng(20261023)


def build_index(k):
    idx = {}
    for c, bars in k.items():
        d = np.array([b["date"] for b in bars])
        cl = np.array([b["close"] for b in bars], dtype=float)
        idx[c] = (d, cl)
    return idx


def main():
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    idx = build_index(k)

    samp = ev.sample(min(6000, len(ev)), random_state=11).copy()
    vol30, vol30a, ok = [], [], []
    for code, date, w in zip(samp["code"].values, samp["date"].values, samp["width30"].values):
        got = idx.get(code)
        if got is None:
            vol30.append(np.nan); vol30a.append(np.nan); ok.append(False); continue
        d, cl = got
        i = int(np.searchsorted(d, date))
        if i < 31:
            vol30.append(np.nan); vol30a.append(np.nan); ok.append(False); continue
        seg = cl[i - 30:i + 1]
        if len(seg) < 31 or seg.min() <= 0:
            vol30.append(np.nan); vol30a.append(np.nan); ok.append(False); continue
        r = seg[1:] / seg[:-1] - 1.0
        v = float(np.std(r, ddof=1))
        vol30.append(v)
        vol30a.append(v * np.sqrt(250))
        ok.append(True)

    samp["vol30"] = vol30
    samp["vol30_ann"] = vol30a
    s = samp[np.array(ok)].copy()

    print("=" * 118)
    print("F. 「30日箱宽 <=15%」到底是不是「波动率 <=15%」？")
    print("=" * 118)
    print(f"  有效抽样 {len(s)} 条（从原始前复权日线重算 30 日已实现波动率）\n")
    print(f"  width30 与 日收益标准差 的相关系数        : {np.corrcoef(s['width30'], s['vol30'])[0,1]:.3f}")
    print(f"  width30 与 年化波动率   的相关系数        : {np.corrcoef(s['width30'], s['vol30_ann'])[0,1]:.3f}")
    print()
    q = s["width30"].quantile([0.25, 0.5, 0.75, 0.9])
    print("  width30(30日振幅) 分位: " + "  ".join(f"{k}={v:.1%}" for k, v in q.items()))
    q2 = s["vol30"].quantile([0.25, 0.5, 0.75, 0.9])
    print("  vol30(日收益标准差) 分位: " + "  ".join(f"{k}={v:.2%}" for k, v in q2.items()))
    q3 = s["vol30_ann"].quantile([0.25, 0.5, 0.75, 0.9])
    print("  年化波动率 分位        : " + "  ".join(f"{k}={v:.1%}" for k, v in q3.items()))
    print()
    sub = s[s["width30"] <= 0.15]
    print(f"  【箱宽<=15%】这一组对应的波动率：")
    print(f"     日收益标准差 中位 {sub['vol30'].median():.2%}（全市场中位 {s['vol30'].median():.2%}）")
    print(f"     年化波动率   中位 {sub['vol30_ann'].median():.1%}（全市场中位 {s['vol30_ann'].median():.1%}）")
    print()
    print("  结论：箱宽是【30日内最高收盘/最低收盘-1】，是区间振幅(range)，不是标准差。")
    print("        两者相关但不相等：箱宽15% 大致对应日波动 ~1.3%、年化 ~20% 上下，")
    print("        而【日波动15%】那种量级在 A 股几乎不存在（日波动15%≈年化240%）。")
    print("        所以『箱宽15% = 波动率15%』这个说法不成立，15% 是振幅口径的 15%。")
    print()

    # 顺带：用真实的已实现波动率做门槛，和箱宽门槛比一比
    print("=" * 118)
    print("F2. 用振幅做门槛 vs 用真实波动率做门槛（抽样口径，同一批样本）")
    print("=" * 118)
    lo, hi = s["beta"].quantile([0.005, 0.995])
    s["beta_c"] = s["beta"].clip(lo, hi)
    s["ab"] = s["fwd60"] - s["beta_c"] * s["mkt_fwd60"]

    def stat(sub, nm):
        if len(sub) < 100:
            return None
        v = sub.groupby("date")["ab"].mean()
        if len(v) < 3:
            return None
        bs = v.values[_rng.integers(0, len(v), size=(1000, len(v)))].mean(axis=1)
        return {"门槛": nm, "样本": len(sub),
                "beta调整%": round(v.mean() * 100, 2),
                "CI": f"[{np.percentile(bs,5)*100:.1f},{np.percentile(bs,95)*100:.1f}]",
                "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
                "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
                "mae20%": round(sub["mae20"].median() * 100, 1)}

    rows = []
    for nm, m in [
        ("不约束", pd.Series(True, index=s.index)),
        ("箱宽<=15%", s["width30"] <= 0.15),
        ("箱宽<=20%", s["width30"] <= 0.20),
        ("日波动<=1.2%", s["vol30"] <= 0.012),
        ("日波动<=1.5%", s["vol30"] <= 0.015),
        ("日波动<=2.0%", s["vol30"] <= 0.020),
        ("箱宽<=15% 且 日波动<=1.5%", (s["width30"] <= 0.15) & (s["vol30"] <= 0.015)),
    ]:
        r = stat(s[m], nm)
        if r:
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  注：此处为抽样（6000 条）口径，样本量有限，仅用于判断『振幅 vs 波动率』哪个门更有效。")


if __name__ == "__main__":
    main()
