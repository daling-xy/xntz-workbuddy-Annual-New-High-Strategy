# -*- coding: utf-8 -*-
"""
两个口径问题的实证检验：

 E. prox 的定义之争
    现在用的是 prox = 【近30日最高收盘】/【250日最高收盘】。
    但用户原话是「距年高的百分比」，直觉上更像【当前收盘】/【250日最高收盘】。
    两者差别很大：前者允许「从年高回落 15% 后横盘」的票通过（只要近30日曾摸到过），
    后者要求当前价就贴着年高。这正是「中信证券/财通证券被选出来」的原因。
    本段检验：哪个口径选股效果更好？

 F. 「箱宽 15%」是不是「波动率 15%」
    width30 = 30日最高收盘/最低收盘 - 1，是【区间振幅 range】，不是标准差波动率。
    本段给出两者的相关性，并直接对比「用振幅做门槛」vs「用波动率做门槛」的效果。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1000
_rng = np.random.default_rng(20261023)


def boot_ci(sub, key, n=BOOT):
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def row(label, sub):
    if sub is None or len(sub) < 200:
        return None
    v, lo, hi = boot_ci(sub, "ab_fwd60")
    e, elo, ehi = boot_ci(sub, "ex_fwd60")
    return {"分组": label, "事件数": len(sub),
            "日均票": round(len(sub) / max(sub["date"].nunique(), 1), 1),
            "beta调整%": round(v * 100, 2), "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
            "去均值超额%": round(e * 100, 2),
            "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
            "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
            "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
            "mae20%": round(sub["mae20"].median() * 100, 1)}


COLS = ["分组", "事件数", "日均票", "beta调整%", "CI", "去均值超额%", "绝对收益%", "胜率%", "大亏%", "mae20%"]


def show(rows, title):
    rs = [r for r in rows if r]
    print(f"--- {title} ---")
    if not rs:
        print("  无样本\n")
        return
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(pd.DataFrame(rs)[COLS].to_string(index=False))
    print()


def load():
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    # 口径 B：当前价距年高
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    return ev


def base(s, amt=4e8):
    return (s["amt30"] >= amt) & (s["turn30"] <= 6.0) & (s["width30"] <= 0.15)


def main(ev):
    print("=" * 132)
    print("E. prox 口径之争：【近30日最高/年高】 vs 【当前价/年高】")
    print("=" * 132)
    print("  统一条件：日均额>=4亿、换手<=6%、30日箱宽<=15%\n")
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows = []
        for v in [0.85, 0.90, 0.95]:
            rows.append(row(f"[{r}] A口径 近30日高/年高 >={v:.2f}", s[base(s) & (s["prox"] >= v)]))
        for v in [0.80, 0.85, 0.90, 0.95]:
            rows.append(row(f"[{r}] B口径 当前价/年高 >={v:.2f}", s[base(s) & (s["prox_now"] >= v)]))
        show(rows, f"[{r}] 两种 prox 口径")

        rows = []
        for v in [0.95, 0.90, 0.85]:
            rows.append(row(f"[{r}] 双口径都满足(近30日高&当前价均>={v:.2f})",
                            s[base(s) & (s["prox"] >= v) & (s["prox_now"] >= v)]))
        show(rows, f"[{r}] 双口径交集（最严：既近30日摸过年高、当前价也贴着年高）")

    print("=" * 132)
    print("F. 「箱宽 15%」= 波动率 15% 吗？")
    print("=" * 132)
    s = ev.sample(min(200000, len(ev)), random_state=7)
    a, b = s["width30"].values, s["ret_std"].values
    m = np.isfinite(a) & np.isfinite(b) & (a < 1) & (b < 1)
    print(f"  width30(30日振幅) 与 ret_std(日收益标准差) 的相关系数: "
          f"{np.corrcoef(a[m], b[m])[0,1]:.3f}   （相关但远不等于同一件事）")
    q = ev["width30"].quantile([0.25, 0.5, 0.75, 0.9]).values
    print(f"  width30 分位: 25%={q[0]:.1%}  50%={q[1]:.1%}  75%={q[2]:.1%}  90%={q[3]:.1%}")
    q2 = ev["ret_std"].quantile([0.25, 0.5, 0.75, 0.9]).values
    print(f"  ret_std 分位: 25%={q2[0]:.2%}  50%={q2[1]:.2%}  75%={q2[2]:.2%}  90%={q2[3]:.2%}")
    print(f"  → 箱宽15% 大约对应波动率的 {ev.loc[ev['width30'] <= 0.15, 'ret_std'].median():.2%}（中位）\n")

    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows = []
        for nm, m in [
            ("无波动约束", pd.Series(True, index=s.index)),
            ("箱宽<=15%", s["width30"] <= 0.15),
            ("箱宽<=20%", s["width30"] <= 0.20),
            ("ret_std<=1.8%", s["ret_std"] <= 0.018),
            ("ret_std<=2.2%", s["ret_std"] <= 0.022),
            ("箱宽<=15% 且 ret_std<=2.2%", (s["width30"] <= 0.15) & (s["ret_std"] <= 0.022)),
        ]:
            sub = s[(s["amt30"] >= 4e8) & (s["turn30"] <= 6.0) & (s["prox"] >= 0.95) & m]
            rows.append(row(f"[{r}] {nm}", sub))
        show(rows, f"[{r}] 用振幅做门槛 vs 用波动率做门槛（统一 prox>=0.95 额>=4亿 换手<=6%）")


if __name__ == "__main__":
    main(load())
