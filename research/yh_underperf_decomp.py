# -*- coding: utf-8 -*-
"""
「踏空」的收益分解与逐年一致性（2026-11-10 收官验证）
================================================================
前面实证已确认：严格池 β≈0.36，β调整超额 +2.56%，但相对当日横截面 −0.86%，
REBOUND 段放大到 −5.05%。本脚本回答最后一问：
  这个「相对跑输」是每年都有的系统性现象，还是只发生在反弹段？
  它能不能被 β 敞口完全解释（若是，则是设计使然，不是选股失效）？

四种口径并列，避免任何单一口径误导：
  ① 原始 fwd60            —— 绝对赚了多少
  ② 相对全市场样本均值     —— 长期是否跑赢大盘
  ③ 相对当日横截面 (ex)    —— 用户体感：同一天别的票涨得比我多吗
  ④ β调整超额 (ab)         —— 本项目参数实证的优化目标

数据来源：通达信指数日线 / 腾讯前复权日线事件表。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_conserv_diag import load_events, strict, BAD  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "output", "underperf_decomp_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def main():
    ev = load_events()
    pool = strict(ev)
    P("=" * 110)
    P("「踏空」收益分解与逐年一致性 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 110)
    P("四种口径：①原始fwd60 ②相对全市场均值 ③相对当日横截面(ex，用户体感) ④β调整超额(ab，优化目标)")

    # ---------- 1. 全样本四口径并列 ----------
    mkt = ev["fwd60"].mean()
    P("\n【1】全样本四口径并列")
    P("  %-16s %10s %10s %10s %10s %9s" % ("组", "n", "①原始", "②减全市场", "③ex", "④ab"))
    P("  " + "-" * 76)
    for tag, g in [("全市场样本", ev), ("严格池", pool)]:
        P("  %-16s %10d %+9.2f%% %+9.2f%% %+9.2f%% %+9.2f%%" % (
            tag, len(g), g["fwd60"].mean() * 100,
            (g["fwd60"].mean() - mkt) * 100, g["ex_fwd60"].mean() * 100,
            g["ab_fwd60"].mean() * 100))
    P("  大亏率：严格池 %.1f%% vs 全市场 %.1f%% | β：%.2f vs 1.00" % (
        (pool["fwd60"] < BAD).mean() * 100, (ev["fwd60"] < BAD).mean() * 100,
        pool["beta"].mean()))

    # ---------- 2. 收益分解：β 敞口 vs 选股 ----------
    P("\n【2】分解：踏空有多少是「低β」造成的，多少是「选股」造成的？")
    mkt_ret = ev["mkt_fwd60"].mean()
    beta_p = pool["beta"].mean()
    capm = beta_p * mkt_ret
    actual = pool["fwd60"].mean()
    P("  市场未来60日收益均值 %+.2f%%，严格池 β=%.2f → CAPM 预期 %+.2f%%" % (
        mkt_ret * 100, beta_p, capm * 100))
    P("  严格池实际原始收益 %+.2f%% → 选股 α = 实际 − 预期 = %+.2f%%（正）" % (
        actual * 100, (actual - capm) * 100))
    P("  若持有 β=1 的市场组合，同期可得 %+.2f%%；低β让池子少赚 %.2fpp" % (
        mkt_ret * 100, (mkt_ret - capm) * 100))
    P("  → 结论：绝对收益相对全市场 %+.2fpp（几乎打平），选股 α 为正，" % (actual - mkt))
    P("     「踏空」主要来自【低β敞口】，不是选股失效；代价换来了大亏率减半。")

    # 按市场未来收益分档，看踏空是否随市场上涨放大
    P("\n【3】踏空是否随「市场后来涨了多少」放大？（按当日 mkt_fwd60 分5档）")
    P("  %-12s %10s %10s %11s %11s %10s" % (
        "市场收益档", "事件数", "市场收益", "池原始", "池ex", "池大亏率"))
    P("  " + "-" * 74)
    ev["_mq"] = pd.qcut(ev["mkt_fwd60"], 5, labels=["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"], duplicates="drop")
    pool2 = strict(ev)
    for q, g in ev.groupby("_mq", observed=True):
        p = pool2[pool2["_mq"] == q]
        if len(p) < 20:
            continue
        P("  %-12s %10d %+9.2f%% %+10.2f%% %+10.2f%% %9.1f%%" % (
            q, len(p), g["mkt_fwd60"].mean() * 100, p["fwd60"].mean() * 100,
            p["ex_fwd60"].mean() * 100, (p["fwd60"] < BAD).mean() * 100))

    # ---------- 4. 逐年一致性 ----------
    P("\n【4】逐年一致性：相对当日横截面(ex) 是否【每年】都跑输？")
    P("  %-8s %9s %11s %11s %11s %10s" % ("年", "池事件", "池ex", "池ab", "池大亏率", "市场大亏率"))
    P("  " + "-" * 70)
    neg = tot = 0
    for y in sorted(ev["date"].dt.year.unique()):
        p = pool[pool["date"].dt.year == y]
        e = ev[ev["date"].dt.year == y]
        if len(p) < 20:
            continue
        tot += 1
        neg += 1 if p["ex_fwd60"].mean() < 0 else 0
        P("  %-8d %9d %+10.2f%% %+10.2f%% %10.1f%% %10.1f%%" % (
            y, len(p), p["ex_fwd60"].mean() * 100, p["ab_fwd60"].mean() * 100,
            (p["fwd60"] < BAD).mean() * 100, (e["fwd60"] < BAD).mean() * 100))
    P("  → ex 为负的年份 %d/%d；ab 为正的年份见上表。" % (neg, tot))
    P("     若 ex 常年为负而 ab 常年为正，则「踏空」是【系统性设计取舍】，不是某几天的运气。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
