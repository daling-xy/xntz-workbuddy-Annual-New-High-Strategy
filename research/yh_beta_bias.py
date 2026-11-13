# -*- coding: utf-8 -*-
"""
口径验证：为什么「回测证据充分」却「实盘感觉踏空」？（2026-11-10）
================================================================
本项目所有参数实证的优化目标都是 ab_fwd60 = fwd60 − beta × mkt_fwd60
（β调整超额，用于剔除市场涨跌的影响）。但投资者实际感受到的是
「相对同期其他股票涨得多不多」= ex_fwd60 = fwd60 − 当日横截面均值。

若入选池的 β 系统性偏低，在【上涨/反弹段】两种口径会给出相反结论：
  ab = fwd − β·mkt  → β 小则减项小 → ab 被美化
  ex = fwd − 横截面均值 → 低β票绝对涨幅低 → ex 为负
本脚本检验这个假设，并量化「优化目标错配」的代价。

数据来源：腾讯前复权日线事件表 + 通达信指数日线（市场收益/宽度分层）。
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
OUT = os.path.join(BASE_DIR, "output", "beta_bias_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def row(tag, g):
    if len(g) == 0:
        return
    P("  %-18s %8d  β=%5.2f | 原始%+7.2f%%  相对横截面%+7.2f%%  β调整%+7.2f%% | 大亏率%5.1f%%" % (
        tag, len(g), g["beta"].mean(), g["fwd60"].mean() * 100,
        g["ex_fwd60"].mean() * 100, g["ab_fwd60"].mean() * 100,
        (g["fwd60"] < BAD).mean() * 100))


def main():
    ev = load_events()
    P("=" * 112)
    P("口径验证 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 112)
    P("ab_fwd60 = fwd60 − beta×mkt_fwd60（本项目所有参数实证的优化目标）")
    P("ex_fwd60 = fwd60 − 当日横截面均值（投资者实际感受到的相对表现）")
    P("mkt_fwd60 为同期市场收益；β 为个股历史 β（已 clip 至 0.5%~99.5% 分位）")

    for s in ["全样本", "REBOUND", "UP", "CHOP", "DOWN"]:
        sub = ev if s == "全样本" else ev[ev["state"] == s]
        pool = strict(sub)
        P("\n" + "=" * 112)
        P("【%s】市场未来60日收益均值 %+.2f%% | 样本 %d 行 | 严格池 %d 个事件" % (
            s, sub["mkt_fwd60"].mean() * 100, len(sub), len(pool)))
        P("  %-18s %8s  %-5s | %-9s %-16s %-13s | %s" % (
            "组", "n", "β", "原始fwd60", "相对横截面", "β调整", "大亏率"))
        P("  " + "-" * 104)
        row("全市场样本", sub)
        row("严格池", pool)
        # 池内按 β 分组
        if len(pool) > 100:
            pool = pool.copy()
            pool["_bq"] = pd.qcut(pool["beta"], 3, labels=["低β", "中β", "高β"], duplicates="drop")
            for q, g in pool.groupby("_bq", observed=True):
                row("  池内/" + str(q), g)
        # 口径差
        if len(pool):
            P("  → 口径差：β调整 %+.2f%%  相对横截面 %+.2f%%  【差 %+.2fpp】" % (
                pool["ab_fwd60"].mean() * 100, pool["ex_fwd60"].mean() * 100,
                (pool["ab_fwd60"].mean() - pool["ex_fwd60"].mean()) * 100))

    # 直接检验「β 越低，两种口径差越大」
    P("\n" + "=" * 112)
    P("机制检验：按 β 分组，看 ab−ex 口径差是否随 β 单调变化")
    P("=" * 112)
    ev["_bg"] = pd.qcut(ev["beta"], 5, labels=["β1低", "β2", "β3", "β4", "β5高"], duplicates="drop")
    P("  %-8s %10s %8s %12s %12s %12s" % ("组", "n", "β均值", "原始", "相对横截面", "β调整"))
    P("  " + "-" * 70)
    for q, g in ev.groupby("_bg", observed=True):
        P("  %-8s %10d %8.2f %+11.2f%% %+11.2f%% %+11.2f%%" % (
            q, len(g), g["beta"].mean(), g["fwd60"].mean() * 100,
            g["ex_fwd60"].mean() * 100, g["ab_fwd60"].mean() * 100))
    P("\n  结论：若低β组的「β调整」明显高于「相对横截面」，说明")
    P("  用 ab_fwd60 做优化目标会系统性偏好低β防御票 → 上涨段回测好看、实盘踏空。")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
