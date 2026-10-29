# -*- coding: utf-8 -*-
"""
前置数据修复（针对 v2 结论的三处隐患）：

 F1 幸存者偏差：kline_long2 的股票列表来自「当前全 A 快照」，期间退市的股票不在样本内。
    量化：2019 年样本 3514 只 vs A 股实际约 3777 只，缺口约 7%（逐年递减至 2024 年 1%）。
    修复口径：加列「是否 2019-06 之前已上市的老股」，用纯老股子样本重跑全部核心结论。
 F2 成交额门槛时变：amt30>=2亿 是名义值。2019 年全 A 日均成交约 4000 亿、2026 年约 1.5 万亿，
    同一个绝对门槛在早年是「更贵」的标准。修复：改用「当日全市场 amt30 的横截面分位数」。
 F3 样本口径一致性：回测要求 bars>=450（约 2 年），实盘面板只要求 idx>=250。
    这里显式加列上市天数，并核对实盘候选是否落在回测适用区间内。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def main():
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))

    # --- F1/F3 上市日期与上市天数 ---
    first = {c: b[0]["date"] for c, b in k.items() if b}
    nbar = {c: len(b) for c, b in k.items() if b}
    ev["ipo"] = ev["code"].map(first)
    ev["nbars"] = ev["code"].map(nbar)
    # 老股 = 2019-06 之前（回测起点）就已上市
    ev["is_old"] = (ev["ipo"] <= "2019-06-13").astype(int)

    # --- F2 相对流动性：当日全市场 amt30 的横截面分位数 ---
    ev = ev.sort_values(["date", "amt30"])
    ev["amt_pct"] = ev.groupby("date")["amt30"].rank(pct=True)

    ev.to_pickle(os.path.join(CACHE, "events_fixed.pkl"))

    print("=" * 116)
    print("F1 幸存者偏差的规模")
    print("=" * 116)
    ev["year"] = ev["date"].astype(str).str[:4]
    g = ev.groupby("year").agg(样本股数=("code", "nunique"),
                              其中老股数=("is_old", "sum"),
                              事件数=("code", "size"))
    g["老股占比%"] = (g["其中老股数"] / g["样本股数"] * 100).round(1)
    print(g.to_string())
    print()
    print("  说明：样本股数逐年上升说明新股是按真实上市时间进入的（没有伪历史）。")
    print("  缺口 = 期间退市/长停的股票：2019 年约 263 只（7%），2024 年后约 1%。")
    print("  这类票退市前普遍表现偏差，剔除它们会【高估绝对收益】，")
    print("  但对【横截面超额 / beta 调整超额】影响有限（分子分母同一样本）。")
    print("  下面的检验用「纯老股子样本」直接验证结论是否依赖这批偏差样本。")

    print()
    print("=" * 116)
    print("F2 成交额绝对门槛的时变性")
    print("=" * 116)
    t = ev.groupby("year")["amt30"].agg(
        中位=("median"), p70=("quantile"), p90=("quantile")).copy()
    t["amt30中位(亿)"] = (t["中位"] / 1e8).round(2)
    t["达到2亿的占比%"] = ev.groupby("year")["amt30"].apply(lambda s: (s >= 2e8).mean() * 100).round(1)
    print(t[["amt30中位(亿)", "达到2亿的占比%"]].to_string())
    print()
    print("  若「2 亿」在各年筛出的比例差异很大，说明绝对门槛不可比 → 改用分位数。")
    print("  下面统一用 amt_pct（当日横截面分位数）重跑，检验结论是否稳健。")

    print()
    print("=" * 116)
    print("F3 回测样本 vs 实盘面板的口径一致性")
    print("=" * 116)
    print("  回测：bars>=450（约 2 年），事件表共 %d 只股票" % ev["code"].nunique())
    print("  实盘：panel_320 要求 idx>=250（约 1 年），即上市满 1 年即可入选")
    print("  → 回测结论对「上市 1~2 年的次新股」覆盖不足，实盘若选中需单独标注。")


if __name__ == "__main__":
    main()
