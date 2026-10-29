# -*- coding: utf-8 -*-
"""
【有效样本量】诊断 —— 为什么「回测了 7 年」却只剩这一点点可用样本

核心问题
--------
fwd60 = 未来【60 个交易日】收益。相邻两个交易日的事件，其收益窗口几乎完全重叠
（第 t 天与第 t+1 天共享 59/60 ≈ 98% 的窗口），因此它们**不是独立观测**。
而此前的 bootstrap 是按【交易日】抽样的，等于默认每天独立 → 置信区间偏窄、
显著性被高估。

本脚本用两种口径估计真实有效样本：
  1) 聚类数：把事件日按「间隔 < 60 交易日」聚为一簇，每簇最多贡献 1 个独立观测
  2) 信息量折算：每簇贡献 min(1, 簇内跨度/60) 个有效观测

注意：这不是精确的统计修正，只是把「量级」摆出来供判断。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin, base_v2  # noqa: E402

H = 60  # fwd60 的持有期（交易日）


def _day_index(ev: pd.DataFrame) -> dict:
    """交易日 -> 全局序号，用于按【交易日】而非自然日计算间隔"""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(ev["date"]).unique()))
    return {d: i for i, d in enumerate(days)}


def eff_n(dates, dmap):
    """返回 (非重叠窗口数, 折算有效观测数, 最大连续跨度)

    注意：上一版用「相邻间隔<60 即合并」会产生链式传递，把跨年的日子全连成一簇，
    结果失真。改为【贪心取非重叠窗口】：从最早日起，跳过其后 60 个交易日内的所有日期，
    再取下一个。这给出「最多能放下多少个互不重叠的持有期」，是干净可解释的下限。
    """
    idx = sorted({dmap[pd.Timestamp(d)] for d in dates if pd.Timestamp(d) in dmap})
    if not idx:
        return 0, 0.0, 0
    n_win = 0
    i = 0
    blocks = []
    while i < len(idx):
        n_win += 1
        start = idx[i]
        j = i
        while j < len(idx) and idx[j] < start + H:
            j += 1
        blocks.append((start, idx[j - 1]))
        i = j
    eff = 0.0
    biggest = 0
    for s, t in blocks:
        span = t - s + 1
        biggest = max(biggest, span)
        eff += min(1.0, span / H)
    return n_win, eff, biggest


def report(name, d, dmap, note=""):
    if len(d) == 0:
        print("  %-34s 无事件" % name)
        return
    nd = d["date"].nunique()
    n_cl, eff, big = eff_n(d["date"].unique(), dmap)
    print("  %-34s 事件%6d  交易日%4d  →  独立簇%3d  有效观测≈%5.1f  最大簇跨度%4d日 %s"
          % (name, len(d), nd, n_cl, eff, big, note))


def bar(t):
    print("=" * 118)
    print(t)
    print("=" * 118)


def main():
    ev = load_events()
    fin = load_fin()
    e = attach_fin(ev, fin)
    dmap = _day_index(ev)

    bar("样本漏斗：从「回测 7 年」到「可用样本」是怎么层层缩水的")
    print("  回测区间: %s ~ %s  共 %d 个交易日" % (ev["date"].min(), ev["date"].max(), ev["date"].nunique()))
    print()
    print("  ▶ 第一层：周期标签筛选")
    print("  %-34s %10s %10s" % ("周期", "事件数", "交易日"))
    print("  " + "-" * 60)
    tot = ev["date"].nunique()
    for rg in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == rg]
        nd = s["date"].nunique()
        print("  %-34s %10d %10d  (占全部交易日 %.1f%%)" % (rg, len(s), nd, nd / tot * 100))
    print()
    print("  ▶ UP 周期的交易日本来就很少，逐年分布：")
    up = ev[ev["regime"] == "UP"].assign(yr=lambda x: pd.to_datetime(x["date"]).dt.year)
    print(up.groupby("yr").agg(事件=("date", "size"), 交易日=("date", "nunique")).to_string())
    print()
    print("  ▶ 第二/三层：再叠加 v2 结构纪律 + 基本面纪律")
    print("  %-34s %10s %10s" % ("层级", "事件数", "交易日"))
    print("  " + "-" * 60)
    upe = e[e["regime"] == "UP"]
    print("  %-34s %10d %10d" % ("UP 全部事件", len(upe), upe["date"].nunique()))
    upb = base_v2(upe[upe["YSTZ"].notna()])
    print("  %-34s %10d %10d" % ("UP + v2 四纪律", len(upb), upb["date"].nunique()))
    up30 = upb[upb["YSTZ"] >= 30]
    print("  %-34s %10d %10d" % ("UP + v2 + 营收>=30%", len(up30), up30["date"].nunique()))
    print()

    bar("有效独立样本：按 fwd60 窗口去重后还剩多少")
    print("  口径：事件日间隔 < %d 交易日算同一簇（收益窗口重叠）；独立簇 ≈ 真实观测数" % H)
    print()
    for rg in ["UP", "CHOP", "DOWN"]:
        se = e[(e["regime"] == rg) & e["YSTZ"].notna()]
        b = base_v2(se)
        print("  【%s】" % rg)
        report("基准池(v2四纪律)", b, dmap)
        for thr in [10, 30, 50]:
            if thr == 10:
                continue
            d = b[b["YSTZ"] >= thr]
            if len(d) >= 15:
                report("+营收>=%d%%" % thr, d, dmap)
        print()

    bar("结论性对照：本轮关键结论各自站在多少有效样本上")
    cases = [
        ("DOWN 基准池(去营收门)", e[(e["regime"] == "DOWN") & e["YSTZ"].notna()].pipe(base_v2)),
        ("DOWN +营收>=30%", base_v2(e[(e["regime"] == "DOWN") & e["YSTZ"].notna()]).pipe(lambda x: x[x["YSTZ"] >= 30])),
        ("UP 基准池", base_v2(e[(e["regime"] == "UP") & e["YSTZ"].notna()])),
        ("UP +营收>=30%", base_v2(e[(e["regime"] == "UP") & e["YSTZ"].notna()]).pipe(lambda x: x[x["YSTZ"] >= 30])),
        ("UP +营收>=50%", base_v2(e[(e["regime"] == "UP") & e["YSTZ"].notna()]).pipe(lambda x: x[x["YSTZ"] >= 50])),
        ("CHOP 基准池", base_v2(e[(e["regime"] == "CHOP") & e["YSTZ"].notna()])),
        ("CHOP +营收>=10%", base_v2(e[(e["regime"] == "CHOP") & e["YSTZ"].notna()]).pipe(lambda x: x[x["YSTZ"] >= 10])),
    ]
    print("  %-34s %8s %8s %10s %12s" % ("结论所依赖的样本", "事件", "交易日", "独立簇", "有效观测≈"))
    print("  " + "-" * 76)
    for nm, d in cases:
        if len(d) == 0:
            continue
        nd = d["date"].nunique()
        n_cl, eff, _ = eff_n(d["date"].unique(), dmap)
        flag = ""
        if eff < 3:
            flag = "  ← 几乎无统计效力"
        elif eff < 6:
            flag = "  ← 极弱"
        print("  %-34s %8d %8d %10d %12.1f%s" % (nm, len(d), nd, n_cl, eff, flag))
    print()
    print("  判读：有效观测 < 3 时，任何『显著』都应视为巧合；< 6 时只能作为方向性线索。")


if __name__ == "__main__":
    main()
