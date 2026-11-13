# -*- coding: utf-8 -*-
"""
换目标函数重新优化：能否找到一条「跟得上市场」的可行解？（2026-11-10）
================================================================
前面所有实证（含四门放宽、替代路径）都以 ab_fwd60（β调整超额）为目标。
research/yh_beta_bias.py 证明：严格池 β≈0.36，在 β调整口径下 +2.56%，
但相对当日横截面只有 −0.86%；REBOUND 段更极端（+2.29% vs −5.05%）。
→ 用户「市场涨了池子不涨」的体感，来自【优化目标与体感口径错配】。

本脚本把目标换成 ex_fwd60（相对当日横截面超额 = 投资者实际体感），重新：
  1. 复跑四门放宽（边际组 + 整池配对区块bootstrap + 逐年一致性）
  2. 网格搜索参数组合，找 ex 口径下的有效解
  3. 对最优组合做区块bootstrap + 逐年一致性 + 大亏率复核

★ 关键纪律：这不是要「改坏主池」，而是检验是否存在一条【不同的、以跟上市场
  为目标】的策略线。主池以 β中性 alpha 为目标，两条线目标不同，不可混用参数。

数据来源：通达信指数日线 / 腾讯前复权日线事件表。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_blockboot_all import make_dmap, block_boot, paired_block_boot, daily, sig  # noqa: E402
from yh_conserv_diag import load_events, BAD  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "output", "ex_objective_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def pool(d, prox=0.95, width=0.15, turn=6.0, ret60=0.19, amt=4e8):
    return d[(d["prox_now"] >= prox) & (d["width30"] <= width) &
             (d["amt30"] >= amt) & (d["turn30"] <= turn) &
             (pd.to_numeric(d["ret60"], errors="coerce") <= ret60)]


def main():
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 112)
    P("换目标函数重优化 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 112)
    P("目标：ex_fwd60 = fwd60 − 当日横截面均值（相对市场体感）")
    P("对照：ab_fwd60 = fwd60 − beta×mkt_fwd60（原优化目标，β中性 alpha）")
    P("样本：%s 行 / %d 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["date"].nunique(), ev["date"].min().date(), ev["date"].max().date()))
    base = pool(ev)
    P("基准池（0.95/15%%/6%%/19%%）：%d 事件，日均 %.2f 只" % (
        len(base), base.groupby("date").size().mean()))
    P("  基准池 ab口径 %+.2f%% | ex口径 %+.2f%% | β均 %.2f | 大亏率 %.1f%%" % (
        base["ab_fwd60"].mean() * 100, base["ex_fwd60"].mean() * 100,
        base["beta"].mean(), (base["fwd60"] < BAD).mean() * 100))

    # ---------- 1. 四门放宽在 ex 口径下复跑 ----------
    P("\n" + "=" * 112)
    P("【1】四门放宽在 ex 口径下的整池配对（放宽池 − 基准池，配对区块bootstrap）")
    P("=" * 112)
    P("  %-22s %10s %10s %12s %10s" % ("假设", "日均票", "ex超额", "配对差", "判定"))
    P("  " + "-" * 70)
    variants = [
        ("prox 0.95→0.90", dict(prox=0.90)),
        ("width 15%→20%", dict(width=0.20)),
        ("width 15%→25%", dict(width=0.25)),
        ("turn 6%→10%", dict(turn=10.0)),
        ("删 ret60 门", dict(ret60=np.inf)),
        ("全部放宽(0.90/20/10/∞)", dict(prox=0.90, width=0.20, turn=10.0, ret60=np.inf)),
    ]
    for tag, kw in variants:
        p = pool(ev, **kw)
        pt, lo, hi, nb = paired_block_boot(p, base, dmap, col="ex_fwd60")
        P("  %-22s %10.2f %+9.2f%% %+11.2fpp %s" % (
            tag, p.groupby("date").size().mean(), p["ex_fwd60"].mean() * 100,
            pt * 100 if not pd.isna(pt) else np.nan, sig(pt, lo, hi)))

    # ---------- 2. 网格搜索 ex 口径最优 ----------
    P("\n" + "=" * 112)
    P("【2】网格搜索（目标：ex_fwd60 最大化；同时报告 ab 口径与大亏率）")
    P("=" * 112)
    rows = []
    for prox in [0.85, 0.90, 0.95]:
        for width in [0.15, 0.20, 0.25, 0.30]:
            for turn in [6.0, 10.0, 99.0]:
                for r60 in [0.19, np.inf]:
                    p = pool(ev, prox=prox, width=width, turn=turn, ret60=r60)
                    if len(p) < 200:
                        continue
                    # 逐年一致性：该组合相对当日市场(ex>0)的年份数
                    ok = tot = 0
                    for y, g in p.groupby(p["date"].dt.year):
                        if len(g) < 30:
                            continue
                        tot += 1
                        ok += 1 if g["ex_fwd60"].mean() > 0 else 0
                    rows.append((prox, width, turn, r60, len(p),
                                 p.groupby("date").size().mean(),
                                 p["ex_fwd60"].mean(), p["ab_fwd60"].mean(),
                                 (p["fwd60"] < BAD).mean(), p["beta"].mean(), ok, tot))
    g = pd.DataFrame(rows, columns=["prox", "width", "turn", "ret60", "n", "日均",
                                    "ex", "ab", "大亏率", "beta", "ex正年", "有效年"])
    g = g.sort_values("ex", ascending=False).reset_index(drop=True)
    _b = g[(np.isclose(g["prox"], 0.95)) & (np.isclose(g["width"], 0.15)) &
           (np.isclose(g["turn"], 6.0)) & (np.isclose(g["ret60"], 0.19))]
    _rank = int(_b.index[0]) + 1 if len(_b) else -1
    P("  %-6s %-7s %-6s %-7s %7s %9s %9s %9s %7s %6s" % (
        "prox", "箱宽", "换手", "ret60", "日均票", "ex超额", "ab超额", "大亏率", "β", "ex正年"))
    P("  " + "-" * 92)
    for _, r in g.head(15).iterrows():
        P("  %-6.2f %-7.0f%% %-6.0f %-7s %7.2f %+8.2f%% %+8.2f%% %8.1f%% %6.2f %5d/%d" % (
            r["prox"], r["width"] * 100, r["turn"],
            "∞" if np.isinf(r["ret60"]) else "%.2f" % r["ret60"],
            r["日均"], r["ex"] * 100, r["ab"] * 100, r["大亏率"] * 100,
            r["beta"], r["ex正年"], r["有效年"]))
    P("  ── 基准组合（0.95/15/6/0.19）在 ex 口径下的排名：第 %d / %d" % (_rank, len(g)))

    # ---------- 3. 最优组合复核 ----------
    P("\n" + "=" * 112)
    P("【3】ex 口径最优组合复核（区块bootstrap / 逐年 / 分状态）")
    P("=" * 112)
    best = g.iloc[0]
    pb = pool(ev, prox=best["prox"], width=best["width"], turn=best["turn"], ret60=best["ret60"])
    P("  最优组合：prox≥%.2f / 箱宽≤%.0f%% / 换手≤%.0f / ret60≤%s" % (
        best["prox"], best["width"] * 100, best["turn"],
        "∞" if np.isinf(best["ret60"]) else "%.2f" % best["ret60"]))
    pt, lo, hi, nb = paired_block_boot(pb, base, dmap, col="ex_fwd60")
    pt2, lo2, hi2, nb2 = paired_block_boot(pb, base, dmap, col="ab_fwd60")
    P("  ex 口径配对差 %+.2fpp CI[%+.2f,%+.2f] %s" % (pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))
    P("  ab 口径配对差 %+.2fpp CI[%+.2f,%+.2f] %s（原目标会怎么判）" % (
        pt2 * 100, lo2 * 100, hi2 * 100, sig(pt2, lo2, hi2)))
    P("  大亏率 %.1f%%（基准 %.1f%%）| β均 %.2f（基准 %.2f）" % (
        (pb["fwd60"] < BAD).mean() * 100, (base["fwd60"] < BAD).mean() * 100,
        pb["beta"].mean(), base["beta"].mean()))
    for s in ["REBOUND", "UP", "CHOP", "DOWN"]:
        a, b = pb[pb["state"] == s], base[base["state"] == s]
        if len(a) < 50:
            continue
        P("    %-9s ex %+7.2f%% (基准 %+7.2f%%) 日均%.1f只 大亏率%.1f%%" % (
            s, a["ex_fwd60"].mean() * 100, b["ex_fwd60"].mean() * 100,
            a.groupby("date").size().mean(), (a["fwd60"] < BAD).mean() * 100))

    P("\n耗时 %.1fs" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
