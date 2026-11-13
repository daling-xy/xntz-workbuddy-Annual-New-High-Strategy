# -*- coding: utf-8 -*-
"""
「不放宽固定阈值」的替代路径验证（2026-11-10）
================================================================
若问题本质是「绝对阈值随市场波动机械收紧」或「收益源错配」，
那么全局放宽阈值是错的解法。本脚本验证三条替代路径：

  A 截面自适应分位门槛
      把箱宽/换手的【绝对阈值】换成【当日横截面分位】。
      市场整体波动放大时自动放松、收敛时自动收紧，
      ★完全不需要判断市场状态★（符合 2026-11-10「不做市场判断」的要求）。
      分位标定为「历史通过率与固定门一致」，保证公平对比。

  B 独立动量卫星线（双轨）
      主池严格门槛不动，另建一条抓「底部反转 / 强势动量」的独立线，
      用不同的收益源覆盖用户想抓的弹性票。检验其是否有独立 alpha、与主池重叠率。

  C 池内排序改革
      不改入池门槛，只检验严格门内 ret20 分位高的票是否后续更强；
      若成立，至少可通过排序把池内最有弹性的票排前面（零成本）。

数据来源：通达信指数日线（宽度分层）＋腾讯前复权日线事件表。
状态分层仅用于历史研究，不对当前市场下判断。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_blockboot_all import make_dmap, paired_block_boot, sig  # noqa: E402
from yh_conserv_diag import load_events, strict, BAD  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "output", "alt_paths_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def fixed(d):
    return d[(d["prox_now"] >= 0.95) & (d["width30"] <= 0.15) &
             (d["amt30"] >= 4e8) & (d["turn30"] <= 6.0) &
             (pd.to_numeric(d["ret60"], errors="coerce") <= 0.19)]


def line(d, tag, dmap, sub_all=None):
    """一条候选池的常规统计 + 相对配对。"""
    if len(d) < 30:
        return "样本不足"
    per = d.groupby("date").size().mean()
    r = d["ab_fwd60"].mean()
    bad = (d["ab_fwd60"] < BAD).mean()
    return per, r, bad


def partA(ev, dmap):
    P("=" * 112)
    P("A  截面自适应分位门槛（不判断市场，用当日横截面相对位置替代绝对阈值）")
    P("=" * 112)
    # 当日横截面分位（事件表为轮转采样，用当日样本内分位近似）
    ev["w_rank"] = ev.groupby("date")["width30"].rank(pct=True)
    ev["t_rank"] = ev.groupby("date")["turn30"].rank(pct=True)

    # 标定：在「其余门已过」的样本上，让分位门的历史通过率 == 固定门通过率
    base_other = ev[(ev["prox_now"] >= 0.95) & (ev["amt30"] >= 4e8) &
                    (pd.to_numeric(ev["ret60"], errors="coerce") <= 0.19)]
    fixed_rate_w = (base_other["width30"] <= 0.15).mean()
    fixed_rate_t = (base_other[(base_other["width30"] <= 0.15)]["turn30"] <= 6.0).mean()
    grid = np.arange(0.05, 0.95, 0.01)
    wr = base_other["w_rank"].dropna().values
    pass_w = np.array([(wr <= q).mean() for q in grid])
    qw = float(grid[np.argmin(np.abs(pass_w - fixed_rate_w))])
    tr = base_other.loc[base_other["w_rank"] <= qw, "t_rank"].dropna().values
    pass_t = np.array([(tr <= q).mean() for q in grid])
    qt = float(grid[np.argmin(np.abs(pass_t - fixed_rate_t))])
    P("  标定：固定箱宽门通过率 %.1f%% → 自适应分位 q_w=%.2f（实际 %.1f%%）" % (
        fixed_rate_w * 100, qw, pass_w[np.argmin(np.abs(pass_w - fixed_rate_w))] * 100))
    P("        固定换手门通过率 %.1f%% → 自适应分位 q_t=%.2f（实际 %.1f%%）" % (
        fixed_rate_t * 100, qt, pass_t[np.argmin(np.abs(pass_t - fixed_rate_t))] * 100))

    ad = ev[(ev["prox_now"] >= 0.95) & (ev["w_rank"] <= qw) &
            (ev["amt30"] >= 4e8) & (ev["t_rank"] <= qt) &
            (pd.to_numeric(ev["ret60"], errors="coerce") <= 0.19)]
    fx = fixed(ev)
    P("\n  %-10s %9s %9s | %10s %10s %10s | %s" % (
        "范围", "固定日均", "自适应日均", "固定超额", "自适应超额", "大亏率差", "配对判定"))
    P("  " + "-" * 92)
    for s in ["全样本", "REBOUND", "UP", "CHOP", "DOWN"]:
        if s == "全样本":
            a, b = ad, fx
        else:
            a, b = ad[ad["state"] == s], fx[fx["state"] == s]
        if len(a) < 100 or len(b) < 100:
            P("  %-10s 样本不足" % s)
            continue
        pt, lo, hi, nb = paired_block_boot(a, b, dmap)
        P("  %-10s %9.2f %9.2f | %+9.2f%% %+9.2f%% %+9.1fpp | %+.2fpp CI[%+.2f,%+.2f] %s" % (
            s, b.groupby("date").size().mean(), a.groupby("date").size().mean(),
            b["ab_fwd60"].mean() * 100, a["ab_fwd60"].mean() * 100,
            ((a["ab_fwd60"] < BAD).mean() - (b["ab_fwd60"] < BAD).mean()) * 100,
            pt * 100 if not pd.isna(pt) else np.nan,
            lo * 100 if not pd.isna(lo) else np.nan,
            hi * 100 if not pd.isna(hi) else np.nan, sig(pt, lo, hi)))

    # 逐年一致性（自适应 − 固定）
    P("\n  逐年方向一致性（配对日均值差，正的年份数 / 有效年）：")
    ok = tot = 0
    for y in sorted(set(fx["date"].dt.year) & set(ad["date"].dt.year)):
        fy, ay = fx[fx["date"].dt.year == y], ad[ad["date"].dt.year == y]
        cf = fy.groupby("date")["ab_fwd60"].mean()
        ca = ay.groupby("date")["ab_fwd60"].mean()
        com = cf.index.intersection(ca.index)
        if len(com) < 20:
            continue
        tot += 1
        ok += 1 if (ca.loc[com] - cf.loc[com]).mean() > 0 else 0
    P("    %d/%d 年自适应更优（及格线 6/8）" % (ok, tot))
    return qw, qt


def partB(ev, dmap):
    P("\n" + "=" * 112)
    P("B  独立动量卫星线（主池不动，另建第二收益源）")
    P("=" * 112)
    # 卫星线：位置放宽到 0.80，但要求近期强势 + 趋势确认 + 放量
    sat = ev[(ev["prox_now"] >= 0.80) & (ev["ret20_pct"] >= 0.80) &
             (ev["amt30"] >= 4e8) & (ev["close"] > ev["ma20"]) &
             (ev["ma20"] > ev["ma60"]) & (ev["vol_dry"] > 1.0)]
    fx = fixed(ev)
    P("  定义：距年高≥0.80 且 ret20 横截面分位前20% 且 额≥4亿 且 close>MA20>MA60 且 近10日量能/30日>1")
    P("  %-10s %9s %9s | %10s %10s %10s | %s" % (
        "范围", "主池日均", "卫星日均", "卫星超额", "当日市场", "卫星大亏率", "配对判定"))
    P("  " + "-" * 96)
    for s in ["全样本", "REBOUND", "UP", "CHOP", "DOWN"]:
        sub = ev if s == "全样本" else ev[ev["state"] == s]
        a, m = sat[sat["state"] == s] if s != "全样本" else sat, sub
        if len(a) < 100:
            P("  %-10s 样本不足" % s)
            continue
        pt, lo, hi, nb = paired_block_boot(a, m, dmap)
        b = fx[fx["state"] == s] if s != "全样本" else fx
        P("  %-10s %9.2f %9.2f | %+9.2f%% %+9.2f%% %9.1f%% | %+.2fpp CI[%+.2f,%+.2f] %s" % (
            s, b.groupby("date").size().mean() if len(b) else 0,
            a.groupby("date").size().mean(), a["ab_fwd60"].mean() * 100,
            m["ab_fwd60"].mean() * 100, (a["ab_fwd60"] < BAD).mean() * 100,
            pt * 100 if not pd.isna(pt) else np.nan,
            lo * 100 if not pd.isna(lo) else np.nan,
            hi * 100 if not pd.isna(hi) else np.nan, sig(pt, lo, hi)))
    # 重叠率
    pk = set(zip(fx["date"], fx["code"]))
    sk = set(zip(sat["date"], sat["code"]))
    P("\n  主池/卫星重合事件：%d 条，占主池 %.1f%%、占卫星 %.1f%% → %s" % (
        len(pk & sk), len(pk & sk) / max(1, len(pk)) * 100,
        len(pk & sk) / max(1, len(sk)) * 100,
        "低重合=独立收益源" if len(pk & sk) / max(1, len(pk)) < 0.3 else "重合偏高"))
    # 逐年方向
    P("  逐年方向一致性（卫星 − 当日市场，配对日均值）：")
    ok = tot = 0
    for y in sorted(ev["date"].dt.year.unique()):
        ay = sat[sat["date"].dt.year == y]
        my = ev[ev["date"].dt.year == y]
        if len(ay) < 30:
            continue
        ca = ay.groupby("date")["ab_fwd60"].mean()
        cm = my.groupby("date")["ab_fwd60"].mean()
        com = ca.index.intersection(cm.index)
        if len(com) < 20:
            continue
        tot += 1
        ok += 1 if (ca.loc[com] - cm.loc[com]).mean() > 0 else 0
    P("    %d/%d 年卫星跑赢当日市场（及格线 6/8）" % (ok, tot))


def partC(ev):
    P("\n" + "=" * 112)
    P("C  池内排序改革：严格门内，近期强势(ret20分位)是否后续更强？")
    P("=" * 112)
    fx = fixed(ev).copy()
    fx["q"] = pd.qcut(fx["ret20_pct"], 5, labels=["Q1弱", "Q2", "Q3", "Q4", "Q5强"], duplicates="drop")
    P("  %-8s %9s %11s %10s" % ("分组", "n", "β调整超额", "大亏率"))
    P("  " + "-" * 44)
    for q, g in fx.groupby("q", observed=True):
        P("  %-8s %9d %+10.2f%% %9.1f%%" % (
            q, len(g), g["ab_fwd60"].mean() * 100, (g["ab_fwd60"] < BAD).mean() * 100))
    # 逐年 Q5−Q1
    ok = tot = 0
    for y, g in fx.groupby(fx["date"].dt.year):
        a = g[g["q"] == "Q5强"]["ab_fwd60"].mean()
        b = g[g["q"] == "Q1弱"]["ab_fwd60"].mean()
        if pd.isna(a) or pd.isna(b):
            continue
        tot += 1
        ok += 1 if a > b else 0
    P("  Q5−Q1 逐年为正：%d/%d 年" % (ok, tot))


def main():
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 112)
    P("替代路径验证 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 112)
    P("事件表 %s 行 / %d 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["date"].nunique(), ev["date"].min().date(), ev["date"].max().date()))
    partA(ev, dmap)
    partB(ev, dmap)
    partC(ev)
    P("\n耗时 %.1fs" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
