# -*- coding: utf-8 -*-
"""
四门放宽实证（2026-11-10 应户主要求）
================================================================
待验假设（围绕"池子感觉不对劲、能否放宽"）：
  H1 距年高 prox_now：0.95 → 0.90
  H2 箱宽   width30 ：15%  → 20%
  H3 换手   turn30  ：6%   → 7%
  H4 ret60 门：0.19 → 删除

方法（沿用既有纪律，两处防坑）：
  ① 边际组检验：在【不含被测门】的样本上，把"新放进来的一批"（边际组 A）
     与"原门内"（B）逐年比 ab_fwd60（收益）与大亏率（ab_fwd60<-20%，风险）。
     ——若样本已含被测门，边际组会被剔光（幸存者偏差，见 补充诊断_20261029 §2）。
  ② 整池配对区块 bootstrap：放宽后整池 vs 现池的日均超额差，
     区块=60交易日、4000 次重抽样、90%CI，全样本 + 分周期。

判定口径（与 year_stability 一致）：
  - 有效年：两组事件数均 ≥8
  - 方向一致：该年 A−B 与全样本方向同号
  - 及格线：≥6/8 年
  - 放宽的采纳逻辑：边际组【不更差】才可放——收益维度 A−B 一致为负、
    或风险维度 A 大亏率一致更高 → 拒绝放宽。
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE_DIR, "cache")
OUT = os.path.join(BASE_DIR, "output", "relax4_20261110.txt")

_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


MIN_N = 8          # 有效年阈值（与 year_stability 一致）
BAD = -0.20        # 大亏定义：ab_fwd60 < -20%


# ---------------------------------------------------------------- 事件表
def load_events():
    """直接读 events_fixed.pkl 全列（yh_rev_fine.load_events 会丢 ret60），
    复用其清洗口径：beta clip + ab_fwd60 + prox_now。"""
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["beta"].notna() & ev["fwd60"].notna()]
    return ev.reset_index(drop=True)


# ---------------------------------------------------------------- 门定义
BASE_GATES = dict(prox=0.95, width=0.15, amt=4e8, turn=6.0, ret60=0.19)


def gate(d: pd.DataFrame, drop: str | None = None, over: dict | None = None) -> pd.DataFrame:
    """施加基准门；drop=被测门名（该门不施加）；over=要覆盖的门阈值字典。"""
    g = dict(BASE_GATES)
    if over:
        g.update(over)
    m = pd.Series(True, index=d.index)
    if drop != "prox":
        m &= d["prox_now"] >= g["prox"]
    if drop != "width":
        m &= d["width30"] <= g["width"]
    if drop != "amt":
        m &= d["amt30"] >= g["amt"]
    if drop != "turn":
        m &= d["turn30"] <= g["turn"]
    if drop != "ret60":
        m &= pd.to_numeric(d["ret60"], errors="coerce") <= g["ret60"]
    return d[m]


# ---------------------------------------------------------------- 逐年 A/B
def year_ab(d: pd.DataFrame, maskA, maskB) -> pd.DataFrame:
    dd = d.copy()
    dd["_y"] = dd["date"].dt.year
    dd["_bad"] = dd["ab_fwd60"] < BAD
    rows = []
    for y, g in dd.groupby("_y"):
        a, b = g[maskA(g)], g[maskB(g)]
        if len(a) == 0 or len(b) == 0:
            rows.append((y, 0 if len(a) == 0 else len(a), 0 if len(b) == 0 else len(b),
                         np.nan, np.nan, np.nan, np.nan))
            continue
        rows.append((y, len(a), len(b),
                     a["ab_fwd60"].mean(), b["ab_fwd60"].mean(),
                     a["_bad"].mean(), b["_bad"].mean()))
    return pd.DataFrame(rows, columns=["年", "nA", "nB", "A", "B", "badA", "badB"])


def report_relax(title: str, d: pd.DataFrame, maskA, maskB, labA: str, labB: str,
                 pool_relaxed: pd.DataFrame, pool_base: pd.DataFrame, dmap: dict):
    """一个放宽假设的完整报告：边际组逐年 + 整池配对 bootstrap。"""
    P("\n" + "=" * 104)
    P(title)
    P("=" * 104)

    # ---- 1. 边际组逐年（收益 + 风险） ----
    a_all, b_all = d[maskA(d)], d[maskB(d)]
    ga, gb = a_all["ab_fwd60"].mean(), b_all["ab_fwd60"].mean()
    P("  边际组 A（%s）n=%d 均值 %+.2f%% | 原门内 B（%s）n=%d 均值 %+.2f%% | A−B %+.2fpp" % (
        labA, len(a_all), ga * 100, labB, len(b_all), gb * 100, (ga - gb) * 100))
    tb = year_ab(d, maskA, maskB)
    P("  %-6s %6s %6s %9s %9s %9s | %8s %8s %9s  %s" % (
        "年", "nA", "nB", "A均值", "B均值", "A−B", "A大亏率", "B大亏率", "大亏率差", "判定"))
    P("  " + "-" * 100)
    ok_ret = tot_ret = ok_bad = tot_bad = 0
    for _, r in tb.iterrows():
        if r["nA"] < MIN_N or r["nB"] < MIN_N or pd.isna(r["A"]):
            P("  %-6d %6d %6d %9s %9s %9s | %8s %8s %9s  样本稀薄" % (
                r["年"], r["nA"], r["nB"], "—", "—", "—", "—", "—", "—"))
            continue
        diff = r["A"] - r["B"]
        same_ret = (diff > 0) == (ga > gb)
        bad_diff = r["badA"] - r["badB"]
        # 风险维度：A 是边际放入组，期望它不更危险 → badA <= badB 为"一致"
        same_bad = bad_diff <= 0
        tot_ret += 1
        tot_bad += 1
        ok_ret += 1 if same_ret else 0
        ok_bad += 1 if same_bad else 0
        P("  %-6d %6d %6d %8.2f%% %8.2f%% %8.2fpp | %7.1f%% %7.1f%% %8.1fpp  收益%s/风险%s" % (
            r["年"], r["nA"], r["nB"], r["A"] * 100, r["B"] * 100, diff * 100,
            r["badA"] * 100, r["badB"] * 100, bad_diff * 100,
            "✓" if same_ret else "✗", "✓" if same_bad else "✗"))
    P("  " + "-" * 100)
    P("  【收益方向一致性】%d/%d 年   【风险方向一致性（A 不更危险）】%d/%d 年   及格线 6/8" % (
        ok_ret, tot_ret, ok_bad, tot_bad))

    # ---- 2. 整池配对区块 bootstrap（放宽池 − 现池） ----
    P("\n  整池对比（放宽池 − 现池，配对区块 bootstrap，90%CI）：")
    P("  %-8s %9s %9s %10s %22s %10s" % ("范围", "现池日均票", "放宽日均票", "配对日", "差值", "判定"))
    for scope_name, sub_r, sub_b in [
        ("全样本", pool_relaxed, pool_base),
        ("UP", pool_relaxed[pool_relaxed["regime"] == "UP"], pool_base[pool_base["regime"] == "UP"]),
        ("CHOP", pool_relaxed[pool_relaxed["regime"] == "CHOP"], pool_base[pool_base["regime"] == "CHOP"]),
        ("DOWN", pool_relaxed[pool_relaxed["regime"] == "DOWN"], pool_base[pool_base["regime"] == "DOWN"]),
    ]:
        if len(sub_r) < 10 or len(sub_b) < 10:
            P("  %-8s 样本不足" % scope_name)
            continue
        pt, lo, hi, nb = paired_block_boot(sub_r, sub_b, dmap)
        per_r = sub_r.groupby("date").size().mean()
        per_b = sub_b.groupby("date").size().mean()
        if pd.isna(pt):
            P("  %-8s %9.2f %9.2f 配对不足" % (scope_name, per_b, per_r))
            continue
        P("  %-8s %9.2f %9.2f %10d %+8.2fpp CI[%+6.2f,%+6.2f] %s" % (
            scope_name, per_b, per_r, nb, pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))

    return dict(ret=(ok_ret, tot_ret), bad=(ok_bad, tot_bad))


def main() -> None:
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 104)
    P("四门放宽实证 · 2026-11-10")
    P("=" * 104)
    P("事件表 events_fixed.pkl：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min().date(), ev["date"].max().date()))
    P("指标：ab_fwd60 = fwd60 − beta×mkt_fwd60（β调整60日超额）；大亏 = ab_fwd60<−20%")
    P("基准门：prox≥0.95 / 箱宽≤15% / 额≥4亿 / 换手≤6% / ret60≤19%")
    P("注意：每个假设都在【不含被测门】的样本上做边际组检验，避免幸存者偏差。")

    results = {}

    # ================= H1：prox 0.95 → 0.90 =================
    d_noprox = gate(ev, drop="prox")
    results["H1 prox 0.90"] = report_relax(
        "H1 距年高 prox_now：0.95 → 0.90", d_noprox,
        lambda x: (x["prox_now"] >= 0.90) & (x["prox_now"] < 0.95),
        lambda x: x["prox_now"] >= 0.95,
        "边际 0.90~0.95", "原门 ≥0.95",
        gate(ev, over=dict(prox=0.90)), gate(ev), dmap)

    # ================= H2：箱宽 15% → 20% =================
    d_nowidth = gate(ev, drop="width")
    results["H2 箱宽 20%"] = report_relax(
        "H2 箱宽 width30：15% → 20%", d_nowidth,
        lambda x: (x["width30"] > 0.15) & (x["width30"] <= 0.20),
        lambda x: x["width30"] <= 0.15,
        "边际 15%~20%", "原门 ≤15%",
        gate(ev, over=dict(width=0.20)), gate(ev), dmap)

    # ================= H3：换手 6% → 7% =================
    d_noturn = gate(ev, drop="turn")
    results["H3 换手 7%"] = report_relax(
        "H3 换手 turn30：6% → 7%", d_noturn,
        lambda x: (x["turn30"] > 6.0) & (x["turn30"] <= 7.0),
        lambda x: x["turn30"] <= 6.0,
        "边际 6%~7%", "原门 ≤6%",
        gate(ev, over=dict(turn=7.0)), gate(ev), dmap)

    # ================= H4：删 ret60 门 =================
    d_noret = gate(ev, drop="ret60")
    results["H4 删ret60"] = report_relax(
        "H4 删除 ret60≤19% 门（复核 10-29 结论）", d_noret,
        lambda x: pd.to_numeric(x["ret60"], errors="coerce") > 0.19,
        lambda x: pd.to_numeric(x["ret60"], errors="coerce") <= 0.19,
        "边际 ret60>19%", "原门 ≤19%",
        gate(ev, drop="ret60", over=dict(ret60=np.inf)), gate(ev), dmap)

    # ================= 汇总 =================
    P("\n" + "=" * 104)
    P("汇总：四门放宽假设裁决")
    P("=" * 104)
    P("  %-16s %-14s %-14s %s" % ("假设", "收益一致性", "风险一致性", "初步判定"))
    P("  " + "-" * 96)
    for k, v in results.items():
        (ok_r, tot_r), (ok_b, tot_b) = v["ret"], v["bad"]
        P("  %-16s %-14s %-14s" % (k, "%d/%d" % (ok_r, tot_r), "%d/%d" % (ok_b, tot_b)))
    P("\n  判定口径：放宽成立的条件是【边际组不更差】——收益 A−B 一致为负 或 风险 A 大亏率")
    P("  一致更高，即拒绝放宽。整池配对 bootstrap 显著为负同样构成拒绝证据。")
    P("  耗时 %.1fs" % (time.time() - t0))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
