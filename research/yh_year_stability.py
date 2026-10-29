# -*- coding: utf-8 -*-
"""
分年度方向一致性检验（过拟合自查第二条）
================================================================
背景：
  yh_d3_d6_rs.py 给出的是【全样本点估计 + 单次配对区块 bootstrap】。
  但按本项目过拟合自查纪律，一个因子要升级为"门槛"，必须过三条：
    ① 样本量够不够  ② 跨年份方向是否一致  ③ 近期方向是否还在
  本脚本专测第 ② 条：逐年拆开，看"预期好的组 − 预期差的组"是否年年同号。

对照组：ret60<=19% 门（已采用的门槛，历史 8/8 年一致）作为及格线参照。

判定口径：
  - 有效年：该年两组事件数均 >= 8，否则标记"样本稀薄"不计入方向统计
  - 方向一致：该年 (A组均值 − B组均值) 与全样本方向同号
  - 及格线：>= 6/8 年（对照 ret60 为 8/8）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_d3_d6_rs import load_full, gate_v2, build_kline_feat, attach_sjltz, P, _lines  # noqa: E402


def gate_struct(ev):
    """仅结构门（prox_now/箱宽/成交额/换手），不含 ret60<=19%。
    用于检验 ret60 本身——若样本已含 ret60 门，高涨幅组会被剔光（幸存者偏差）。"""
    return ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) &
              (ev["amt30"] >= 4e8) & (ev["turn30"] <= 6.0) &
              ev["ab_fwd60"].notna()].copy()
from yh_blockboot_all import make_dmap, block_boot, daily  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "year_stability.txt")
MIN_N = 8


def year_table(d: pd.DataFrame, maskA, maskB, labA, labB) -> tuple:
    """逐年 A/B 两组均值、胜率、大亏率与差值。"""
    d = d.copy()
    d["_y"] = d["date"].dt.year
    rows = []
    for y, g in d.groupby("_y"):
        a, b = g[maskA(g)], g[maskB(g)]
        if len(a) == 0 or len(b) == 0:
            rows.append((y, len(a), len(b), np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        ma, mb = a["ab_fwd60"].mean(), b["ab_fwd60"].mean()
        rows.append((y, len(a), len(b), ma, mb, ma - mb,
                     (a["ab_fwd60"] > 0).mean() - (b["ab_fwd60"] > 0).mean(),
                     (a["ab_fwd60"] < -0.20).mean() - (b["ab_fwd60"] < -0.20).mean()))
    return pd.DataFrame(rows, columns=["年", "nA", "nB", "A", "B", "差",
                                       "胜率差", "大亏率差"]), None


def report(d: pd.DataFrame, maskA, maskB, labA, labB, title, note=""):
    P("\n" + "=" * 100)
    P(title)
    P("=" * 100)
    if note:
        P("  " + note)
    # 全样本方向
    a, b = d[maskA(d)], d[maskB(d)]
    ga, gb = a["ab_fwd60"].mean(), b["ab_fwd60"].mean()
    full_sign = 1 if ga > gb else -1
    P("  全样本：A(%s) n=%d 均值 %+.2f%%  |  B(%s) n=%d 均值 %+.2f%%  |  差 %+.2fpp%s" % (
        labA, len(a), ga * 100, labB, len(b), gb * 100, (ga - gb) * 100,
        "（A 优于 B，预期方向为正）" if full_sign > 0 else "（B 优于 A，预期方向为负）"))
    P("")
    P("  %-6s %6s %6s %10s %10s %10s %10s %10s  %s" % (
        "年", "nA", "nB", "A均值", "B均值", "差值", "胜率差", "大亏率差", "判定"))
    P("  " + "-" * 96)
    tb, _ = year_table(d, maskA, maskB, labA, labB)
    ok = tot = 0
    for _, r in tb.iterrows():
        if r["nA"] < MIN_N or r["nB"] < MIN_N or pd.isna(r["差"]):
            P("  %-6d %6d %6d %10s %10s %10s %10s %10s  样本稀薄" % (
                r["年"], r["nA"], r["nB"], "—", "—", "—", "—", "—"))
            continue
        tot += 1
        same = (r["差"] > 0) == (full_sign > 0)
        ok += 1 if same else 0
        P("  %-6d %6d %6d %9.2f%% %9.2f%% %9.2fpp %9.1fpp %9.1fpp  %s" % (
            r["年"], r["nA"], r["nB"], r["A"] * 100, r["B"] * 100, r["差"] * 100,
            r["胜率差"] * 100, r["大亏率差"] * 100, "✓一致" if same else "✗反向"))
    P("  " + "-" * 96)
    P("  【方向一致性】%d / %d 年（有效年）  —— 及格线 6/8" % (ok, tot))
    return ok, tot


def badrate_year(d: pd.DataFrame, maskA, maskB, labA, labB, title, invert=False):
    """大亏率（ab_fwd60 < -20%）逐年对照。ret60 门正是靠这一项 8/8 通过的，
    故三个新因子也必须过这一关才算公平比较。
    invert=True 表示经典预期 B 组更安全（用于反经典因子）。"""
    P("\n" + "=" * 100)
    P(title)
    P("=" * 100)
    dd = d.copy()
    dd["_y"] = dd["date"].dt.year
    dd["_bad"] = dd["ab_fwd60"] < -0.20
    a_all, b_all = dd[maskA(dd)], dd[maskB(dd)]
    ba, bb = a_all["_bad"].mean(), b_all["_bad"].mean()
    expect = "A 更安全" if ((ba < bb) != invert) else "B 更安全"
    P("  全样本：A(%s) n=%d 大亏率 %.1f%%  |  B(%s) n=%d 大亏率 %.1f%%  →  %s" % (
        labA, len(a_all), ba * 100, labB, len(b_all), bb * 100, expect))
    P("")
    P("  %-6s %8s %8s %14s %14s %10s  %s" % (
        "年", "nA", "nB", "A大亏率", "B大亏率", "差", "判定"))
    P("  " + "-" * 78)
    ok = tot = 0
    for y, g in dd.groupby("_y"):
        lo_, hi_ = g[maskA(g)], g[maskB(g)]
        if len(lo_) < MIN_N or len(hi_) < MIN_N:
            P("  %-6d %8d %8d %14s %14s %10s  样本稀薄" % (y, len(lo_), len(hi_), "—", "—", "—"))
            continue
        bl, bh = lo_["_bad"].mean(), hi_["_bad"].mean()
        tot += 1
        same = (bl < bh) != invert
        ok += 1 if same else 0
        P("  %-6d %8d %8d %13.1f%% %13.1f%% %9.1fpp  %s" % (
            y, len(lo_), len(hi_), bl * 100, bh * 100, (bl - bh) * 100,
            "✓一致" if same else "✗反向"))
    P("  " + "-" * 78)
    P("  【大亏率方向一致性】%d / %d 年" % (ok, tot))
    return ok, tot


def main():
    ev = load_full()
    d = gate_v2(ev)
    dmap = make_dmap(ev)
    P("=" * 100)
    P("分年度方向一致性检验 · 当前 v2 完整结构门内（含 ret60<=19%）")
    P("事件 %d 条，%d 只，%d 个交易日" % (len(d), d["code"].nunique(), d["date"].nunique()))
    P("=" * 100)

    feat = build_kline_feat()
    d = pd.merge(d, feat[["code", "date", "vr", "rs_pct"]], on=["code", "date"], how="left")
    d = attach_sjltz(d)
    P("\n  覆盖率：vr %.1f%%  rs_pct %.1f%%  SJLTZ %.1f%%" % (
        100 * d["vr"].notna().mean(), 100 * d["rs_pct"].notna().mean(),
        100 * d["SJLTZ"].notna().mean()))

    res = {}
    res_bad = {}

    # 对照组必须在【未加 ret60 门】的结构门样本上做，否则高涨幅组已被剔光（幸存者偏差）
    d0 = gate_struct(ev)
    P("\n  【对照组样本】仅结构门（不含 ret60 上限门）：%d 条，%d 只" % (
        len(d0), d0["code"].nunique()))

    # ---------- 对照组 1：ret60 收益口径 ----------
    med = d0["ret60"].astype(float).median()
    P("\n  【对照组 1】ret60 分组中位数 = %.1f%%" % (med * 100))
    res["ret60"] = report(
        d0, lambda x: x["ret60"].astype(float) <= med,
        lambda x: x["ret60"].astype(float) > med,
        "低涨幅", "高涨幅",
        "对照组 1 · ret60 前期涨幅 · 收益口径（样本为仅结构门，避免幸存者偏差）",
        "注：用中位数切分（非 19% 阈值），只为检验方向一致性")

    # ---------- 对照组 2：19% 阈值下的【大亏率】逐年一致性 ----------
    P("\n  【对照组 2】ret60<=19% 阈值 · 大亏率（<-20%）逐年对照（样本为仅结构门）")
    P("  %-6s %8s %8s %14s %14s %10s  %s" % (
        "年", "n低", "n高", "低涨幅大亏率", "高涨幅大亏率", "差", "判定"))
    P("  " + "-" * 78)
    dd = d0.copy()
    dd["_y"] = dd["date"].dt.year
    dd["_bad"] = dd["ab_fwd60"] < -0.20
    ok2 = tot2 = 0
    for y, g in dd.groupby("_y"):
        lo_ = g[g["ret60"].astype(float) <= 0.19]
        hi_ = g[g["ret60"].astype(float) > 0.19]
        if len(lo_) < MIN_N or len(hi_) < MIN_N:
            P("  %-6d %8d %8d %14s %14s %10s  样本稀薄" % (y, len(lo_), len(hi_), "—", "—", "—"))
            continue
        bl, bh = lo_["_bad"].mean(), hi_["_bad"].mean()
        tot2 += 1
        same = bl < bh
        ok2 += 1 if same else 0
        P("  %-6d %8d %8d %13.1f%% %13.1f%% %9.1fpp  %s" % (
            y, len(lo_), len(hi_), bl * 100, bh * 100, (bl - bh) * 100,
            "✓低涨幅更安全" if same else "✗反向"))
    P("  " + "-" * 78)
    P("  【大亏率方向一致性】%d / %d 年 —— ret60 门当年正是靠这一项 8/8 通过的" % (ok2, tot2))
    res_bad["ret60"] = (ok2, tot2)

    # ---------- D3 量能 ----------
    res["vr"] = report(
        d, lambda x: x["vr"] >= 0.8, lambda x: x["vr"] < 0.8,
        "非缩量 vr>=0.8", "缩量 vr<0.8",
        "D3 量能确认（vr = 当日成交额 / 近20日日均成交额）",
        "若缩量确实更差，则每年 A−B 应为正")
    res_bad["vr"] = badrate_year(
        d, lambda x: x["vr"] >= 0.8, lambda x: x["vr"] < 0.8,
        "非缩量 vr>=0.8", "缩量 vr<0.8",
        "D3 量能 · 风险维度（大亏率逐年）")

    # ---------- RS 相对强度 ----------
    res["rs"] = report(
        d, lambda x: x["rs_pct"] >= 70, lambda x: x["rs_pct"] < 70,
        "强 RS>=70", "弱 RS<70",
        "RS 相对强度（rs_pct = 当日全市场 ret250 百分位）",
        "若 RS 高确实更好，则每年 A−B 应为正")
    res_bad["rs"] = badrate_year(
        d, lambda x: x["rs_pct"] >= 70, lambda x: x["rs_pct"] < 70,
        "强 RS>=70", "弱 RS<70",
        "RS 相对强度 · 风险维度（大亏率逐年）")

    # ---------- D6 净利同比 ----------
    ds = d[d["SJLTZ"].notna()]
    res["sjltz"] = report(
        ds, lambda x: x["SJLTZ"] < 0, lambda x: x["SJLTZ"] >= 25,
        "净利负增长", "净利>=25%(CANSLIM C)",
        "D6 净利同比 SJLTZ（CANSLIM 的 C），仅 SJLTZ 覆盖段 n=%d" % len(ds),
        "经典预期是 B 更好；若实测 A−B 为正，说明门内反经典")
    res_bad["sjltz"] = badrate_year(
        ds, lambda x: x["SJLTZ"] < 0, lambda x: x["SJLTZ"] >= 25,
        "净利负增长", "净利>=25%(CANSLIM C)",
        "D6 净利同比 · 风险维度（大亏率逐年）")

    # ---------- 汇总 ----------
    P("\n" + "=" * 100)
    P("汇总：跨年份方向一致性（及格线 6/8，对照 ret60 为基准）")
    P("=" * 100)
    P("  %-10s %-16s %-16s %s" % ("因子", "收益一致性", "大亏率一致性", "结论"))
    P("  " + "-" * 78)
    verdict = {
        "ret60": "对照组：收益弱(3/8)、风险极稳(8/8) → 定位「风险门」是自洽的",
        "vr": "收益 4/8 且前四年全反向 → 近期效应，不加门槛",
        "rs": "收益 5/8，2021 极端反向 → 只做标注",
        "sjltz": "收益 3/6，近两年连续反向 → 门内无稳定方向，不加任何增速门",
    }
    for k in ["ret60", "vr", "rs", "sjltz"]:
        ok, tot = res[k]
        okb, totb = res_bad.get(k, (0, 0))
        P("  %-10s %-16s %-16s %s" % (
            k, "%d / %d" % (ok, tot), "%d / %d" % (okb, totb), verdict[k]))
    P("")
    P("  及格线：任一维度需 >= 6/8 才考虑升级为门槛（对照组 ret60 靠风险维度 8/8 通过）")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
