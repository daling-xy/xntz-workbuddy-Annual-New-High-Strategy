# -*- coding: utf-8 -*-
"""
各项筛选门槛的【边际贡献 / 票量成本】（修正版）

上一版缺陷：以「六道门全开」为基准做 leave-one-out，但全开后在 DOWN 只剩 38 个事件，
基准本身不显著 → 由此算出的增量不可靠。

修正：改为【递进式】——先落结构门（prox/箱宽/额/换手），再逐道叠加基本面门
（营收>=30% / 上期>0 / 营收规模>=2亿），每一步用「上一步」做配对 bootstrap 参照。
这既贴近实际选股流程，也保证参照组有足够样本。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_regime import boot_ci_daily, paired_boot, sig  # noqa: E402
from yh_rev_fine import load_events, load_fin, attach_fin  # noqa: E402

STRUCT = {
    "prox_now>=0.95": lambda d: d["prox_now"] >= 0.95,
    "箱宽<=15%":      lambda d: d["width30"] <= 0.15,
    "额>=4亿":        lambda d: d["amt30"] >= 4e8,
    "换手<=6%":       lambda d: d["turn30"] <= 6.0,
}


def bar(t):
    print("=" * 124)
    print(t)
    print("=" * 124)


def show(label, d, ref=None):
    n = len(d)
    nd = d["date"].nunique()
    per = n / max(nd, 1)
    pt, lo, hi = boot_ci_daily(d)
    line = ("  %-30s n=%5d 日=%4d 日均%5.2f只 β%5.2f %+8.2f%% CI[%+7.2f,%+7.2f] 胜率%5.1f%% 大亏%5.1f%% %s"
            % (label, n, nd, per, d["beta"].median(), pt * 100, lo * 100, hi * 100,
               float((d["fwd60"] > 0).mean() * 100),
               float((d["fwd60"] < -0.20).mean() * 100), sig(pt, lo, hi)))
    if ref is not None and ref is not d:
        df, dlo, dhi, nday = paired_boot(d, ref)
        line += ("\n  %-30s %34s 相对上一步 %+7.2fpp [%+6.2f,%+6.2f] (%3d日) %s"
                 % ("", "", df * 100, dlo * 100, dhi * 100, nday, sig(df, dlo, dhi)))
    print(line)
    return d


def run(rg):
    ev = load_events()
    fin = load_fin()
    e = attach_fin(ev, fin)
    e = e.sort_values(["code", "date"])
    e["YSTZ_prev"] = e.groupby("code")["YSTZ"].shift(1)
    sub = e[(e["regime"] == rg) & e["YSTZ"].notna() & e["YSTZ_prev"].notna()].copy()

    bar("%s 周期 · 递进式门槛贡献（每一步以上一步为参照）" % rg)
    print("  起始候选池: %d 事件 / %d 交易日" % (len(sub), sub["date"].nunique()))
    print()

    cur = sub.copy()
    print("  ▶ 第0步：无门（全池）")
    cur = show("无门", cur)
    print()

    print("  ▶ 结构门（v2 四项）")
    for k, fn in STRUCT.items():
        nxt = cur[fn(cur)]
        show("+" + k, nxt, cur)
        cur = nxt
    struct = cur
    print()

    print("  ▶ 基本面门（在结构门之上叠加）")
    steps = [
        ("营收>=30%", lambda d: d["YSTZ"] >= 30.0),
        ("上期营收>0", lambda d: d["YSTZ_prev"] > 0),
        ("单期营收>=2亿", lambda d: d["revenue"] >= 2e8) if "revenue" in cur.columns else None,
    ]
    steps = [s for s in steps if s]
    having = struct
    print("  %-30s %5s %6s %10s %14s" % ("配置", "n", "日", "日均票", "相对上一步"))
    print("  " + "-" * 74)
    for name, fn in steps:
        nxt = having[fn(having)]
        pt, lo, hi = boot_ci_daily(nxt)
        nnd = nxt["date"].nunique()
        if having is struct:
            ref_label = "结构门"
        else:
            ref_label = "上一步"
        df, dlo, dhi, nday = paired_boot(nxt, having)
        print("  %-30s %5d %6d %10.2f  %+8.2fpp [%+6.2f,%+6.2f] (%d日 vs %s) %s"
              % ("+" + name, len(nxt), nnd, len(nxt) / max(nnd, 1),
                 df * 100, dlo * 100, dhi * 100, nday, ref_label, sig(df, dlo, dhi)))
        having = nxt
    print()

    # ---------- 单独放松基本面门：对照 ----------
    bar("%s 周期 · 只放松营收门槛这一道，其余不变" % rg)
    base_full = struct[(struct["YSTZ"] >= 30.0) & (struct["YSTZ_prev"] > 0)]
    if "revenue" in base_full.columns:
        base_full = base_full[base_full["revenue"] >= 2e8]
    print("  %-30s %5s %6s %10s %12s %16s" % ("配置", "n", "日", "日均票", "超额", "vs 全纪律"))
    print("  " + "-" * 84)
    show_row = lambda lb, d: print(
        "  %-30s %5d %6d %10.2f  %+8.2f%%  %+8.2fpp [%+6.2f,%+6.2f] %s"
        % (lb, len(d), d["date"].nunique(), len(d) / max(d["date"].nunique(), 1),
           boot_ci_daily(d)[0] * 100,
           paired_boot(d, base_full)[0] * 100,
           paired_boot(d, base_full)[1] * 100,
           paired_boot(d, base_full)[2] * 100,
           sig(*boot_ci_daily(d)))) if len(d) >= 15 and d["date"].nunique() >= 8 else print(
        "  %-30s %5d  样本不足" % (lb, len(d)))
    show_row("全纪律(营收30)", base_full)
    for thr in [0, 10, 15, 20, 25]:
        lb = struct[(struct["YSTZ"] >= thr) & (struct["YSTZ_prev"] > 0)]
        if "revenue" in lb.columns:
            lb = lb[lb["revenue"] >= 2e8]
        show_row("营收门槛>=%d" % thr, lb)
    print()

    # ---------- 等票量：放松营收 vs 收紧箱宽 ----------
    bar("%s 周期 · 等票量对照（在相近日均票下比表现）" % rg)
    print("  目的：检验用户的问题——「放松营收换票量」是否等价于「维持营收但收紧别的」")
    print()
    rows = []
    for rev in [0, 15, 20, 25, 30, 40]:
        for w in [0.15, 0.12, 0.10]:
            d = struct[(struct["width30"] <= w) & (struct["YSTZ"] >= rev) &
                       (struct["YSTZ_prev"] > 0)]
            if "revenue" in d.columns:
                d = d[d["revenue"] >= 2e8]
            if len(d) < 15 or d["date"].nunique() < 8:
                continue
            pt, lo, hi = boot_ci_daily(d)
            nnd = d["date"].nunique()
            rows.append((rev, w, len(d), len(d) / nnd, pt, lo, hi))
    if rows:
        rows.sort(key=lambda r: r[3])
        print("  %-8s %-8s %6s %10s %14s %14s" % ("营收门槛", "箱宽", "n", "日均票", "超额", "90CI"))
        print("  " + "-" * 70)
        for rev, w, n, per, pt, lo, hi in rows:
            print("  >=%-6d %-8s %6d %10.2f  %+8.2f%%  [%+7.2f,%+7.2f] %s"
                  % (rev, "%d%%" % (w * 100), n, per, pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))
    print()


def main():
    for rg in ["DOWN", "UP"]:
        run(rg)
        print()


if __name__ == "__main__":
    main()
