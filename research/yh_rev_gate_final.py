# -*- coding: utf-8 -*-
"""
【营收增速门槛 · 最终裁决】(2026-10-27)

要回答的问题只剩一个：在已通过结构门（距年高>=0.95 / 箱宽<=15% / 额>=4亿 / 换手<=6%）
的池子里，营收增速这条规则【是否还有存在的必要】？

分两类检验：
  A 类（alpha 筛选）：设下限 X  vs  不设门槛   —— 加了这条能否提高收益？
  B 类（风险排雷）  ：仅剔除极端负增长（YSTZ < -30% / -50%）  vs  不剔除
                     —— 这条不问收益，问的是"不排会不会踩雷"（用左尾指标衡量）

方法：配对区块 bootstrap（区块=60交易日，配对=只在两边都有观测的交易日做差）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin          # noqa: E402
from yh_blockboot_all import make_dmap, daily, paired_block_boot, block_boot  # noqa: E402

RNG = np.random.default_rng(20261027)

# v2 结构门（不含任何营收条件）
STRUCT = dict(prox=0.95, width=0.15, amt=4e8, turn=6.0)


def struct_gate(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d["prox_now"] >= STRUCT["prox"]) &
             (d["width30"] <= STRUCT["width"]) &
             (d["amt30"] >= STRUCT["amt"]) &
             (d["turn30"] <= STRUCT["turn"])]


def sig(pt, lo, hi):
    if any(pd.isna(x) for x in (pt, lo, hi)):
        return "??"
    if lo > 0:
        return "显著为正"
    if hi < 0:
        return "显著为负"
    return "含零"


def main() -> None:
    ev = load_events()
    fin = load_fin()
    ev = attach_fin(ev, fin).sort_values(["code", "date"])
    ev["YSTZ_prev"] = ev.groupby("code")["YSTZ"].shift(1)
    dmap = make_dmap(ev)

    base = struct_gate(ev)
    base = base[base["YSTZ"].notna()]
    print("=" * 104)
    print("营收增速门槛 · 最终裁决   结构门: prox>=0.95 width<=15% amt>=4亿 turn<=6%  (不含营收条件)")
    print("=" * 104)
    print(f"事件表 {len(ev):,} 行 / {ev['date'].nunique()} 交易日 → 区块 {int(np.ceil(ev['date'].nunique()/60))} 个")
    print(f"结构门内且营收非空: {len(base):,} 条")

    SCOPES = [("全样本", base)]
    for rg in ["UP", "CHOP", "DOWN"]:
        SCOPES.append((rg, base[base["regime"] == rg]))

    # ---------------------------------------------------------- A. 样本规模
    print("\n" + "-" * 104)
    print("A. 各周期到底有多少『独立信息』（这是后面所有 CI 宽度的根源）")
    print("-" * 104)
    print("%-8s %10s %10s %10s %10s %12s" % ("周期", "事件数", "交易日", "区块数", "日均票", "占交易日%"))
    for name, sub in SCOPES:
        nd = sub["date"].nunique()
        ndays_all = ev["date"].nunique()
        bis = pd.to_datetime(sub["date"]).map(lambda x: dmap.get(pd.Timestamp(x), np.nan))
        nb = len(np.unique(bis.dropna().astype(int) // 60))
        print("%-8s %10s %10d %10d %10.2f %11.1f%%" % (
            name, f"{len(sub):,}", nd, nb, len(sub) / max(nd, 1), nd / ndays_all * 100))

    # ---------------------------------------------------------- B. alpha 筛选
    print("\n" + "-" * 104)
    print("B. 【alpha 筛选】设营收下限 vs 完全不设门槛  （配对区块 bootstrap，差值 = 设门槛 − 不设）")
    print("-" * 104)
    print("%-8s %10s %12s %10s %10s %10s %10s  %s" % (
        "周期", "门槛", "日均票", "点估计", "差值pp", "90%CI", "区块", "判定"))

    for name, sub in SCOPES:
        ref = sub  # 不设门槛
        for thr in [0.0, 10.0, 20.0, 30.0, 50.0]:
            alt = sub[sub["YSTZ"] >= thr]
            pt, lo, hi, nb = paired_block_boot(alt, ref, dmap)
            v, _, _, _ = block_boot(daily(alt), dmap)
            nd = alt["date"].nunique()
            print("%-8s %10s %12.2f %9.2f%% %+9.2f [%+5.2f,%+5.2f] %8d  %s" % (
                name, f">={thr:.0f}%", len(alt) / max(nd, 1), v * 100,
                pt * 100, lo * 100, hi * 100, nb, sig(pt, lo, hi)))
        print()

    # ---------------------------------------------------------- C. 风险排雷
    print("-" * 104)
    print("C. 【风险排雷】仅剔除极端负增长 vs 完全不剔除  （不问收益，问左尾与胜率）")
    print("-" * 104)
    print("%-8s %14s %10s %10s %10s %10s %10s  %s" % (
        "周期", "规则", "日均票", "收益差pp", "90%CI", "胜率差pp", "大亏差pp", "判定"))
    for name, sub in SCOPES:
        for cut, lab in [(-30.0, "剔除 <-30%"), (-50.0, "剔除 <-50%")]:
            alt = sub[sub["YSTZ"] >= cut]
            pt, lo, hi, nb = paired_block_boot(alt, sub, dmap)
            # 左尾指标：胜率与大亏率（跌超20%）的绝对水平差
            w_ref = float((sub["fwd60"] > 0).mean())
            w_alt = float((alt["fwd60"] > 0).mean())
            b_ref = float((sub["fwd60"] <= -0.20).mean())
            b_alt = float((alt["fwd60"] <= -0.20).mean())
            nd = alt["date"].nunique()
            print("%-8s %14s %10.2f %+9.2f [%+5.2f,%+5.2f] %+9.2f %+9.2f  %s" % (
                name, lab, len(alt) / max(nd, 1), pt * 100, lo * 100, hi * 100,
                (w_alt - w_ref) * 100, (b_alt - b_ref) * 100, sig(pt, lo, hi)))
        # 参照：负增长组自身的绝对表现
        neg = sub[sub["YSTZ"] < 0]
        if len(neg) > 200:
            v, l, h, _ = block_boot(daily(neg), dmap)
            print("%-8s %14s  n=%-7d 收益 %.2f%% [%+.2f,%+.2f]  胜率%.1f%%  大亏%.1f%%" % (
                name, "【负增长组】", len(neg), v * 100, l * 100, h * 100,
                float((neg["fwd60"] > 0).mean()) * 100,
                float((neg["fwd60"] <= -0.20).mean()) * 100))
        pos = sub[sub["YSTZ"] >= 30]
        if len(pos) > 200:
            v, l, h, _ = block_boot(daily(pos), dmap)
            print("%-8s %14s  n=%-7d 收益 %.2f%% [%+.2f,%+.2f]  胜率%.1f%%  大亏%.1f%%" % (
                name, "【>=30%组】", len(pos), v * 100, l * 100, h * 100,
                float((pos["fwd60"] > 0).mean()) * 100,
                float((pos["fwd60"] <= -0.20).mean()) * 100))
        print()

    # ---------------------------------------------------------- D. 门槛的代价
    print("-" * 104)
    print("D. 保留门槛要付出什么（UP 期 ≥30% 为例）")
    print("-" * 104)
    up = base[base["regime"] == "UP"]
    for thr in [0.0, 30.0]:
        alt = up[up["YSTZ"] >= thr]
        nd = alt["date"].nunique()
        print("  UP ≥%-4.0f%%  事件 %7d  日均票 %5.2f  覆盖交易日 %4d (%.0f%%)  β中位 %.2f" % (
            thr, len(alt), len(alt) / max(nd, 1), nd, nd / up["date"].nunique() * 100,
            alt["beta"].median()))
    nd = up["date"].nunique()
    print("  UP 不设    事件 %7d  日均票 %5.2f  覆盖交易日 %4d (%.0f%%)  β中位 %.2f" % (
        len(up), len(up) / nd, nd, 100.0, up["beta"].median()))


if __name__ == "__main__":
    main()
