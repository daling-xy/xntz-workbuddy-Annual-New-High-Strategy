# -*- coding: utf-8 -*-
"""
【为什么中信证券能进 0.85 档？】—— 检验「箱宽小」到底代表什么

缘起：2026-10-23 实盘，中信证券 600030 出现在「当前价/年高>=0.85 且 箱宽<=15%」这一档。
但它距年高 −14.6%、近60日 −10.7%、箱内位置只有 9%（贴在箱底）。
也就是说它属于「跌下来之后在低位窄幅整理」，而不是「高位窄幅蓄势」。

箱宽（30日振幅）只衡量【波动幅度】，不衡量【价格在箱中的位置】。
本脚本检验：在放宽档（0.85~0.95）里，用 pos_in_box 把「箱底横盘」剔掉有没有用。

数据来源：腾讯前复权日线（全历史）/ 通达信指数日线；事件表 2019-06~2026-07。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1000
_rng = np.random.default_rng(20261023)


def boot_ci(sub, key, n=BOOT):
    """按交易日整簇重抽样：先按 date 聚成日均值，再对日均值序列有放回重抽"""
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def load():
    p = os.path.join(CACHE, "events_fixed.pkl")
    ev = pd.read_pickle(p if os.path.exists(p) else os.path.join(CACHE, "events_long_beta.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    return ev


def stat(tag, d):
    if len(d) < 30:
        print(f"  {tag:<34} n={len(d):>6}  样本不足")
        return
    est, lo, hi = boot_ci(d, "ab_fwd60")
    ex, _, _ = boot_ci(d, "ex_fwd60")
    win = float((d["ab_fwd60"] > 0).mean())
    sig = "显著为正" if (lo > 0) else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<34} n={len(d):>6}  β中位 {d['beta'].median():>5.2f}  "
          f"β调整 {est*100:+6.2f}%  90%CI [{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率 {win*100:4.1f}%  {sig}")


def ladder_evidence(regime="DOWN"):
    """
    返回「下跌市 + 完整纪律」下各 prox_now 档位的历史实证，
    供 run_screen_v2.py 在实盘阶梯表里标注，避免把「无效档」误当成推荐。
    返回 {档位tag: {"n":int, "est":str, "ci":str, "verdict":str}}
    """
    ev_raw = load()
    ev = ev_raw[ev_raw["beta"].notna() & ev_raw["mkt_fwd60"].notna() & ev_raw["fwd60"].notna()]
    dn = ev[ev["regime"] == regime]
    g = dn[(dn["width30"] <= 0.15) & (dn["amt30"] >= 4e8) & (dn["turn30"] <= 6.0)]

    # 2026-10-27 起改用【区块 bootstrap】：fwd60 窗口重叠 98%，按天抽样会低估
    # 标准误、把噪声判成 alpha。区块=60交易日，全局约 29 块。
    try:
        from yh_blockboot_all import make_dmap, daily, block_boot
        dmap = make_dmap(ev_raw)
        use_block = True
    except Exception:
        use_block = False
        dmap = None

    out = {}
    for tag, lo_, hi_ in [("0.95", 0.95, 1.01), ("0.90", 0.90, 0.95),
                          ("0.85", 0.85, 0.90), ("0.80", 0.80, 0.85)]:
        d = g[(g["prox_now"] >= lo_) & (g["prox_now"] < hi_)]
        if len(d) < 30:
            out[tag] = {"n": len(d), "est": "—", "ci": "—", "verdict": "样本不足"}
            continue
        if use_block:
            est, lo, hi, nb = block_boot(daily(d, "ab_fwd60"), dmap)
        else:
            est, lo, hi = boot_ci(d, "ab_fwd60")
        if any(pd.isna(x) for x in (est, lo, hi)):
            out[tag] = {"n": len(d), "est": f"{est*100:+.2f}%", "ci": "—",
                        "verdict": "区块不足"}
            continue
        v = ("显著为正·有证据" if lo > 0 else "显著为负" if hi < 0 else "含零·无证据")
        out[tag] = {"n": len(d), "est": f"{est*100:+.2f}%",
                    "ci": f"[{lo*100:+.1f},{hi*100:+.1f}]", "verdict": v}
    return out


def main():
    ev = load()
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()
            & ev["pos_in_box"].notna()]
    dn = ev[ev["regime"] == "DOWN"].copy()

    # 完整纪律（v2 修正版的三个技术门槛）
    DIS = (dn["width30"] <= 0.15) & (dn["amt30"] >= 4e8) & (dn["turn30"] <= 6.0)
    g = dn[DIS]

    print("=" * 112)
    print("A. 下跌市 + 完整纪律：同一个箱宽约束下，价格在箱里的【位置】决定成败")
    print("=" * 112)
    print(f"  满足纪律的事件数: {len(g)}")
    print()
    pbins = [(0.0, 0.2, "箱底 0~20%"), (0.2, 0.4, "箱下 20~40%"), (0.4, 0.6, "箱中 40~60%"),
             (0.6, 0.8, "箱上 60~80%"), (0.8, 1.01, "箱顶 80~100%")]
    for lo_, hi_, ptag in pbins:
        d = g[(g["pos_in_box"] >= lo_) & (g["pos_in_box"] < hi_)]
        stat(f"  箱内位置 {ptag}", d)
    print()

    print("=" * 112)
    print("B. 交叉：距年高 × 箱内位置（下跌市 + 完整纪律）")
    print("=" * 112)
    xbins = [(0.95, 1.01, ">=0.95 贴年高"), (0.90, 0.95, "0.90~0.95"),
             (0.85, 0.90, "0.85~0.90"), (0.80, 0.85, "0.80~0.85")]
    for xlo, xhi, xtag in xbins:
        sub = g[(g["prox_now"] >= xlo) & (g["prox_now"] < xhi)]
        print(f"\n  【{xtag}】共 {len(sub)} 个事件")
        for lo_, hi_, ptag in pbins:
            d = sub[(sub["pos_in_box"] >= lo_) & (sub["pos_in_box"] < hi_)]
            stat(f"      + {ptag}", d)
    print()

    print("=" * 112)
    print("C. 关键检验：放宽档 0.85~0.95，加上「箱内位置>=50%」能不能救回来？")
    print("=" * 112)
    stat("  放宽档 0.85~0.95（不加位置约束）",
         g[(g["prox_now"] >= 0.85) & (g["prox_now"] < 0.95)])
    stat("  放宽档 0.85~0.95 + 箱位>=50%",
         g[(g["prox_now"] >= 0.85) & (g["prox_now"] < 0.95) & (g["pos_in_box"] >= 0.5)])
    stat("  放宽档 0.85~0.95 + 箱位>=70%",
         g[(g["prox_now"] >= 0.85) & (g["prox_now"] < 0.95) & (g["pos_in_box"] >= 0.7)])
    print()
    stat("  推荐档 >=0.95（不加位置约束）", g[g["prox_now"] >= 0.95])
    stat("  推荐档 >=0.95 + 箱位>=50%", g[(g["prox_now"] >= 0.95) & (g["pos_in_box"] >= 0.5)])
    print()

    print("=" * 112)
    print("D. 中信证券今日（2026-10-22 评估）落在哪一格，历史上这一格表现如何")
    print("=" * 112)
    print("  中信证券 600030：当前价/年高 0.854　箱宽 11.9%　箱内位置 9.1%　近60日 −10.7%")
    stat("  格子：0.85~0.90 × 箱底0~20%",
         g[(g["prox_now"] >= 0.85) & (g["prox_now"] < 0.90) & (g["pos_in_box"] < 0.2)])
    print()

    print("=" * 112)
    print("E. 对照：箱底横盘 vs 箱顶横盘，两组的收益分布差多少（下跌市+纪律）")
    print("=" * 112)
    for tag, m in [("箱底 0~20%", g["pos_in_box"] < 0.2), ("箱顶 80~100%", g["pos_in_box"] >= 0.8)]:
        d = g[m]
        if len(d) < 30:
            continue
        print(f"  {tag}  n={len(d)}")
        print(f"    绝对收益 均值 {d['fwd60'].mean()*100:+.2f}%  中位 {d['fwd60'].median()*100:+.2f}%  "
              f"胜率 {(d['fwd60']>0).mean()*100:.1f}%")
        print(f"    大跌(<−20%)占比 {(d['fwd60'] < -0.20).mean()*100:5.1f}%   "
              f"大涨(>+20%)占比 {(d['fwd60'] > 0.20).mean()*100:5.1f}%")
        print(f"    未来20日最大回撤 MAE20 均值 {d['mae20'].mean()*100:+.2f}%" if "mae20" in d.columns else "")
    print("=" * 112)
    print("F. 各 prox_now 档位的历史实证（下跌市 + 完整纪律）—— 供实盘阶梯表标注")
    print("=" * 112)
    for k, v in ladder_evidence("DOWN").items():
        print(f"  当前价/年高 >={k}  n={v['n']:>5}  β调整 {v['est']:>7}  90%CI {v['ci']:>14}  {v['verdict']}")
    print()
    print("※ 数据来源：腾讯前复权日线／通达信指数日线。以上均为历史统计，不构成投资建议。")


if __name__ == "__main__":
    main()
