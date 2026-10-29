# -*- coding: utf-8 -*-
"""
【为什么今天没捕捉到乐鑫科技 688018？】

2026-10-23 实盘，688018 只有两条不达标，但都是硬门槛：
    箱宽 42.2%  >>  15%
    营收同比 18.9%  <  30%
它属于「已经在突破途中」（近20日 +34.9%、箱内位置 90%、贴年高 0.969），
而本策略赚的是「高位窄幅蓄势后的再突破」。两者是两种东西。

本脚本检验：贴着年高但箱宽大的票，历史上到底好不好？我们是不是漏了好东西？
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


def stat(tag, d, note=""):
    if len(d) < 30:
        print(f"  {tag:<30} n={len(d):>6}  样本不足")
        return
    est, lo, hi = boot_ci(d, "ab_fwd60")
    win = float((d["ab_fwd60"] > 0).mean())
    mae = d["mae20"].mean() if "mae20" in d.columns else np.nan
    big_loss = float((d["fwd60"] < -0.20).mean())
    sig = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<30} n={len(d):>6}  β中位 {d['beta'].median():>5.2f}  "
          f"β调整 {est*100:+6.2f}%  CI [{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率 {win*100:4.1f}%  大跌率 {big_loss*100:4.1f}%  "
          f"MAE20 {mae*100:+5.2f}%  {sig}")


def main():
    print("=" * 116)
    print("A. 乐鑫科技 688018 逐项对照（评估日 2026-10-22）")
    print("=" * 116)
    from yh_panel import build_panel
    p = build_panel(asof_offset=1, days=320)
    r = p.loc["688018"]
    pn = 1.0 + float(r["dd_from_high"])
    checks = [
        ("当前价/年高 >= 0.95", pn >= 0.95, f"{pn:.3f}"),
        ("30日箱宽 <= 15%", r["width30"] <= 0.15, f"{r['width30']:.1%}"),
        ("30日均额 >= 4亿", r["amt30"] >= 4e8, f"{r['amt30']/1e8:.1f}亿"),
        ("换手 <= 6%", r["turn30"] <= 6.0, f"{r['turn30']:.2f}%"),
        ("营收同比 >= 30%", r["rev_yoy"] >= 30.0, f"{r['rev_yoy']:.1f}%"),
        ("上期营收同比 > 0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"),
        ("单期营收 >= 2亿", r["revenue"] >= 2e8, f"{r['revenue']/1e8:.1f}亿"),
        ("未亏损", r["is_loss"] == 0, "否"),
        ("行业未剔除", r["board"] not in ["银行", "保险"], str(r["board"])),
    ]
    for name, ok, val in checks:
        print(f"  {'✓' if ok else '✗'} {name:<22} 实测 {val}")
    print(f"\n  其它：箱内位置 {r['pos_in_box']:.0%}（贴箱顶） 近20日 {r['ret20']:+.1%} "
          f"近60日 {r['ret60']:+.1%} 年高 {r['hi250']:.2f} 现价 {r['close']:.2f} "
          f"PE {r['pe_ttm']:.0f} ROE {r['roe_w']:.1f}%")
    print(f"  → 卡在【箱宽 {r['width30']:.1%} ≫ 15%】和【营收同比 {r['rev_yoy']:.1f}% < 30%】两条。")
    print()

    ev = load()
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    dn = ev[ev["regime"] == "DOWN"]

    print("=" * 116)
    print("B. 【核心】下跌市 · 贴年高(>=0.95) · 额>=4亿 · 换手<=6% —— 只放开箱宽，看会不会更好")
    print("=" * 116)
    base = dn[(dn["prox_now"] >= 0.95) & (dn["amt30"] >= 4e8) & (dn["turn30"] <= 6.0)]
    print(f"  贴年高 + 额 + 换手 的事件数: {len(base)}（箱宽不设限）\n")
    for lo_, hi_, tag in [(0.0, 0.15, "箱宽 <=15%（策略采用）"),
                          (0.15, 0.25, "箱宽 15~25%"),
                          (0.25, 0.40, "箱宽 25~40%"),
                          (0.40, 1.01, "箱宽 >40%（乐鑫所在）")]:
        stat(f"  {tag}", base[(base["width30"] >= lo_) & (base["width30"] < hi_)])
    print()

    print("=" * 116)
    print("C. 全周期（不分涨跌）复核：贴年高 + 额 + 换手，按箱宽分组")
    print("=" * 116)
    base2 = ev[(ev["prox_now"] >= 0.95) & (ev["amt30"] >= 4e8) & (ev["turn30"] <= 6.0)]
    print(f"  事件数: {len(base2)}\n")
    for lo_, hi_, tag in [(0.0, 0.15, "箱宽 <=15%"), (0.15, 0.25, "箱宽 15~25%"),
                          (0.25, 0.40, "箱宽 25~40%"), (0.40, 1.01, "箱宽 >40%")]:
        stat(f"  {tag}", base2[(base2["width30"] >= lo_) & (base2["width30"] < hi_)])
    print()

    print("=" * 116)
    print("D. 那如果专门为「大箱宽 + 贴年高」开一档，值不值？（下跌市）")
    print("=" * 116)
    stat("  箱宽>40% + 贴年高 + 额 + 换手",
         base[base["width30"] > 0.40])
    stat("  箱宽>40% + 贴年高 + 额 + 换手 + 箱位>=80%",
         base[(base["width30"] > 0.40) & (base["pos_in_box"] >= 0.8)])
    stat("  【对照】箱宽<=15% + 贴年高 + 额 + 换手",
         base[base["width30"] <= 0.15])
    print()
    print("※ 数据来源：腾讯前复权日线／通达信指数日线／东方财富财报。历史统计，不构成投资建议。")


if __name__ == "__main__":
    main()
