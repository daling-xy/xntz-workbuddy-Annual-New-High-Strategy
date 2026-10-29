# -*- coding: utf-8 -*-
"""
【到底要不要分周期？】—— 统一参数 vs 分周期各自最优 的实证对比

用户问题：
  "先不要区分上升/震荡/下跌期，如果这样的话现在的选股会有什么明显的差距吗？"

本脚本回答三件事：
  A 段  确认 v2 当前三个周期的参数是否真的完全一致（若一致，则分不分周期
        对【入选名单】没有影响，只影响【建议仓位】）
  B 段  给每个周期单独做网格搜索，看看"各自最优"能比"统一参数"好多少
  C 段  结论：分周期的增量到底值不值得

评价指标：优先看 bootstrap 90%CI 的【下界】（更稳健，避免小样本挑到噪声），
          同时报 n、均值、胜率、大亏率、MAE20。

数据来源：腾讯前复权日线（全历史）、通达信指数日线、东方财富财报。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1000
_rng = np.random.default_rng(20261026)

LINE = "=" * 118
NEED = ["code", "date", "regime", "prox_now", "prox", "width30", "amt30",
        "turn30", "beta", "fwd60", "mkt_fwd60", "ex_fwd60", "ab_fwd60", "mae20"]

# 统一参数（当前 v2 修正版采用）
UNIFIED = dict(prox_now_min=0.95, width30_max=0.15, amt30_min=4e8,
               turn30_min=0.0, turn30_max=6.0)

PROX_GRID = [0.95, 0.92, 0.90, 0.88, 0.85]
WIDTH_GRID = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]


def boot_ci(sub, key, n=BOOT):
    """按交易日整簇重抽样：先按 date 聚成日均值，再对日均值序列有放回重抽"""
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def measure(d, tag="", w=28):
    """返回一行统计；样本不足返回 None"""
    if len(d) < 30:
        return None
    est, lo, hi = boot_ci(d, "ab_fwd60")
    win = float((d["ab_fwd60"] > 0).mean())
    big = float((d["fwd60"] < -0.20).mean())
    mae = float(d["mae20"].mean()) if "mae20" in d.columns else np.nan
    return dict(tag=tag, n=len(d), beta=d["beta"].median(), est=est,
                lo=lo, hi=hi, win=win, big=big, mae=mae)


def show(r, w=28):
    if r is None:
        return
    sig = ("显著为正" if r["lo"] > 0 else
           "显著为负" if r["hi"] < 0 else "含零·无效")
    print(f"  {r['tag']:<{w}} n={r['n']:>6}  β中位 {r['beta']:>5.2f}  "
          f"β调整 {r['est']*100:+6.2f}%  90%CI [{r['lo']*100:+5.2f},{r['hi']*100:+5.2f}]  "
          f"胜率 {r['win']*100:4.1f}%  大亏 {r['big']*100:4.1f}%  MAE20 {r['mae']*100:+6.2f}%  {sig}")


def sub_by(d, prox=None, width=None, amt=None, turn=None):
    m = pd.Series(True, index=d.index)
    if prox is not None:
        m &= (d["prox_now"] >= prox)
    if width is not None:
        m &= (d["width30"] <= width)
    if amt is not None:
        m &= (d["amt30"] >= amt)
    if turn is not None:
        m &= (d["turn30"] <= turn)
    return d[m]


def main():
    from yh_screen import PARAMS_BY_REGIME, REGIME_SIZE, REGIME_EVIDENCE
    p = os.path.join(CACHE, "events_fixed.pkl")
    ev = pd.read_pickle(p if os.path.exists(p) else os.path.join(CACHE, "events_long_beta.pkl"))
    lo_, hi_ = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo_, hi_)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    keep = [c for c in NEED if c in ev.columns]
    ev = ev[keep]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]

    # ============ A 段：确认三周期参数是否一致 ============
    print(LINE)
    print("A. 先确认：v2 修正版三个周期的选股门槛，是不是本来就已经一样了？")
    print(LINE)
    print()
    keys = sorted(set().union(*[set(v.keys()) for v in PARAMS_BY_REGIME.values()]))
    print(f"  {'参数':<16}" + "".join(f"{r:>14}" for r in ["UP", "CHOP", "DOWN"]))
    same = True
    for k in keys:
        vals = [PARAMS_BY_REGIME[r].get(k) for r in ["UP", "CHOP", "DOWN"]]
        def fmt(v):
            if v is None:
                return "—"
            return f"{v/1e8:.1f}亿" if k == "amt30_min" else f"{v:.2f}"
        print(f"  {k:<16}" + "".join(f"{fmt(v):>14}" for v in vals))
        if len(set(map(str, vals))) != 1:
            same = False
    print()
    if same:
        print("  ✔ 三个周期的门槛【完全一致】—— 不存在「按周期切换门槛」这回事。")
        print()
        print("  ⚠ 那为什么还要算 regime？因为它的用途早就【不是换门槛】，而是：")
        for r in ["UP", "CHOP", "DOWN"]:
            print(f"      {r:<5} → {REGIME_SIZE.get(r, '—')}")
        print()
        print("  也就是说：周期标签只决定【下注多少】，不决定【下注什么】。")
    else:
        print("  ✘ 三个周期门槛不一致，需要逐周期讨论。")
    print()

    # ============ B 段：各周期单独网格搜索 ============
    print(LINE)
    print("B. 如果给每个周期单独找最优门槛，能比统一门槛好多少？")
    print(LINE)
    print("  固定：日均额>=4亿、换手<=6%")
    print("  搜索：距年高门槛 × 箱宽门槛")
    print()

    best = {}
    for reg in ["ALL", "UP", "CHOP", "DOWN"]:
        d0 = ev if reg == "ALL" else ev[ev["regime"] == reg]
        print(f"  --- {reg} 期（全量 n={len(d0):,}）---")
        print(f"  {'距年高':>8}{'箱宽':>8}{'n':>8}{'β中位':>8}"
              f"{'β调整':>9}{'CI下界':>9}{'胜率':>8}{'大亏':>8}{'MAE20':>9}")
        rows = []
        for pw in PROX_GRID:
            for wd in WIDTH_GRID:
                r = measure(sub_by(d0, pw, wd, 4e8, 6.0),
                            tag=f"{pw:.2f}/{wd:.0%}")
                if r:
                    rows.append((pw, wd, r))
        # 按 CI 下界排序（稳健优先），再按均值
        rows.sort(key=lambda x: (x[2]["lo"], x[2]["est"]), reverse=True)
        for pw, wd, r in rows:
            print(f"  {pw:>8.2f}{wd:>8.0%}{r['n']:>8}{r['beta']:>8.2f}"
                  f"{r['est']*100:>8.2f}%{r['lo']*100:>8.2f}%"
                  f"{r['win']*100:>7.1f}%{r['big']*100:>7.1f}%{r['mae']*100:>8.2f}%")
        if rows:
            best[reg] = rows[0]
            print(f"  ★ {reg} 期最优（按 CI 下界）：距年高>={rows[0][0]:.2f}、"
                  f"箱宽<={rows[0][1]:.0%} → β调整 {rows[0][2]['est']*100:+.2f}%，"
                  f"CI下界 {rows[0][2]['lo']*100:+.2f}%")
        print()

    # ============ C 段：统一门槛 vs 分周期最优，差多少 ============
    print(LINE)
    print("C. 最终对决：统一门槛 vs 分周期各自最优")
    print(LINE)
    print()
    print(f"  {'周期':<8}{'统一 0.95/15%':>26}{'该周期最优':>26}{'差值(CI下界)':>16}")
    print("  " + "-" * 76)
    for reg in ["ALL", "UP", "CHOP", "DOWN"]:
        d0 = ev if reg == "ALL" else ev[ev["regime"] == reg]
        ru = measure(sub_by(d0, UNIFIED["prox_now_min"], UNIFIED["width30_max"],
                            UNIFIED["amt30_min"], UNIFIED["turn30_max"]))
        rb = best.get(reg)
        if ru is None or rb is None:
            continue
        rb = rb[2]
        u = f"n={ru['n']:>5} {ru['est']*100:+6.2f}% [{ru['lo']*100:+.2f}"
        b = f"n={rb['n']:>5} {rb['est']*100:+6.2f}% [{rb['lo']*100:+.2f}"
        d = rb["lo"] - ru["lo"]
        print(f"  {reg:<8}{u:>26}{b:>26}{d*100:>15.2f}pp")
    print()
    print("  注：差值按 90%CI 下界算，'最优' 是样本内搜索结果，未做样本外验证。")
    print()

    print("※ 数据来源：腾讯前复权日线／通达信指数日线／东方财富财报。")
    print("※ 以上均为历史统计与回测结论，不构成投资建议。")


if __name__ == "__main__":
    main()
