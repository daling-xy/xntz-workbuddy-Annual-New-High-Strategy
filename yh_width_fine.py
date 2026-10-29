# -*- coding: utf-8 -*-
"""
【箱宽 15% 的稳健性解剖】——回答"15 是怎么得出来的、它到底稳不稳"

背景（见 output/final_calib2.txt G 段）：
  15% 最初是从一个【只有 15% 和 20% 两个点】的网格里选出来的。
  这么粗的网格，凭什么相信中间没有更好的点？更细的点是平台还是悬崖？

本脚本做 5 段，用同一种口径（v2 现场：prox_now>=0.95 + 额>=4亿 + 换手<=6%）：
  A 细扫【累积型】阈值 8%~26%，步长 1%：看 15 附近是不是平滑
  B 细扫【切片型】相邻箱体 1% 宽：定位真正的能力断崖在哪一格
  C 分周期（UP/DOWN）各自重扫：DOWN 才是当前所处环境，重点看
  D 前后半样本对照：15 这个点在两段时间里是否都成立
  E 配对 bootstrap：15% 与相邻档（12/18/20）的差异是否统计显著

数据来源：腾讯前复权日线（2018 至今全历史）、通达信指数日线、东方财富财报。
所有结论均为历史统计，不构成投资建议。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 2000
_rng = np.random.default_rng(20261027)

LINE = "=" * 120
SUB = "-" * 120

NEED = ["code", "date", "regime", "prox_now", "prox", "width30",
        "amt30", "turn30", "beta", "fwd60", "mkt_fwd60", "ex_fwd60",
        "ab_fwd60", "mae20"]


def load_events() -> pd.DataFrame:
    p = os.path.join(CACHE, "events_fixed.pkl")
    if not os.path.exists(p):
        p = os.path.join(CACHE, "events_long_beta.pkl")
    ev = pd.read_pickle(p)
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev = ev[[c for c in NEED if c in ev.columns]]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    return ev.reset_index(drop=True)


def boot_ci(sub: pd.DataFrame, key: str = "ab_fwd60", n: int = BOOT):
    """按交易日整簇重抽样"""
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = float(v.mean())
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def stat_row(tag: str, d: pd.DataFrame, key: str = "ab_fwd60", w: int = 26):
    n = len(d)
    if n < 30:
        print(f"  {tag:<{w}} n={n:>6}  样本不足")
        return None
    est, lo, hi = boot_ci(d, key)
    win = float((d[key] > 0).mean())
    big = float((d["fwd60"] < -0.20).mean())
    mae = float(d["mae20"].mean()) if "mae20" in d.columns else np.nan
    sig = "显著为正" if (lo > 0) else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<{w}} n={n:>6}  β中位{d['beta'].median():>5.2f}  "
          f"{est*100:+6.2f}%  CI[{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率{win*100:4.1f}%  大亏{big*100:4.1f}%  MAE{mae*100:+6.2f}%  {sig}")
    return est


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def base(ev: pd.DataFrame) -> pd.DataFrame:
    """v2 现场的其余纪律（不含箱宽）：prox_now>=0.95 + 额>=4亿 + 换手<=6%"""
    d = ev[ev["prox_now"] >= 0.95]
    d = d[(d["amt30"] >= 4e8) & (d["turn30"] <= 6.0)]
    return d


def main() -> None:
    ev = load_events()
    print(f"事件表: {len(ev):,} 行 / {ev['code'].nunique()} 股 / "
          f"{ev['date'].nunique()} 交易日")
    print(f"日期区间: {ev['date'].min()} ~ {ev['date'].max()}")

    b = base(ev)
    print(f"\nv2 基准池（prox_now>=0.95 + 额>=4亿 + 换手<=6%，不加箱宽）: "
          f"{len(b):,} 事件")

    # ---------------- A. 累积型细扫 ----------------
    section("A. 累积型细扫：箱宽 <= X%  （步长 1%）")
    print("  口径：v2 现场（prox_now>=0.95 + 额>=4亿 + 换手<=6%）  超额：ab_fwd60（beta调整）")
    print()
    grid_a = {}
    for x in range(8, 27):
        d = b[b["width30"] <= x / 100.0]
        e = stat_row(f"箱宽<={x}%", d)
        grid_a[x] = e
    print()
    vals = {k: v for k, v in grid_a.items() if v is not None}
    ranked = sorted(vals.items(), key=lambda kv: -kv[1])
    print("  → 超额最高的 5 个阈值:",
          ", ".join([f"{k}%({v*100:+.2f}%)" for k, v in ranked[:5]]))
    print("  → 15% 在所有细扫阈值里的排名:",
          [k for k, _ in ranked].index(15) + 1, "/", len(ranked))

    # ---------------- B. 切片型细扫 ----------------
    section("B. 切片型细扫：相邻 1% 箱体（定位能力断崖）")
    print("  每个格子 = 单独一批事件（如 14~15%），互不重叠")
    print()
    edges = [(8, 10), (10, 12), (12, 13), (13, 14), (14, 15), (15, 16),
             (16, 17), (17, 18), (18, 19), (19, 20), (20, 22), (22, 25), (25, 40)]
    for lo, hi in edges:
        d = b[(b["width30"] > lo / 100.0) & (b["width30"] <= hi / 100.0)]
        stat_row(f"箱宽 {lo}~{hi}%", d)

    # ---------------- C. 分周期重扫 ----------------
    for rg in ["UP", "DOWN"]:
        section(f"C. 分周期重扫 —— [{rg}] 周期（当前 v2 实盘环境之一）")
        br = base(ev[ev["regime"] == rg])
        print(f"  {rg} 基准池: {len(br):,} 事件")
        print()
        for x in [10, 12, 13, 14, 15, 16, 17, 18, 20, 25]:
            stat_row(f"箱宽<={x}%", br[br["width30"] <= x / 100.0])

    # ---------------- D. 前后半样本 ----------------
    section("D. 前后半样本对照：15% 在两段时间里是否都成立")
    ds = sorted(ev["date"].unique())
    mid = ds[len(ds) // 2]
    print(f"  分界日: {mid}   前半 ~{ds[len(ds)//2 - 1]}   后半 {mid}~")
    print()
    for tag, part in [("前半", ev[ev["date"] < mid]), ("后半", ev[ev["date"] >= mid])]:
        bp = base(part)
        print(f"  【{tag}】基准池 {len(bp):,} 事件")
        for x in [10, 12, 14, 15, 16, 18, 20, 25]:
            stat_row(f"    箱宽<={x}%", bp[bp["width30"] <= x / 100.0])
        print()

    # ---------------- E. 配对 bootstrap ----------------
    section("E. 配对 bootstrap：15% 相对相邻档的差异是否显著")
    print("  方法：同一批交易日上，两档的日均超额之差，按交易日整簇重抽样 2000 次")
    print("  口径：累积型 <=X%，只保留【两档都非空】的交易日")
    print()

    def daily_series(d: pd.DataFrame) -> pd.Series:
        return d.groupby("date")["ab_fwd60"].mean().dropna()

    s15 = daily_series(b[b["width30"] <= 0.15])
    for rival in [12, 14, 16, 18, 20, 25]:
        sr = daily_series(b[b["width30"] <= rival / 100.0])
        common = s15.index.intersection(sr.index)
        if len(common) < 20:
            print(f"  15% vs {rival}%: 共同交易日不足")
            continue
        diff = (s15.loc[common] - sr.loc[common]).values
        est = float(diff.mean())
        idx = _rng.integers(0, len(diff), size=(BOOT, len(diff)))
        bs = diff[idx].mean(axis=1)
        lo, hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
        verdict = ("15% 显著更优" if lo > 0 else
                   ("15% 显著更差" if hi < 0 else "无显著差异（打平）"))
        print(f"  15% vs <= {rival:>2}%:  差 {est*100:+6.2f}pp  "
              f"CI[{lo*100:+5.2f},{hi*100:+5.2f}]  90%CI {'显著' if lo*hi>0 else '含零'}  "
              f"→ {verdict}  (共同交易日 {len(common)})")

    print()
    print(LINE)
    print("结论提示：若 15% 与 14%/16%/18% 均无显著差异，说明 15 不是精确最优点，")
    print("          而是一个【稳健平台】内的代表值——这种参数可以放心用；")
    print("          若与某档差异显著，则该档才是真正的能力边界，需要重新定标。")
    print(LINE)


if __name__ == "__main__":
    main()
