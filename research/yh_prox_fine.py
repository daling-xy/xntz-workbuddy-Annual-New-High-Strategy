# -*- coding: utf-8 -*-
"""
【prox_now 门槛解剖】—— 回答「距年高 5% 以内（>=0.95）是不是太苛刻」

触发问题：乐鑫科技 688018 贴着年高（prox_now = 1.000）却没被选中，
          是不是 0.95 这条线太苛刻？要不要放宽？

本脚本做 5 段：
  A 细扫 prox_now 门槛 0.88 ~ 1.00（固定箱宽<=15% + 额>=4亿 + 换手<=6%）
  B 分周期（UP/DOWN/CHOP）各自重扫 —— DOWN 是当前实盘环境
  C 前后半样本对照：0.95 在两段时间是否都成立
  D 配对 bootstrap：0.95 相对相邻档（0.90/0.92/0.98/1.00）是否显著更优
  E 【关键】乐鑫这类票的真实形态：贴年高 + 大箱宽，历史上到底赚不赚钱

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
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = float(v.mean())
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def stat_row(tag: str, d: pd.DataFrame, w: int = 22):
    n = len(d)
    if n < 30:
        print(f"  {tag:<{w}} n={n:>6}  样本不足")
        return None
    est, lo, hi = boot_ci(d)
    win = float((d["ab_fwd60"] > 0).mean())
    big = float((d["fwd60"] < -0.20).mean())
    mae = float(d["mae20"].mean()) if "mae20" in d.columns else np.nan
    per = float(d.groupby("date").size().mean())
    sig = "显著为正" if (lo > 0) else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<{w}} n={n:>6}  日均{per:>5.2f}只  β中位{d['beta'].median():>5.2f}  "
          f"{est*100:+6.2f}%  CI[{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率{win*100:4.1f}%  大亏{big*100:4.1f}%  MAE{mae*100:+6.2f}%  {sig}")
    return est


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def with_box(d: pd.DataFrame) -> pd.DataFrame:
    """v2 的其余纪律：箱宽<=15% + 额>=4亿 + 换手 0~6%"""
    return d[(d["width30"] <= 0.15) & (d["amt30"] >= 4e8) & (d["turn30"] <= 6.0)]


def main() -> None:
    ev = load_events()
    print(f"事件表: {len(ev):,} 行 / {ev['code'].nunique()} 股 / "
          f"{ev['date'].nunique()} 交易日")
    print(f"日期区间: {ev['date'].min()} ~ {ev['date'].max()}")

    w = with_box(ev)
    print(f"\n基准池（箱宽<=15% + 额>=4亿 + 换手<=6%，不设 prox 门槛）: {len(w):,} 事件")

    grid = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0001]

    # ---------------- A ----------------
    section("A. prox_now 门槛细扫（全市场，固定箱宽<=15%+额>=4亿+换手<=6%）")
    print("  超额：ab_fwd60（beta 调整，60日前瞻）")
    print()
    res = {}
    for x in grid:
        d = w[w["prox_now"] >= x]
        lbl = ("严格=年高" if x > 1.0 else f">={x:.2f}")
        res[x] = stat_row(f"prox_now {lbl}", d)
    print()
    vals = {k: v for k, v in res.items() if v is not None}
    ranked = sorted(vals.items(), key=lambda kv: -kv[1])
    print("  → 超额最高的 4 档:",
          ", ".join([f"{k:.2f}({v*100:+.2f}%)" for k, v in ranked[:4]]))
    print(f"  → 0.95 的排名: {[k for k, _ in ranked].index(0.95) + 1} / {len(ranked)}")

    # ---------------- B ----------------
    for rg in ["DOWN", "UP", "CHOP"]:
        section(f"B. 分周期重扫 —— [{rg}]")
        wr = with_box(ev[ev["regime"] == rg])
        print(f"  {rg} 基准池: {len(wr):,} 事件")
        print()
        for x in [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 1.0001]:
            d = wr[wr["prox_now"] >= x]
            lbl = ("严格=年高" if x > 1.0 else f">={x:.2f}")
            stat_row(f"prox_now {lbl}", d)

    # ---------------- C ----------------
    section("C. 前后半样本对照（分界 2022-12-27）")
    ds = sorted(ev["date"].unique())
    mid = ds[len(ds) // 2]
    print(f"  分界日: {mid}")
    print()
    for tag, part in [("前半", ev[ev["date"] < mid]), ("后半", ev[ev["date"] >= mid])]:
        wp = with_box(part)
        print(f"  【{tag}】基准池 {len(wp):,} 事件")
        for x in [0.90, 0.92, 0.95, 0.98, 1.0001]:
            lbl = ("严格=年高" if x > 1.0 else f">={x:.2f}")
            stat_row(f"    {lbl}", wp[wp["prox_now"] >= x])
        print()

    # ---------------- D ----------------
    section("D. 配对 bootstrap：0.95 相对相邻档")
    print("  同一批交易日上两档日均超额之差，整簇重抽样 2000 次")
    print()

    def daily(d: pd.DataFrame) -> pd.Series:
        return d.groupby("date")["ab_fwd60"].mean().dropna()

    s95 = daily(w[w["prox_now"] >= 0.95])
    for rival, lbl in [(0.90, "0.90"), (0.92, "0.92"), (0.94, "0.94"),
                       (0.96, "0.96"), (0.98, "0.98"), (1.0001, "严格=年高")]:
        sr = daily(w[w["prox_now"] >= rival])
        common = s95.index.intersection(sr.index)
        if len(common) < 20:
            continue
        diff = (s95.loc[common] - sr.loc[common]).values
        est = float(diff.mean())
        idx = _rng.integers(0, len(diff), size=(BOOT, len(diff)))
        bs = diff[idx].mean(axis=1)
        lo, hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
        verdict = ("0.95 显著更优" if lo > 0 else
                   ("0.95 显著更差" if hi < 0 else "无显著差异（打平）"))
        print(f"  0.95 vs >= {lbl:<8}: 差 {est*100:+6.2f}pp  "
              f"CI[{lo*100:+5.2f},{hi*100:+5.2f}]  → {verdict}  ({len(common)} 日)")

    # ---------------- E ----------------
    section("E. 【乐鑫形态】贴年高 + 大箱宽，历史上赚不赚钱？")
    print("  乐鑫 688018 当前形态：prox_now = 1.000（正好站年高）、箱宽 50.45%、")
    print("  近20日 +48.2%、近60日 +48.5% —— 属「一波流冲高」，非「高位窄幅整理」")
    print()
    d_all = ev[(ev["amt30"] >= 4e8) & (ev["turn30"] <= 6.0)]
    print("  【全周期】贴年高 + 额 + 换手，按箱宽分组")
    base = d_all[d_all["prox_now"] >= 0.95]
    for lo_, hi_, lbl in [(0.0, 0.15, "箱宽<=15%（策略采用）"),
                          (0.15, 0.25, "箱宽 15~25%"),
                          (0.25, 0.40, "箱宽 25~40%"),
                          (0.40, 9.99, "箱宽>40%（乐鑫所在）")]:
        d = base[(base["width30"] > lo_) & (base["width30"] <= hi_)]
        stat_row(lbl, d, w=24)
    print()
    print("  【DOWN 期】同上")
    bdown = d_all[(d_all["prox_now"] >= 0.95) & (d_all["regime"] == "DOWN")]
    for lo_, hi_, lbl in [(0.0, 0.15, "箱宽<=15%（策略采用）"),
                          (0.15, 0.25, "箱宽 15~25%"),
                          (0.25, 0.40, "箱宽 25~40%"),
                          (0.40, 9.99, "箱宽>40%（乐鑫所在）")]:
        d = bdown[(bdown["width30"] > lo_) & (bdown["width30"] <= hi_)]
        stat_row(lbl, d, w=24)

    print()
    print(LINE)
    print("若放宽 prox 到 0.90 便能选中乐鑫 → 值得改；")
    print("若乐鑫被卡的主因是箱宽/营收，放宽 prox 也选中不了 → 改了白改。")
    print(LINE)


if __name__ == "__main__":
    main()
