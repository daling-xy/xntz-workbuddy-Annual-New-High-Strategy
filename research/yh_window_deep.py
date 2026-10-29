# -*- coding: utf-8 -*-
"""
样本区间深挖：近期有效性是否衰减？
==================================
承接 yh_sample_window.py 发现的矛盾：
    B 段（跨起点）→ 参数稳健，都指向 <=10%
    E 段（时间加权）→ 半衰期 250 天时最优档漂移到 25%
    F 段（fwd20 近12月）→ 约束越严超额越低，与全样本相反

本脚本用【同一标签 fwd60】做分期检验，排除换标签带来的干扰：
    H 分期增量趋势：每半年一期，看「15% 相对不约束」的增量是否递减
    I 早晚期对照  ：2024 前后划分，比较区分度
    J 样本量代价  ：缩短起点会让 CI 宽多少
    K 结论
"""
import os
import gc
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_final_calib2 import boot_ci

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
LINE = "=" * 78

NEED = ["code", "date", "prox_now", "dd_from_high", "width30",
        "amt30", "turn30", "fwd60", "ex_fwd60", "ab_fwd60",
        "mkt_fwd60", "beta", "regime"]

AMT_MIN = 4e8
TURN_LO, TURN_HI = 0.0, 6.0
PROX_THR = 0.95


def load_events() -> pd.DataFrame:
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    ev = ev[[c for c in NEED if c in ev.columns]].copy()
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi).astype("float32")
    ev["ab_fwd60"] = (ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]).astype("float32")
    gc.collect()
    return ev


def mask_of(df: pd.DataFrame, w_thr: float, prox: float = PROX_THR) -> pd.Series:
    m = (df["prox_now"] >= prox) & (df["amt30"] >= AMT_MIN) \
        & (df["turn30"] >= TURN_LO) & (df["turn30"] <= TURN_HI)
    if w_thr < 1.0:
        m &= (df["width30"] <= w_thr)
    return m


def wmean(d: pd.DataFrame, halflife: float = None, end=None) -> tuple:
    """返回 (加权/日等权均值, n, CI下界, CI上界)"""
    if len(d) < 30:
        return np.nan, len(d), np.nan, np.nan
    if halflife is None:
        try:
            est, lo, hi = boot_ci(d, "ab_fwd60", n=400)
            return float(est) * 100, len(d), float(lo) * 100, float(hi) * 100
        except Exception:
            est = d.groupby("date")["ab_fwd60"].mean().mean()
            return float(est) * 100, len(d), np.nan, np.nan
    # 指数加权（日等权基础上再按时间衰减）
    dv = d.groupby("date")["ab_fwd60"].mean().dropna()
    age = (end - pd.to_datetime(dv.index)).days.values.astype("float64")
    w = np.exp(-np.log(2) * age / halflife)
    return float((dv.values * w).sum() / w.sum()) * 100, len(d), np.nan, np.nan


# ----------------------------------------------------------------------
# H 段：分期增量趋势
# ----------------------------------------------------------------------
def seg_h(ev: pd.DataFrame):
    print(LINE)
    print("H. 分期增量趋势：箱宽约束的『增量』是不是越来越小？")
    print(LINE)
    ev = ev.copy()
    d = pd.to_datetime(ev["date"])
    ev["ym"] = d.dt.strftime("%Y-%m")
    ev["half"] = d.dt.year.astype(str) + "H" + np.where(d.dt.month <= 6, "1", "2")

    print("  统一前置：当前价/年高>=0.95 且 日均额>=4亿 且 换手0~6%")
    print()
    print(f"  {'半年期':<10}{'不约束n':>8}{'不约束':>9}{'15%_n':>7}{'15%':>9}{'增量':>9}{'10%_n':>7}{'10%':>9}{'增量':>9}")
    print("  " + "-" * 76)

    rows = []
    for h, sub in ev.groupby("half"):
        if len(sub) < 2000:
            continue
        e0, n0, l0, u0 = wmean(sub[mask_of(sub, 1.00)])
        e15, n15, l15, u15 = wmean(sub[mask_of(sub, 0.15)])
        e10, n10, l10, u10 = wmean(sub[mask_of(sub, 0.10)])
        g15 = e15 - e0 if e15 == e15 and e0 == e0 else np.nan
        g10 = e10 - e0 if e10 == e10 and e0 == e0 else np.nan
        rows.append({"半年期": h, "n0": n0, "e0": e0, "n15": n15, "e15": e15,
                     "g15": g15, "n10": n10, "e10": e10, "g10": g10})
        fmt = lambda x: f"{x:+.2f}" if x == x else "   n/a"
        print(f"  {h:<10}{n0:>8}{fmt(e0):>9}{n15:>7}{fmt(e15):>9}{fmt(g15):>9}"
              f"{n10:>7}{fmt(e10):>9}{fmt(g10):>9}")
    print()
    print("  增量 = 该箱宽档的 β调整超额 − 不约束箱宽时的 β调整超额")
    print("        （衡量『收紧箱宽』这个动作值不值得）")
    print()

    g = pd.DataFrame(rows)
    # 前后期对比
    g["期序"] = range(len(g))
    early = g[g["期序"] < len(g) // 2]
    late = g[g["期序"] >= len(g) // 2]
    print("  ★ 前半段 vs 后半段（箱宽约束的平均增量）：")
    print(f"     前半段（{early['半年期'].iloc[0]}~{early['半年期'].iloc[-1]}）  "
          f"15%增量 {early['g15'].mean():+.2f}%   10%增量 {early['g10'].mean():+.2f}%")
    print(f"     后半段（{late['半年期'].iloc[0]}~{late['半年期'].iloc[-1]}）  "
          f"15%增量 {late['g15'].mean():+.2f}%   10%增量 {late['g10'].mean():+.2f}%")
    d15 = late["g15"].mean() - early["g15"].mean()
    print()
    if abs(d15) < 0.5:
        print(f"  ✅ 15% 增量前后期差异 {d15:+.2f}pp，在噪音范围内 → 因子未明显衰减")
    elif d15 < 0:
        print(f"  ⚠ 15% 增量后半段下降 {abs(d15):.2f}pp → 因子有效性在减弱，需持续监控")
    else:
        print(f"  ✅ 15% 增量后半段反而提升 {d15:+.2f}pp → 因子依然有效")
    print()
    return g


# ----------------------------------------------------------------------
# I 段：早晚期对照
# ----------------------------------------------------------------------
def seg_i(ev: pd.DataFrame):
    print(LINE)
    print("I. 早晚期对照：2024 年前后，箱宽门槛还灵不灵")
    print(LINE)
    ev = ev.copy()
    d = pd.to_datetime(ev["date"])
    ev["y"] = d.dt.year

    periods = [("早期 2019-2021", 2019, 2021), ("中期 2022-2023", 2022, 2023),
               ("近期 2024-2026", 2024, 2026)]
    rows = []
    for tag, a, b in periods:
        sub = ev[(ev["y"] >= a) & (ev["y"] <= b)]
        if len(sub) < 1000:
            continue
        e0, n0, l0, u0 = wmean(sub[mask_of(sub, 1.00)])
        e15, n15, l15, u15 = wmean(sub[mask_of(sub, 0.15)])
        e25, n25, l25, u25 = wmean(sub[mask_of(sub, 0.25)])
        rows.append({
            "时期": tag, "不约束n": n0, "不约束": e0,
            "15%n": n15, "15%": e15, "15%CI下": l15, "15%判定": (
                "显著为正" if l15 == l15 and l15 > 0 else "含零·无证据"),
            "25%n": n25, "25%": e25,
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 240):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()
    return df


# ----------------------------------------------------------------------
# J 段：样本量代价
# ----------------------------------------------------------------------
def seg_j(ev: pd.DataFrame):
    print(LINE)
    print("J. 缩短起点的代价：样本少了，CI 会宽多少")
    print(LINE)
    ev = ev.copy()
    ev["y"] = pd.to_datetime(ev["date"]).dt.year

    rows = []
    for s in [2019, 2020, 2021, 2022, 2023, 2024]:
        sub = ev[ev["y"] >= s]
        if len(sub) < 500:
            continue
        m = mask_of(sub, 0.15)
        d = sub[m]
        if len(d) < 30:
            continue
        try:
            est, lo, hi = boot_ci(d, "ab_fwd60", n=600)
            est, lo, hi = float(est)*100, float(lo)*100, float(hi)*100
        except Exception:
            est = d.groupby("date")["ab_fwd60"].mean().mean()*100; lo = hi = np.nan
        rows.append({
            "起点": s, "可用事件": f"{len(sub):,}", "15%命中": len(d),
            "β调整超额": est, "CI下界": lo, "CI上界": hi, "CI宽度": hi-lo,
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 240):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()
    base = df.iloc[0]
    print(f"  ★ 以 {int(base['起点'])} 起为基准（CI 宽度 {base['CI宽度']:.2f}pp）：")
    for _, r in df.iloc[1:].iterrows():
        print(f"     {int(r['起点'])} 起  → 命中少 {int(base['15%命中']-r['15%命中']):>5} 条，"
              f"CI 宽度 {r['CI宽度']:>5.2f}pp（放大 {r['CI宽度']/base['CI宽度']:.2f} 倍）")
    print()
    print("  → 起点后移 = 用『更窄的置信区间』换『更新的样本』，这笔交易划不划算，")
    print("     取决于 J 段：参数在不同起点下是否发生漂移。")
    print()
    return df


def main():
    ev = load_events()
    print(f"载入事件 {len(ev):,}，交易日 {ev['date'].nunique():,}，股票 {ev['code'].nunique():,}")
    print()
    seg_h(ev)
    gc.collect()
    seg_i(ev)
    gc.collect()
    seg_j(ev)

    print(LINE)
    print("※ 数据来源：腾讯前复权日线与换手率（通达信数据源）、通达信指数日线、")
    print("※ 东方财富财报。以上均为历史统计与回测结论，不构成投资建议。")
    print(LINE)


if __name__ == "__main__":
    main()
