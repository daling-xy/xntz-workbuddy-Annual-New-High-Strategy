# -*- coding: utf-8 -*-
"""
配对检验：不同箱宽档的差值到底是不是噪音？
==========================================
yh_sample_window.py 的 E 段出现「半衰期250天时最优档漂移到25%」，
但各档数值（+3.11 / +2.54 / +1.79 / +3.12 / +1.57）挨得很近，
很可能是噪音。本脚本用【配对 bootstrap】直接检验差值：

    同一批重抽样的交易日下，一次算出 A 档和 B 档的均值，
    再取差 → 得到差值的抽样分布 → 看差值 CI 是否含零

这样能消除「两档共用同一批交易日」带来的相关性，比分别看 CI 严格得多。
"""
import os
import gc
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
LINE = "=" * 78

NEED = ["code", "date", "prox_now", "dd_from_high", "width30",
        "amt30", "turn30", "fwd60", "ex_fwd60", "ab_fwd60",
        "mkt_fwd60", "beta", "regime"]

AMT_MIN = 4e8
TURN_LO, TURN_HI = 0.0, 6.0
PROX_THR = 0.95
B = 2000


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


def base_mask(df: pd.DataFrame) -> pd.Series:
    return (df["prox_now"] >= PROX_THR) & (df["amt30"] >= AMT_MIN) \
        & (df["turn30"] >= TURN_LO) & (df["turn30"] <= TURN_HI)


def width_mask(df: pd.DataFrame, w: float) -> pd.Series:
    m = base_mask(df)
    if w < 1.0:
        m &= (df["width30"] <= w)
    return m


def daily_series(d: pd.DataFrame) -> pd.Series:
    """日等权的日度均值序列"""
    return d.groupby("date")["ab_fwd60"].mean().dropna()


def pair_diff(ev: pd.DataFrame, wA: float, wB: float, halflife: float = None):
    """检验 wA 档 与 wB 档 的差值是否显著不为零（配对整簇 bootstrap）"""
    dA = ev[width_mask(ev, wA)]
    dB = ev[width_mask(ev, wB)]
    sA, sB = daily_series(dA), daily_series(dB)
    # 对齐到共同交易日
    common = sA.index.intersection(sB.index)
    if len(common) < 20:
        return None
    a = sA.loc[common].values.astype("float64")
    b = sB.loc[common].values.astype("float64")

    end = pd.to_datetime(common).max()
    if halflife is None:
        w = np.ones(len(common))
    else:
        age = (end - pd.to_datetime(common)).days.values.astype("float64")
        w = np.exp(-np.log(2) * age / halflife)

    est = ((a - b) * w).sum() / w.sum()
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(common), size=(B, len(common)))
    wa = w[idx]
    boot = ((a[idx] - b[idx]) * wa).sum(axis=1) / wa.sum(axis=1)
    lo, hi = np.percentile(boot, [5, 95])
    return {"n_days": len(common), "nA": len(dA), "nB": len(dB),
            "diff": float(est)*100, "lo": float(lo)*100, "hi": float(hi)*100,
            "p_gt0": float((boot > 0).mean())}


def main():
    ev = load_events()
    print(f"载入事件 {len(ev):,}，交易日 {ev['date'].nunique():,}")
    print()
    print(LINE)
    print("配对 bootstrap：箱宽档之间的差值，到底含不含零？")
    print(LINE)
    print("  配对方式：同一批重抽样的交易日下同时算两档均值，再取差")
    print("  p_gt0 = 1000+ 次重抽样中『A档好过B档』的比例，>0.95 才算真赢")
    print()

    # 全样本（等权）
    print("  【全样本 · 等权】")
    print(f"  {'对比':<26}{'共同交易日':>10}{'差值(A−B)':>12}{'CI下界':>10}{'CI上界':>10}{'p(A>B)':>9}  判定")
    print("  " + "-" * 92)
    for wA, wB, tag in [(0.10, 0.15, "10% vs 15%"),
                        (0.15, 0.20, "15% vs 20%"),
                        (0.15, 0.25, "15% vs 25%"),
                        (0.15, 1.00, "15% vs 不约束"),
                        (0.20, 0.25, "20% vs 25%"),
                        (0.10, 1.00, "10% vs 不约束")]:
        r = pair_diff(ev, wA, wB)
        if r is None:
            print(f"  {tag:<26}{'样本不足':>10}")
            continue
        judge = "显著为正" if r["lo"] > 0 else ("显著为负" if r["hi"] < 0 else "含零·无证据")
        print(f"  {tag:<26}{r['n_days']:>10}{r['diff']:>+12.2f}{r['lo']:>+10.2f}"
              f"{r['hi']:>+10.2f}{r['p_gt0']:>9.2f}  {judge}")
    print()

    # 近期加权（半衰期 250 天）
    print("  【近期加权 · 半衰期250天】")
    print(f"  {'对比':<26}{'共同交易日':>10}{'差值(A−B)':>12}{'CI下界':>10}{'CI上界':>10}{'p(A>B)':>9}  判定")
    print("  " + "-" * 92)
    for wA, wB, tag in [(0.10, 0.15, "10% vs 15%"),
                        (0.15, 0.20, "15% vs 20%"),
                        (0.15, 0.25, "15% vs 25%"),
                        (0.15, 1.00, "15% vs 不约束"),
                        (0.20, 0.25, "20% vs 25%")]:
        r = pair_diff(ev, wA, wB, halflife=250)
        if r is None:
            print(f"  {tag:<26}{'样本不足':>10}")
            continue
        judge = "显著为正" if r["lo"] > 0 else ("显著为负" if r["hi"] < 0 else "含零·无证据")
        print(f"  {tag:<26}{r['n_days']:>10}{r['diff']:>+12.2f}{r['lo']:>+10.2f}"
              f"{r['hi']:>+10.2f}{r['p_gt0']:>9.2f}  {judge}")
    print()

    print(LINE)
    print("※ 数据来源：腾讯前复权日线与换手率（通达信数据源）、通达信指数日线、")
    print("※ 东方财富财报。以上均为历史统计与回测结论，不构成投资建议。")
    print(LINE)


if __name__ == "__main__":
    main()
