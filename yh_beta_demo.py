# -*- coding: utf-8 -*-
"""
β 计算全链路演示（用真实数据）

目的：把「Cov / Var / β」和「分组求 β 中位数」这两件事拆开看清：
  第一步：每只股票【自己】先算出一个 β（跟任何分组、任何筛选条件都无关）
  第二步：事后把事件按「当前价/年高」分档，只做一件事——数一数每档里 β 的中位数
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_beta import build_market_ret, BETA_W  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BETA_W = 250


def demo_one_stock(code: str, asof: str, kmap, mkt: pd.Series):
    """对一只票，用 asof 之前 250 个交易日手算 β，逐步打印中间量。"""
    bars = kmap.get(code)
    if not bars:
        print(f"  找不到 {code}")
        return None
    dates = [b["date"] for b in bars]
    if asof not in dates:
        print(f"  {code} 没有 {asof} 这一天")
        return None
    i = dates.index(asof)
    if i < BETA_W:
        print(f"  {code} 历史不足 250 天")
        return None

    cl = np.array([float(b["close"]) for b in bars])
    r = np.empty(len(cl)); r[0] = np.nan
    r[1:] = cl[1:] / np.maximum(cl[:-1], 1e-9) - 1.0        # 个股日收益

    mv = mkt.values
    mpos = {d: k for k, d in enumerate(mkt.index)}
    m = np.array([mv[mpos[d]] if d in mpos else np.nan for d in dates])  # 市场日收益

    j = i - BETA_W + 1
    rr, mm = r[j:i+1], m[j:i+1]
    ok = np.isfinite(rr) & np.isfinite(mm)
    rr, mm = rr[ok], mm[ok]
    n = len(rr)

    rbar, mbar = rr.mean(), mm.mean()
    dev_r, dev_m = rr - rbar, mm - mbar
    cov_num = float((dev_r * dev_m).sum())          # Σ(r−r̄)(m−m̄)
    var_num = float((dev_m ** 2).sum())             # Σ(m−m̄)²
    # 样本版（除以 n−1）—— 但 β 是比值，(n−1) 上下约掉，结果完全一样
    cov_s = cov_num / (n - 1)
    var_s = var_num / (n - 1)
    beta = cov_s / var_s

    print(f"\n  【{code}】截至 {asof}，用前 {n} 个交易日")
    print(f"  个股日收益均值 r̄ = {rbar*100:+.4f}%    市场日收益均值 m̄ = {mbar*100:+.4f}%")
    print(f"\n  前 8 天的明细（%）：")
    print(f"  {'日':>3} {'个股 r':>9} {'市场 m':>9} {'r−r̄':>9} {'m−m̄':>9} {'(r−r̄)(m−m̄)':>13} {'(m−m̄)²':>10}")
    for k in range(8):
        print(f"  {k+1:>3} {rr[k]*100:>8.3f}% {mm[k]*100:>8.3f}% "
              f"{dev_r[k]*100:>8.3f} {dev_m[k]*100:>8.3f} "
              f"{dev_r[k]*dev_m[k]*1e4:>12.4f} {dev_m[k]**2*1e4:>9.4f}")
    print(f"  ...（共 {n} 天）")
    print(f"\n  Σ(r−r̄)(m−m̄) = {cov_num*1e4:.3f}（万分之一单位）")
    print(f"  Σ(m−m̄)²      = {var_num*1e4:.3f}")
    print(f"  Cov(r,m) = {cov_num*1e4:.3f} / {n-1} = {cov_s*1e4:.4f}")
    print(f"  Var(m)   = {var_num*1e4:.3f} / {n-1} = {var_s*1e4:.4f}")
    print(f"  β = Cov ÷ Var = {cov_s*1e4:.4f} ÷ {var_s*1e4:.4f} = {beta:.4f}")
    return beta


def main():
    mkt = build_market_ret()
    with open(os.path.join(CACHE, "kline_long2.pkl"), "rb") as f:
        kmap = pickle.load(f)

    print("=" * 96)
    print("第一步：β 是【每一只股票自己】的一个属性，跟任何筛选条件无关")
    print("=" * 96)
    for code, asof in [("600030", "2026-10-21"), ("601108", "2026-10-21"),
                       ("600519", "2026-10-21")]:
        demo_one_stock(code, asof, kmap, mkt)

    print("\n" + "=" * 96)
    print("第二步：事后分组 —— 只是「数一数每组里 β 的中位数是多少」，不再重算 β")
    print("=" * 96)
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    sub = ev[ev["beta"].notna()].copy()
    print(f"  事件总数（beta 非空）: {len(sub)}")
    print(f"  每个事件自带一个 beta（该股在该日事前算好的）→ 分组只是取中位数\n")

    bins = [(0.95, 1.01, ">=0.95"), (0.90, 0.95, "0.90~0.95"),
            (0.85, 0.90, "0.85~0.90"), (0.70, 0.80, "0.70~0.80"),
            (0.00, 0.60, "<0.60")]
    rows = []
    for lo, hi, tag in bins:
        d = sub[(sub["prox_now"] >= lo) & (sub["prox_now"] < hi)]
        rows.append({"当前价/年高": tag, "事件数": len(d),
                     "β中位数": round(float(d["beta"].median()), 3),
                     "β均值": round(float(d["beta"].mean()), 3),
                     "β的10分位": round(float(d["beta"].quantile(.1)), 3),
                     "β的90分位": round(float(d["beta"].quantile(.9)), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  注意：这里的分组【没有加任何筛选条件】，是全样本。")
    print("  β 不是分组之后算的，是先有 β、再按别的条件分组、再看这组 β 的分布。")

    print("\n" + "=" * 96)
    print("第三步：α（beta 调整超额）为什么叫这个名字")
    print("=" * 96)
    dn = sub[(sub["regime"] == "DOWN") & sub["mkt_fwd60"].notna() & sub["fwd60"].notna()]
    print(f"  下跌市事件数: {len(dn)}")
    print("\n  拿两组出来做收益拆解（数字为该组均值）：\n")
    print(f"  {'组':<16}{'β中位':>8}{'市场60日':>10}{'绝对收益':>10}{'β×市场':>10}{'去均值超额':>12}{'β调整超额':>12}")
    for tag, mask in [("贴年高 >=0.95", dn["prox_now"] >= 0.95),
                      ("深跌 <0.70", dn["prox_now"] < 0.70)]:
        d = dn[mask]
        print(f"  {tag:<16}{d['beta'].median():>8.3f}"
              f"{d['mkt_fwd60'].mean()*100:>9.2f}%"
              f"{d['fwd60'].mean()*100:>9.2f}%"
              f"{(d['beta']*d['mkt_fwd60']).mean()*100:>9.2f}%"
              f"{d['ex_fwd60'].mean()*100:>11.2f}%"
              f"{d['ab_fwd60'].mean()*100:>11.2f}%")
    print("\n  读法：'β×市场' 这一列 = 市场本来就会给你的；")
    print("        '绝对收益' 减去它 = 'β调整超额'，也就是市场解释完之后剩下的部分 = α。")


if __name__ == "__main__":
    main()
