# -*- coding: utf-8 -*-
"""
实证：门槛越松 → 进来的票 beta 越高？

按「当前价/年高」分组，看每组的平均 beta、去均值超额、beta 调整超额。
用于验证「下跌市放宽 prox 门槛 = 系统性引入高 beta 样本」这个说法。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_beta import add_beta  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load():
    p = os.path.join(CACHE, "events_long_beta.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_5.pkl"))
    st = pd.read_pickle(os.path.join(CACHE, "regime_state.pkl"))[["date", "regime"]]
    return add_beta(ev.merge(st, on="date", how="inner"))


def main():
    ev = load()
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]

    print("=" * 92)
    print("A. 「距年高越远」的票，beta 是不是越高？（全样本，按当前价/年高分组）")
    print("=" * 92)
    bins = [(0.95, 1.01, ">=0.95 贴着年高"), (0.90, 0.95, "0.90~0.95"),
            (0.85, 0.90, "0.85~0.90"), (0.80, 0.85, "0.80~0.85"),
            (0.70, 0.80, "0.70~0.80"), (0.60, 0.70, "0.60~0.70"),
            (0.00, 0.60, "<0.60 深跌")]
    rows = []
    for lo, hi, tag in bins:
        d = ev[(ev["prox_now"] >= lo) & (ev["prox_now"] < hi)]
        if len(d) < 200:
            continue
        rows.append({"当前价/年高": tag, "事件数": len(d),
                     "beta中位": round(float(d["beta"].median()), 3),
                     "beta均值": round(float(d["beta"].mean()), 3),
                     "去均值超额%": round(float(d["ex_fwd60"].mean())*100, 2),
                     "beta调整超额%": round(float(d["ab_fwd60"].mean())*100, 2),
                     "绝对收益%": round(float(d["fwd60"].mean())*100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=" * 92)
    print("B. 只看【下跌市 DOWN】：放宽门槛带来的到底是不是高 beta？")
    print("=" * 92)
    dn = ev[ev["regime"] == "DOWN"]
    rows = []
    for lo, hi, tag in bins:
        d = dn[(dn["prox_now"] >= lo) & (dn["prox_now"] < hi)]
        if len(d) < 100:
            continue
        rows.append({"当前价/年高": tag, "事件数": len(d),
                     "beta中位": round(float(d["beta"].median()), 3),
                     "去均值超额%": round(float(d["ex_fwd60"].mean())*100, 2),
                     "beta调整超额%": round(float(d["ab_fwd60"].mean())*100, 2),
                     "绝对收益%": round(float(d["fwd60"].mean())*100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=" * 92)
    print("C. 关键对照：DOWN 期「贴年高」vs「深跌」两组的收益来源拆解")
    print("=" * 92)
    for tag, m in [("贴年高 prox_now>=0.95", dn["prox_now"] >= 0.95),
                   ("深跌   prox_now<0.70", dn["prox_now"] < 0.70)]:
        d = dn[m]
        if len(d) < 50:
            continue
        print(f"  {tag}  n={len(d)}")
        print(f"    beta 中位        : {d['beta'].median():.3f}")
        print(f"    市场同期收益中位  : {d['mkt_fwd60'].median():+.2%}")
        print(f"    绝对收益均值      : {d['fwd60'].mean():+.2%}")
        print(f"    beta×市场(应得)   : {(d['beta']*d['mkt_fwd60']).mean():+.2%}")
        print(f"    去均值超额        : {d['ex_fwd60'].mean():+.2%}   <- 未剔除 beta")
        print(f"    beta 调整超额     : {d['ab_fwd60'].mean():+.2%}   <- 剔除 beta 后")
        print()


    print("=" * 92)
    print("D. 【关键】DOWN 期 + 完整纪律（箱宽<=15% 额>=4亿 换手<=6%）后，再按距年高分组")
    print("    —— 对比 B 段（无任何纪律），看「同一个因子」的结论会不会翻转")
    print("=" * 92)
    g = dn[(dn["width30"] <= 0.15) & (dn["amt30"] >= 4e8) & (dn["turn30"] <= 6.0)]
    print(f"    满足纪律的事件数: {len(g)}")
    rows = []
    for lo, hi, tag in bins:
        d = g[(g["prox_now"] >= lo) & (g["prox_now"] < hi)]
        if len(d) < 50:
            continue
        rows.append({"当前价/年高": tag, "事件数": len(d),
                     "beta中位": round(float(d["beta"].median()), 3),
                     "去均值超额%": round(float(d["ex_fwd60"].mean()) * 100, 2),
                     "beta调整超额%": round(float(d["ab_fwd60"].mean()) * 100, 2),
                     "绝对收益%": round(float(d["fwd60"].mean()) * 100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=" * 92)
    print("E. 同一个因子的结论翻转对照（DOWN 期，深跌 <0.70 vs 贴年高 >=0.95）")
    print("=" * 92)
    for tag, m in [("贴年高 >=0.95", 0.95), ("深跌   <0.70", 0.70)]:
        if m == 0.95:
            a, b = dn[dn["prox_now"] >= 0.95], g[g["prox_now"] >= 0.95]
        else:
            a, b = dn[dn["prox_now"] < 0.70], g[g["prox_now"] < 0.70]
        if len(a) < 50 or len(b) < 50:
            continue
        print(f"  {tag}")
        print(f"    无纪律 : n={len(a):>7}  beta {a['beta'].median():.3f}  "
              f"去均值 {a['ex_fwd60'].mean()*100:+.2f}%  beta调整 {a['ab_fwd60'].mean()*100:+.2f}%")
        print(f"    加纪律 : n={len(b):>7}  beta {b['beta'].median():.3f}  "
              f"去均值 {b['ex_fwd60'].mean()*100:+.2f}%  beta调整 {b['ab_fwd60'].mean()*100:+.2f}%")
        print()


if __name__ == "__main__":
    main()
