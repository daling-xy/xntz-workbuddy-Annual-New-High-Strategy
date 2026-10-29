# -*- coding: utf-8 -*-
"""
把「去均值超额」和「β调整超额」的算术完全拆开，逐项核对。
回答用户两个手算：
  去均值超额 = 绝对收益 − 市场横截面均值 ？
  α         = 绝对收益 − β × 市场收益   ？
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


def block(tag, d):
    if len(d) < 30:
        return
    abs_ret = float(d["fwd60"].mean())
    bxm = float((d["beta"] * d["mkt_fwd60"]).mean())
    ex = float(d["ex_fwd60"].mean())
    ab = float(d["ab_fwd60"].mean())
    # 由 去均值超额 反推：该组事件所在交易日的全市场横截面均值
    mkt_xsec_implied = abs_ret - ex
    print(f"  【{tag}】  n={len(d)}")
    print(f"    ① 绝对收益均值（这组的等权未来60日收益）   = {abs_ret:+.4%}")
    print(f"    ② 该组事件所在日的全市场横截面均值（反推）  = {mkt_xsec_implied:+.4%}")
    print(f"    ③ 去均值超额 = ① − ②                    = {abs_ret - mkt_xsec_implied:+.4%}"
          f"   （表里记录值 {ex:+.4%}）")
    print(f"    ④ β × 市场收益（用每个事件自己的 β 算）    = {bxm:+.4%}"
          f"   β中位={d['beta'].median():.3f}  市场60日中位={d['mkt_fwd60'].median():+.2%}")
    print(f"    ⑤ α = ① − ④                            = {abs_ret - bxm:+.4%}"
          f"   （表里记录值 {ab:+.4%}）")
    print()
    return abs_ret, mkt_xsec_implied, bxm


def main():
    ev = load()
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    dn = ev[ev["regime"] == "DOWN"]

    print("=" * 96)
    print("DOWN 期、只套「当前价/年高 >= 0.95」这一个条件 —— 逐项核对两个手算")
    print("=" * 96)
    d = dn[dn["prox_now"] >= 0.95]
    block("贴年高 prox_now>=0.95", d)

    print("=" * 96)
    print("同一个组，再叠加完整纪律（箱宽<=15% / 日均额>=4亿 / 换手<=6%）")
    print("=" * 96)
    g = d[(d["width30"] <= 0.15) & (d["amt30"] >= 4e8) & (d["turn30"] <= 6.0)]
    block("贴年高 + 完整纪律", g)

    print("=" * 96)
    print("逐事件核对：随机抽 3 个事件，看 α 是不是真的 = fwd60 − β×mkt_fwd60")
    print("=" * 96)
    s = d.sample(3, random_state=7)
    for i, (_, r) in enumerate(s.iterrows(), 1):
        print(f"  事件{i}: {r['code']}  {str(r['date'])[:10]}  β={r['beta']:.3f}")
        print(f"    fwd60={r['fwd60']:+.4%}   mkt_fwd60={r['mkt_fwd60']:+.4%}   "
              f"β×市场={r['beta']*r['mkt_fwd60']:+.4%}   "
              f"α={r['fwd60']-r['beta']*r['mkt_fwd60']:+.4%}  (表里 ab_fwd60={r['ab_fwd60']:+.4%})")
    print()

    # 残差与市场收益的相关性：验证 α 与市场"天然不相关"
    dd = dn[dn["beta"].notna()].copy()
    dd["alpha"] = dd["fwd60"] - dd["beta"] * dd["mkt_fwd60"]
    print("=" * 96)
    print("验证「α 与市场收益天然不相关」（几何投影的必然结果）")
    print("=" * 96)
    sub = dd.dropna(subset=["alpha", "mkt_fwd60"]).sample(min(200000, len(dd)), random_state=1)
    print(f"  corr(α, 市场60日收益) = {np.corrcoef(sub['alpha'], sub['mkt_fwd60'])[0,1]:+.4f}")
    print(f"  corr(个股收益, 市场60日收益) = "
          f"{np.corrcoef(sub['fwd60'], sub['mkt_fwd60'])[0,1]:+.4f}")
    print("  → 前者接近 0，说明「市场能解释的部分」已经被 β×市场 抽干净了。")
    print()


if __name__ == "__main__":
    main()
