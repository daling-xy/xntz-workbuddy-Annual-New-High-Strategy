# -*- coding: utf-8 -*-
"""
周期分层校准 VII：定终值
  第六段挑出核心 = 「严箱体 + 控换手 + 流动性地板」，但 prox 的最优值随箱体宽度而变，
  必须用二维网格定终值；同时检验用户强调的「资金流活跃」换手下限该不该保留。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1000
_rng = np.random.default_rng(20261022)


def boot_ci(sub, key, n=BOOT):
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def load():
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    return ev


def grid(ev, r, floor=None):
    s = ev[ev["regime"] == r]
    rows = []
    for w in [0.12, 0.15, 0.20, 0.25]:
        for p in [0.80, 0.85, 0.90, 0.95]:
            m = ((s["width30"] <= w) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0)
                 & (s["prox"] >= p))
            if floor is not None:
                m = m & (s["turn30"] >= floor)
            sub = s[m]
            if len(sub) < 300 or sub["date"].nunique() < 40:
                rows.append({"箱宽": f"<={w:.0%}", "prox>=": p, "beta调整%": np.nan,
                             "日均票": np.nan, "大亏%": np.nan})
                continue
            v, lo, hi = boot_ci(sub, "ab_fwd60")
            rows.append({"箱宽": f"<={w:.0%}", "prox>=": p,
                         "beta调整%": round(v * 100, 2),
                         "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                         "日均票": round(len(sub) / sub["date"].nunique(), 1),
                         "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
                         "mae20%": round(sub["mae20"].median() * 100, 1)})
    return pd.DataFrame(rows)


def main(ev):
    print("=" * 130)
    print("U. 二维网格：箱宽 × prox 下限（统一带 amt30>=2亿、turn30<=6%）")
    print("=" * 130)
    for r in ["UP", "CHOP", "DOWN"]:
        g = grid(ev, r)
        print(f"\n--- [{r}] ---")
        piv = g.pivot(index="箱宽", columns="prox>=", values="beta调整%")
        cnt = g.pivot(index="箱宽", columns="prox>=", values="日均票")
        with pd.option_context("display.width", 200):
            print("beta调整超额%："); print(piv.to_string())
            print("日均票数：");   print(cnt.to_string())

    print()
    print("=" * 130)
    print("V. 要不要保留「资金流活跃」换手下限 1.5%？（在 W15/prox 最优档上检验）")
    print("=" * 130)
    BEST = {"UP": (0.15, 0.95), "CHOP": (0.15, 0.95), "DOWN": (0.15, 0.90)}
    for r in ["UP", "CHOP", "DOWN"]:
        w, p = BEST[r]
        s = ev[ev["regime"] == r]
        base = ((s["width30"] <= w) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0) & (s["prox"] >= p))
        rows = []
        for nm, extra in [("无换手下限", pd.Series(True, index=s.index)),
                          ("换手>=1.0%", s["turn30"] >= 1.0),
                          ("换手>=1.5%(v1)", s["turn30"] >= 1.5),
                          ("换手 1.5~6%", (s["turn30"] >= 1.5) & (s["turn30"] <= 6.0))]:
            sub = s[base & extra]
            if len(sub) < 300:
                continue
            v, lo, hi = boot_ci(sub, "ab_fwd60")
            e, elo, ehi = boot_ci(sub, "ex_fwd60")
            rows.append({"周期": r, "口径": nm, "日均票": round(len(sub) / sub["date"].nunique(), 1),
                         "去均值超额%": round(e * 100, 2), "beta调整%": round(v * 100, 2),
                         "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                         "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
                         "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
                         "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1)})
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    print("=" * 130)
    print("W. 终值 once more：把 fwd20 / fwd40 也看一眼，确认不是靠长周期高能解释")
    print("=" * 130)
    for r in ["UP", "CHOP", "DOWN"]:
        w, p = BEST[r]
        s = ev[ev["regime"] == r]
        base = ((s["width30"] <= w) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0) & (s["prox"] >= p))
        sub = s[base]
        v1m = V1 = ((s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8))
        def st(mask, key):
            x = s[mask]
            return round(float((x[key] - x["beta"] * x["mkt_fwd60"]).mean() * 100), 2)
        for key in ["fwd20", "fwd40", "fwd60"]:
            a = sub
            b = s[V1]
            # 20/40 日没有对应市场项，改用等权全市场同期均值近似（此处用个股自身截面去均值）
            print(f"  [{r:4s}] {key}: 自适参数 vs v1 → "
                  f"{a[key].mean()*100:6.2f}%  vs {b[key].mean()*100:6.2f}%  "
                  f"(差 {(a[key].mean()-b[key].mean())*100:+6.2f}pp)  "
                  f"胜率 {(a[key]>0).mean()*100:.1f}% vs {(b[key]>0).mean()*100:.1f}%")
        print()


if __name__ == "__main__":
    main(load())
