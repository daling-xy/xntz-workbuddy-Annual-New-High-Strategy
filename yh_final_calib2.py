# -*- coding: utf-8 -*-
"""
【终局定参 v3】两处口径修正后的重新搜索：
  修正1  prox 改用【当前收盘/250日最高收盘】（即 1+dd_from_high），而不是【近30日最高/年高】。
         证据：def_check E 段——DOWN 期 0.95 档，B口径 +3.60% vs A口径 +2.31%；
         UP +4.22% vs +1.90%；胜率与回撤也全面变好。
         直觉上也更贴合「距年高」的原意：A口径会让「从年高回落15%后横盘」的票通过。
  修正2  成交额门槛维持 4 亿不放宽（上一版降到 2 亿的证据不稳定，撤回）。
  修正3  剔除 ret_std 死列，箱宽仍用振幅口径（比波动率门槛更有效）。

v1 基准固定： A口径 prox>=0.90 / 箱宽<=25% / 额>=4亿 / 换手 1.5~10%
"""
from __future__ import annotations

import os
import sys

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
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    return ev


def M(s, prox_now=None, prox=None, w=None, amt=None, t0=0.0, t1=6.0):
    m = (s["turn30"] >= t0) & (s["turn30"] <= t1)
    if prox_now is not None:
        m = m & (s["prox_now"] >= prox_now)
    if prox is not None:
        m = m & (s["prox"] >= prox)
    if w is not None:
        m = m & (s["width30"] <= w)
    if amt is not None:
        m = m & (s["amt30"] >= amt)
    return m


V1 = dict(prox=0.90, w=0.25, amt=4e8, t0=1.5, t1=10.0)
SEGS = [("全", None, None), ("前", None, "2022"), ("后", "2023", None)]


def eval_cfg(ev, r, cfg, old_only=False):
    deltas, cnt, full_v = [], [], np.nan
    for i, (sn, y0, y1) in enumerate(SEGS):
        s = ev[ev["regime"] == r]
        if old_only:
            s = s[s["is_old"] == 1]
        if y0:
            s = s[s["year"] >= y0]
        if y1:
            s = s[s["year"] <= y1]
        v = boot_ci(s[M(s, **cfg)], "ab_fwd60")[0]
        b = boot_ci(s[M(s, **V1)], "ab_fwd60")[0]
        deltas.append((v - b) * 100)
        if i == 0:
            sub = s[M(s, **cfg)]
            full_v = v * 100
            cnt.append(len(sub) / max(sub["date"].nunique(), 1))
    ok = all(d > 0 for d in deltas)
    return (*[round(d, 2) for d in deltas], "✔" if ok else "✘",
            round(cnt[0], 1), round(full_v, 2))


GRID = []
for p in [0.90, 0.95]:
    for w in [0.15, 0.20]:
        for t0, t1 in [(0.0, 6.0), (1.5, 6.0), (0.0, 10.0)]:
            GRID.append(dict(prox_now=p, w=w, amt=4e8, t0=t0, t1=t1))


def main(ev):
    print("=" * 130)
    print("G. 用修正口径【当前价/年高】重新搜索（相对同期 v1 的三段全胜）")
    print("=" * 130)
    best = {}
    for r in ["UP", "CHOP", "DOWN"]:
        rows = []
        for cfg in GRID:
            d0, d1, d2, ok, c, v = eval_cfg(ev, r, cfg)
            if not np.isfinite(v):
                continue
            rows.append({"prox_now": cfg["prox_now"], "箱宽": f"{cfg['w']:.0%}",
                         "换手": f"{cfg['t0']}~{cfg['t1']:.0f}%",
                         "beta调整%": v, "Δ全": d0, "Δ前": d1, "Δ后": d2,
                         "日均票": c, "三段全胜": ok})
        df = pd.DataFrame(rows)
        print(f"\n--- [{r}] ---")
        with pd.option_context("display.width", 240, "display.max_columns", 40):
            print(df.to_string(index=False))
        win = df[df["三段全胜"] == "✔"]
        if len(win):
            b = win.loc[win["Δ全"].idxmax()]
            best[r] = b
            print(f"    → 推荐 当前价/年高>={b['prox_now']:.2f} / 箱宽<={b['箱宽']} / 换手{b['换手']} "
                  f"（beta调整 {b['beta调整%']:+.2f}%，Δ {b['Δ全']:+.2f}/{b['Δ前']:+.2f}/{b['Δ后']:+.2f}pp，日均 {b['日均票']} 只）")
        else:
            best[r] = df.loc[df["beta调整%"].idxmax()]
            print("    → 无三段全胜解，退而取全样本最优")

    print()
    print("=" * 130)
    print("H. 老股子样本复核（剔除幸存者偏差嫌疑样本）")
    print("=" * 130)
    rows = []
    for r, b in best.items():
        cfg = dict(prox_now=float(b["prox_now"]), w=float(b["箱宽"].strip("%")) / 100,
                   amt=4e8, t0=float(b["换手"].split("~")[0]),
                   t1=float(b["换手"].split("~")[1].strip("%")))
        d0, d1, d2, ok, c, v = eval_cfg(ev, r, cfg, old_only=True)
        rows.append({"周期": r,
                     "参数": f"当前价/年高>={cfg['prox_now']:.2f} 箱宽<={cfg['w']:.0%} 额>=4亿 换手{cfg['t0']}~{cfg['t1']:.0f}%",
                     "老股beta调整%": v, "Δ全": d0, "Δ前": d1, "Δ后": d2, "日均票": c, "三段全胜": ok})
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("=" * 130)
    print("I. v3 最终参数表")
    print("=" * 130)
    rows = []
    for r, b in best.items():
        rows.append({"周期": r, "prox(当前价/年高)": float(b["prox_now"]),
                     "箱宽": b["箱宽"], "成交额": "4亿", "换手": b["换手"],
                     "beta调整超额%": b["beta调整%"], "相对v1(pp)": b["Δ全"],
                     "日均票": b["日均票"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  三项修正：① prox 改用【当前价/年高】；② 成交额维持 4 亿（撤回降到 2 亿）；")
    print("            ③ 箱宽 = 30日振幅（非波动率），15% 对应日波动中位 1.73%、年化 27.3%。")


if __name__ == "__main__":
    main(load())
