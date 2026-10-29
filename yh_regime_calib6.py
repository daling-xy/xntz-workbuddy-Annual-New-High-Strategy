# -*- coding: utf-8 -*-
"""
周期分层校准 VI（最终裁决）：把「绝对符号稳定」改成更公平的判据——
每个候选在同一周期、同一半样本里，相对 v1 原版是否都占优。
最终产出 v2 的分周期参数。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1200
_rng = np.random.default_rng(20261022)
V1 = lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8)

CAND = {
    "UP": {
        "v1 基准 prox>=0.90 W25 A4": V1,
        "prox>=0.95 W25 A2": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8),
        "prox>=0.95 W25 A2 T6": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.95 W15 A2 T6": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.90 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.85 W15 A2 T6": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
    },
    "CHOP": {
        "v1 基准 prox>=0.90 W25 A4": V1,
        "prox>=0.90 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.95 W15 A2 T6": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox0.90~0.98 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & (s["prox"] < 0.98) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.85 W15 A2 T6": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
    },
    "DOWN": {
        "v1 基准 prox>=0.90 W25 A4": V1,
        "prox>=0.85 W25 A2": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8),
        "prox>=0.85 W15 A2 T6": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox>=0.85 W15 A2 T3": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 3),
        "prox>=0.90 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox0.85~0.95 W15 A2 T6": lambda s: (s["prox"] >= 0.85) & (s["prox"] < 0.95) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
        "prox0.80~0.95 W15 A2 T6": lambda s: (s["prox"] >= 0.80) & (s["prox"] < 0.95) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6),
    },
}


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


def main(ev):
    print("=" * 128)
    print("S. 最终裁决：候选相对 v1 基准的「三段全胜」检验（全样本 / 2019-2022 / 2023-2026）")
    print("=" * 128)
    segs = [("全样本", lambda s: s),
            ("2019-2022", lambda s: s[s["year"] <= "2022"]),
            ("2023-2026", lambda s: s[s["year"] >= "2023"])]
    final = {}
    for r in ["UP", "CHOP", "DOWN"]:
        rows = []
        base_vals = {}
        for sn, sf in segs:
            s = sf(ev[ev["regime"] == r])
            base_vals[sn] = boot_ci(s[V1(s)], "ab_fwd60")[0] * 100
        for nm, fn in CAND[r].items():
            cells, deltas, cnt = {}, [], []
            for sn, sf in segs:
                s = sf(ev[ev["regime"] == r])
                sub = s[fn(s)]
                v = boot_ci(sub, "ab_fwd60")[0] * 100
                cells[sn] = v
                cnt.append(round(len(sub) / max(sub["date"].nunique(), 1), 1))
                if nm.startswith("v1"):
                    deltas.append(0.0)
                else:
                    deltas.append(v - base_vals[sn])
            rows.append({"参数集": nm, "全样本%": round(cells["全样本"], 2),
                         "前半%": round(cells["2019-2022"], 2), "后半%": round(cells["2023-2026"], 2),
                         "Δ全": round(deltas[0], 2), "Δ前": round(deltas[1], 2), "Δ后": round(deltas[2], 2),
                         "日均票": cnt[0],
                         "三段全胜": "✔" if (nm.startswith("v1") or min(deltas) > 0) else "✘"})
        df = pd.DataFrame(rows)
        print(f"\n--- [{r}] ---")
        print(df.to_string(index=False))
        win = df[(~df["参数集"].str.startswith("v1")) & (df["三段全胜"] == "✔")]
        if len(win):
            best = win.loc[win["Δ全"].idxmax()]
            final[r] = best["参数集"]
            print(f"    → 推荐：{best['参数集']}  （三段均优于 v1，Δ 提升 {best['Δ全']:+.2f}/{best['Δ前']:+.2f}/{best['Δ后']:+.2f} pp）")
    print()
    print("=" * 128)
    print("T. v2 分周期参数（用于 yh_screen.py 的 PARAMS_BY_REGIME）")
    print("=" * 128)
    for k, v in final.items():
        print(f"   {k:5s} → {v}")


if __name__ == "__main__":
    main(load())
