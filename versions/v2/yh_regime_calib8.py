# -*- coding: utf-8 -*-
"""
周期分层校准 VIII：确定 v2 终值并做最后的时段稳健性体检。
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


V1 = lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8)

POL = {
    "UP": {
        "A 无换手下限定": (0.95, 0.15, 2e8, 0.0, 6.0),
        "B 换手 1.0~6%": (0.95, 0.15, 2e8, 1.0, 6.0),
        "C 换手 1.5~6%": (0.95, 0.15, 2e8, 1.5, 6.0),
    },
    "CHOP": {
        "A 无换手下限定": (0.95, 0.15, 2e8, 0.0, 6.0),
        "B 换手 1.0~6%": (0.95, 0.15, 2e8, 1.0, 6.0),
        "C 换手 1.5~6%": (0.95, 0.15, 2e8, 1.5, 6.0),
    },
    "DOWN": {
        "A 无换手下限定": (0.90, 0.15, 2e8, 0.0, 6.0),
        "B 换手 1.0~6%": (0.90, 0.15, 2e8, 1.0, 6.0),
        "C 放宽prox到0.85": (0.85, 0.15, 2e8, 0.0, 6.0),
        "D 放宽prox到0.80": (0.80, 0.15, 2e8, 0.0, 6.0),
    },
}


def mask(s, cfg):
    p, w, a, t0, t1 = cfg
    return ((s["prox"] >= p) & (s["width30"] <= w) & (s["amt30"] >= a)
            & (s["turn30"] >= t0) & (s["turn30"] <= t1))


SEGS = [("2019-2020", "2019", "2020"), ("2021-2022", "2021", "2022"),
        ("2023-2024", "2023", "2024"), ("2025-2026", "2025", "2026")]


def main(ev):
    print("=" * 132)
    print("X. v2 终值候选：每 ~2 年一段的稳健性体检（相对同期 v1 基准的差值，单位 pp）")
    print("=" * 132)
    for r in ["UP", "CHOP", "DOWN"]:
        print(f"\n--- [{r}] ---")
        rows = []
        for nm, cfg in POL[r].items():
            rec = {"口径": nm}
            allv = []
            for sn, y0, y1 in SEGS:
                s = ev[(ev["regime"] == r) & (ev["year"] >= y0) & (ev["year"] <= y1)]
                v = boot_ci(s[mask(s, cfg)], "ab_fwd60")[0] * 100
                b = boot_ci(s[V1(s)], "ab_fwd60")[0] * 100
                rec[sn] = round(v, 2)
                rec[sn + "Δ"] = round(v - b, 2)
                allv.append(v - b)
            full = ev[ev["regime"] == r]
            vf = boot_ci(full[mask(full, cfg)], "ab_fwd60")[0] * 100
            bf = boot_ci(full[V1(full)], "ab_fwd60")[0] * 100
            rec["全样本Δ"] = round(vf - bf, 2)
            rec["四段皆胜"] = "✔" if min(allv) > 0 else "✘"
            rows.append(rec)
        cols = ["口径"] + [c for sn, _, _ in SEGS for c in (sn, sn + "Δ")] + ["全样本Δ", "四段皆胜"]
        df = pd.DataFrame(rows)[cols]
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(df.to_string(index=False))

    print()
    print("=" * 132)
    print("Y. 最终采用的分周期参数（写入 yh_screen.py::PARAMS_BY_REGIME）")
    print("=" * 132)
    FINAL = {
        "UP": ("prox>=0.95", 0.95, 0.15, 2e8, 1.0, 6.0),
        "CHOP": ("prox>=0.95", 0.95, 0.15, 2e8, 1.5, 6.0),
        "DOWN": ("prox>=0.90", 0.90, 0.15, 2e8, 0.0, 6.0),
    }
    rows = []
    for r in ["UP", "CHOP", "DOWN"]:
        nm = FINAL[r][0]
        full = ev[ev["regime"] == r]
        sub = full[mask(full, FINAL[r][1:])]
        v, lo, hi = boot_ci(sub, "ab_fwd60")
        e, elo, ehi = boot_ci(sub, "ex_fwd60")
        bv, _, _ = boot_ci(full[V1(full)], "ab_fwd60")
        rows.append({"周期": r, "参数": nm + " 箱宽<=15% 额>=2亿 换手<=6%",
                     "日均票": round(len(sub) / sub["date"].nunique(), 1),
                     "beta调整超额%": round(v * 100, 2), "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                     "vs v1(pp)": round((v - bv) * 100, 2),
                     "去均值超额%": round(e * 100, 2),
                     "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
                     "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
                     "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
                     "mae20%": round(sub["mae20"].median() * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(load())
