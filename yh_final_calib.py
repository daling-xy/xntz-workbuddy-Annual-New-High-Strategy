# -*- coding: utf-8 -*-
"""
【终局定参】在修复后的数据上重新搜索 v2 参数。

上一版 yh_fix_verify 的 Z1（唯一有效的那段）已经暴露一个必须撤回的改动：
  把成交额门槛从 4 亿降到 2 亿，在【相对流动性口径】下对 DOWN/CHOP 都是负贡献。
本脚本在两种流动性口径 + 纯老股子样本上，重新搜索 prox × 箱宽 × 换手 的三段全胜解，
并把「是否放松成交额」单独作为一项来裁决。

v1 基准固定为： prox>=0.90 / 箱宽<=25% / 日均额>=4亿 / 换手 1.5~10%
所有增量都是「相对同期同口径 v1 的 beta 调整超额之差」。
"""
from __future__ import annotations

import os
import sys
import itertools

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
    return ev


def mask(s, prox, w, amt=None, amt_pct=None, t0=0.0, t1=6.0):
    m = ((s["prox"] >= prox) & (s["width30"] <= w)
         & (s["turn30"] >= t0) & (s["turn30"] <= t1))
    if amt is not None:
        m = m & (s["amt30"] >= amt)
    if amt_pct is not None:
        m = m & (s["amt_pct"] >= amt_pct)
    return m


V1 = dict(prox=0.90, w=0.25, amt=4e8, t0=1.5, t1=10.0)
V1P = dict(prox=0.90, w=0.25, amt_pct=0.85, t0=1.5, t1=10.0)

SEGS = [("全", None, None), ("前", None, "2022"), ("后", "2023", None)]


def seg_frames(ev, regime, old_only=False):
    out = []
    for sn, y0, y1 in SEGS:
        s = ev[ev["regime"] == regime]
        if old_only:
            s = s[s["is_old"] == 1]
        if y0:
            s = s[s["year"] >= y0]
        if y1:
            s = s[s["year"] <= y1]
        out.append((sn, s))
    return out


def eval_cfg(ev, regime, cfg, base, old_only=False):
    """返回 (Δ全, Δ前, Δ后, 全胜, 日均票, 全样本beta%)"""
    deltas, cnt = [], []
    full_v = np.nan
    for i, (sn, s) in enumerate(seg_frames(ev, regime, old_only)):
        v = boot_ci(s[mask(s, **cfg)], "ab_fwd60")[0]
        b = boot_ci(s[mask(s, **base)], "ab_fwd60")[0]
        d = (v - b) * 100
        deltas.append(d)
        if i == 0:
            sub = s[mask(s, **cfg)]
            full_v = v * 100
            cnt.append(len(sub) / max(sub["date"].nunique(), 1))
    ok = all(d > 0 for d in deltas)
    return (*[round(d, 2) for d in deltas], "✔" if ok else "✘",
            round(cnt[0], 1), round(full_v, 2))


def part_A(ev):
    """裁决：成交额门槛该不该从 4 亿放松？"""
    print("=" * 130)
    print("A. 裁决一：成交额门槛该不该从 4 亿放松？（固定 prox=0.90 / 箱宽=25% / 换手 1.5~10%，只变流动性）")
    print("=" * 130)
    for r in ["UP", "CHOP", "DOWN"]:
        rows = []
        for nm, cfg in [
            ("4亿 (v1原值)", dict(prox=0.90, w=0.25, amt=4e8, t0=1.5, t1=10.0)),
            ("2亿", dict(prox=0.90, w=0.25, amt=2e8, t0=1.5, t1=10.0)),
            ("1亿", dict(prox=0.90, w=0.25, amt=1e8, t0=1.5, t1=10.0)),
            ("前15% (≈4亿)", dict(prox=0.90, w=0.25, amt_pct=0.85, t0=1.5, t1=10.0)),
            ("前25% (≈2亿)", dict(prox=0.90, w=0.25, amt_pct=0.75, t0=1.5, t1=10.0)),
            ("前40%", dict(prox=0.90, w=0.25, amt_pct=0.60, t0=1.5, t1=10.0)),
        ]:
            base = V1P if "amt_pct" in cfg else V1
            sub = ev[ev["regime"] == r]
            v, lo, hi = boot_ci(sub[mask(sub, **cfg)], "ab_fwd60")
            rows.append({"流动性门槛": nm, "beta调整%": round(v * 100, 2),
                         "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                         "日均票": round(len(sub[mask(sub, **cfg)]) / sub["date"].nunique(), 1)})
        print(f"\n--- [{r}] （同一基准口径内横向可比）---")
        print(pd.DataFrame(rows).to_string(index=False))
    print()


def part_B(ev):
    """固定流动性=4亿(v1原值)，搜 prox × 箱宽 × 换手 的三段全胜解"""
    print("=" * 130)
    print("B. 裁决二：固定成交额 4 亿不动，只搜 prox × 箱宽 × 换手（相对同期 v1 的三段全胜）")
    print("=" * 130)
    grid = []
    for prox in [0.90, 0.95]:
        for w in [0.15, 0.20]:
            for t0, t1 in [(0.0, 6.0), (1.0, 6.0), (1.5, 6.0), (0.0, 10.0)]:
                grid.append(dict(prox=prox, w=w, amt=4e8, t0=t0, t1=t1))
    best = {}
    for r in ["UP", "CHOP", "DOWN"]:
        rows = []
        for cfg in grid:
            d0, d1, d2, ok, c, v = eval_cfg(ev, r, cfg, V1)
            if not np.isfinite(v):
                continue
            rows.append({"prox": cfg["prox"], "箱宽": f"{cfg['w']:.0%}",
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
            print(f"    → 推荐 prox>={b['prox']} / 箱宽<={b['箱宽']} / 换手 {b['换手']} "
                  f"（beta调整 {b['beta调整%']:+.2f}%，Δ {b['Δ全']:+.2f}/{b['Δ前']:+.2f}/{b['Δ后']:+.2f}pp，日均 {b['日均票']} 只）")
        else:
            print("    → 无三段全胜解")
    print()
    return best


def part_C(ev, best):
    """纯老股子样本复核（剔除幸存者偏差嫌疑样本）"""
    print("=" * 130)
    print("C. 裁决三：纯「老股」子样本复核（剔除 2019-06 之后上市的股票，样本 3367 只 / 115 万事件）")
    print("=" * 130)
    rows = []
    for r, b in best.items():
        cfg = dict(prox=float(b["prox"]), w=float(b["箱宽"].strip("%")) / 100,
                   amt=4e8, t0=float(b["换手"].split("~")[0]), t1=float(b["换手"].split("~")[1].strip("%")))
        d0, d1, d2, ok, c, v = eval_cfg(ev, r, cfg, V1, old_only=True)
        rows.append({"周期": r, "参数": f"prox>={cfg['prox']:.2f} 箱宽<={cfg['w']:.0%} 额>=4亿 换手{cfg['t0']}~{cfg['t1']:.0f}%",
                     "老股样本beta调整%": v, "Δ全": d0, "Δ前": d1, "Δ后": d2,
                     "日均票": c, "三段全胜": ok})
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def part_D(ev, best):
    print("=" * 130)
    print("D. 最终推荐（成交额门槛维持 v1 的 4 亿不放宽）")
    print("=" * 130)
    rows = []
    for r, b in best.items():
        rows.append({"周期": r,
                     "prox_min": float(b["prox"]),
                     "width30_max": b["箱宽"],
                     "amt30_min": "4亿（不变）",
                     "turn30": b["换手"],
                     "beta调整超额%": b["beta调整%"],
                     "相对v1增量pp": b["Δ全"],
                     "日均票": b["日均票"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  与上一版（未修复数据）的差异：成交额门槛从 2 亿【撤回】为 4 亿。")


if __name__ == "__main__":
    ev = load()
    part_A(ev)
    best = part_B(ev)
    if best:
        part_C(ev, best)
        part_D(ev, best)
