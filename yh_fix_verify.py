# -*- coding: utf-8 -*-
"""
前置数据修复后的复核：把 v2 相对 v1 的提升做【逐项归因】，确认到底是谁在起作用。

最关键的质疑必须排除：v2 把成交额门槛从 4 亿降到 2 亿，等于从大盘切向中小盘。
如果提升主要来自「放低流动性门槛」，那和「收紧箱宽」没关系，结论就站不住。
做法：从 v1 出发，每次只改一项，看增量；并给出「保留 4 亿门槛」的 v2 变体做对照。
同时用 F2 修复后的相对流动性口径（当日横截面分位数）重跑一遍。
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
    p = os.path.join(CACHE, "events_fixed.pkl")
    if not os.path.exists(p):
        raise SystemExit("请先运行 yh_fix_data.py")
    ev = pd.read_pickle(p)
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    return ev


def M(s, prox, w, amt=None, amt_pct=None, t0=0.0, t1=6.0):
    m = (s["prox"] >= prox) & (s["width30"] <= w) & (s["turn30"] >= t0) & (s["turn30"] <= t1)
    if amt is not None:
        m = m & (s["amt30"] >= amt)
    if amt_pct is not None:
        m = m & (s["amt_pct"] >= amt_pct)
    return m


# 阶梯：从 v1 出发逐项改 (prox, 箱宽, 绝对额, 换手下, 换手上)
LADDER = [
    ("0. v1 原版                    ", 0.90, 0.25, 4e8, 1.5, 10.0),
    ("1. 只改箱宽→15%               ", 0.90, 0.15, 4e8, 1.5, 10.0),
    ("2. 只加换手天花板→≤6%         ", 0.90, 0.25, 4e8, 1.5, 6.0),
    ("3. 只取消换手下限定           ", 0.90, 0.25, 4e8, 0.0, 10.0),
    ("4. 只放低成交额→2亿           ", 0.90, 0.25, 2e8, 1.5, 10.0),
    ("5. 箱宽15% + 换手≤6%          ", 0.90, 0.15, 4e8, 1.5, 6.0),
    ("6. v2 完整(含放低成交额)      ", 0.90, 0.15, 2e8, 0.0, 6.0),
    ("7. v2 但【保留4亿】←关键对照  ", 0.90, 0.15, 4e8, 0.0, 6.0),
    ("8. v2 但【保留4亿+换手下限定】", 0.90, 0.15, 4e8, 1.5, 6.0),
]


def show(ev, use_pct=False):
    tag = "相对流动性口径(amt_pct)" if use_pct else "绝对成交额口径"
    print("=" * 128)
    print(f"Z1. 逐项归因阶梯 —— {tag}")
    print("=" * 128)
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows = []
        for nm, pr, w, a, t0, t1 in LADDER:
            if use_pct:
                ap = 0.85 if a >= 4e8 else 0.75       # 4亿≈前15%，2亿≈前25%
                m = M(s, pr, w, amt_pct=ap, t0=t0, t1=t1)
            else:
                m = M(s, pr, w, amt=a, t0=t0, t1=t1)
            sub = s[m]
            if len(sub) < 200:
                rows.append({"改动": nm, "事件数": len(sub), "beta调整%": np.nan,
                             "日均票": np.nan, "大亏%": np.nan})
                continue
            v, lo, hi = boot_ci(sub, "ab_fwd60")
            rows.append({"改动": nm, "事件数": len(sub),
                         "beta调整%": round(v * 100, 2), "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                         "日均票": round(len(sub) / sub["date"].nunique(), 1),
                         "大亏%": round((sub["fwd60"] <= -0.20).mean() * 100, 1)})
        print(f"\n--- [{r}] ---")
        with pd.option_context("display.width", 260, "display.max_columns", 40):
            print(pd.DataFrame(rows).to_string(index=False))
    print()


def ladder2(ev):
    """用相对流动性口径，把「保留 4 亿」这一关键对照做三段全胜检验"""
    print("=" * 128)
    print("Z2. 关键对照：v2 若不放松成交额（仍用 4 亿 / 前15%），还跑不跑得赢 v1？")
    print("=" * 128)
    segs = [("全样本", None, None), ("2019-2022", None, "2022"), ("2023-2026", "2023", None)]
    for r in ["UP", "CHOP", "DOWN"]:
        rows = []
        base = {}
        for sn, y0, y1 in segs:
            s = ev[ev["regime"] == r]
            if y0:
                s = s[s["year"] >= y0]
            if y1:
                s = s[s["year"] <= y1]
            base[sn] = boot_ci(s[M(s, 0.90, 0.25, amt=4e8, t0=1.5, t1=10.0)], "ab_fwd60")[0] * 100
        cands = [
            ("v2-放低成交额 (2亿, 换手≤6)", 0.90, 0.15, 2e8, 0.0, 6.0),
            ("v2-保留4亿   (换手≤6)", 0.90, 0.15, 4e8, 0.0, 6.0),
            ("v2-保留4亿+换手下限定(1.5~6)", 0.90, 0.15, 4e8, 1.5, 6.0),
        ]
        for nm, pr, w, a, t0, t1 in cands:
            rec = {"口径": nm, "Δ全": np.nan, "Δ前": np.nan, "Δ后": np.nan}
            ok = True
            for sn, y0, y1 in segs:
                s = ev[ev["regime"] == r]
                if y0:
                    s = s[s["year"] >= y0]
                if y1:
                    s = s[s["year"] <= y1]
                v = boot_ci(s[M(s, pr, w, amt=a, t0=t0, t1=t1)], "ab_fwd60")[0] * 100
                d = v - base[sn]
                rec["Δ" + sn[:1]] = round(d, 2)
                if d <= 0:
                    ok = False
            rec["四段/三段全胜"] = "✔" if ok else "✘"
            rows.append(rec)
        print(f"\n--- [{r}] ---（数字为相对同期 v1 的增量，pp）")
        print(pd.DataFrame(rows)[["口径", "Δ全", "Δ前", "Δ后", "四段/三段全胜"]].to_string(index=False))
    print()


def main(ev):
    show(ev, use_pct=False)
    show(ev, use_pct=True)
    ladder2(ev)

    print("=" * 128)
    print("Z3. 纯「老股」子样本（剔除幸存者偏差嫌疑样本）重跑终值")
    print("=" * 128)
    old = ev[ev["is_old"] == 1]
    print(f"  老股子样本：{old['code'].nunique()} 只 / {len(old)} 事件 / {old['date'].min()}~{old['date'].max()}")
    rows = []
    for r in ["UP", "CHOP", "DOWN"]:
        s = old[old["regime"] == r]
        for nm, m in [
            ("v1 原版", M(s, 0.90, 0.25, 4e8, 1.5, 10.0)),
            ("v2(2亿,换手≤6)", M(s, 0.90, 0.15, 2e8, 0.0, 6.0)),
            ("v2(保留4亿)", M(s, 0.90, 0.15, 4e8, 0.0, 6.0)),
        ]:
            sub = s[m]
            if len(sub) < 200:
                continue
            v, lo, hi = boot_ci(sub, "ab_fwd60")
            rows.append({"周期": r, "口径": nm, "事件数": len(sub),
                         "beta调整%": round(v * 100, 2), "CI": f"[{lo*100:.1f},{hi*100:.1f}]",
                         "日均票": round(len(sub) / sub["date"].nunique(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()


if __name__ == "__main__":
    main(load())
