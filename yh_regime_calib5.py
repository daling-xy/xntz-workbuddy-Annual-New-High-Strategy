# -*- coding: utf-8 -*-
"""
周期分层校准 V（收口）：给 CHOP 补同一套检验，并输出最终「分周期政策」vs「单一政策」。
第四段已确认通过前后半样本双重稳健性检验的两套：
   UP   : prox>=0.95 / width30<=0.15 / amt>=2e8 / turn<=6%   → beta调整 +2.97%（前半+1.60 后半+3.69）
   DOWN : prox>=0.85 / width30<=0.15 / amt>=2e8 / turn<=6%   → beta调整 +1.95%（前半+2.92 后半+1.37）
本段补齐 CHOP，并做最后的政策对比与「均值方差」视角。
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


def boot_ci(sub, key, n=BOOT):
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def row(label, sub):
    if sub is None or len(sub) == 0:
        return None
    e, lo, hi = boot_ci(sub, "ex_fwd60")
    ae, alo, ahi = boot_ci(sub, "ab_fwd60")
    return {
        "分组": label, "事件数": int(len(sub)), "交易日": int(sub["date"].nunique()),
        "日均票": round(len(sub) / max(sub["date"].nunique(), 1), 1),
        "去均值超额%": round(e * 100, 2), "90%CI": f"[{lo*100:.1f},{hi*100:.1f}]",
        "beta调整超额%": round(ae * 100, 2), "beta调整CI": f"[{alo*100:.1f},{ahi*100:.1f}]",
        "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
        "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
        "大亏<=-20%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
        "大赢>=+20%": round((sub["fwd60"] >= 0.20).mean() * 100, 1),
        "mae20中位%": round(sub["mae20"].median() * 100, 1),
    }


COLS = ["分组", "事件数", "交易日", "日均票", "去均值超额%", "90%CI",
        "beta调整超额%", "beta调整CI", "绝对收益%", "胜率%",
        "大亏<=-20%", "大赢>=+20%", "mae20中位%"]


def show(rows, title):
    rs = [r for r in rows if r]
    if not rs:
        print(f"--- {title} --- 无样本\n")
        return
    print(f"--- {title} ---")
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(pd.DataFrame(rs)[COLS].to_string(index=False))
    print()


def load():
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    return ev


BASE_W15 = lambda s: (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0)

CHOP_CAND = {
    "C0 v1 prox>=0.90 W25 A4": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8),
    "C1 prox>=0.85 W15 A2 T6": lambda s: (s["prox"] >= 0.85) & BASE_W15(s),
    "C2 prox>=0.90 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & BASE_W15(s),
    "C3 prox>=0.95 W15 A2 T6": lambda s: (s["prox"] >= 0.95) & BASE_W15(s),
    "C4 prox0.90~0.98 W15 A2 T6": lambda s: (s["prox"] >= 0.90) & (s["prox"] < 0.98) & BASE_W15(s),
}


def main(ev):
    print("=" * 124)
    print("O. CHOP 周期补检验（同一口径：严箱体15% + 流动性地板2亿 + 换手<=6%）")
    print("=" * 124)
    verdict = {}
    a = ev[ev["regime"] == "CHOP"]
    h1 = ev[(ev["regime"] == "CHOP") & (ev["year"] <= "2022")]
    h2 = ev[(ev["regime"] == "CHOP") & (ev["year"] >= "2023")]
    for nm, fn in CHOP_CAND.items():
        rows = [row(f"[CHOP] {nm} 全样本", a[fn(a)]),
                row(f"[CHOP] {nm} 2019-2022", h1[fn(h1)]),
                row(f"[CHOP] {nm} 2023-2026", h2[fn(h2)])]
        show(rows, f"[CHOP] {nm}")
        vals = [boot_ci(x[fn(x)], "ab_fwd60")[0] * 100 for x in (a, h1, h2)]
        ok = np.sign(vals[1]) == np.sign(vals[2])
        verdict[nm] = (*[round(v, 2) for v in vals], "✔稳定" if ok else "✘翻转")
        print(f"    判决：全样本 {vals[0]:+.2f}% | 前半 {vals[1]:+.2f}% | 后半 {vals[2]:+.2f}% → {'稳定' if ok else '不稳定'}\n")

    print("=" * 124)
    print("P. 最终推荐政策 vs 全局单一政策 vs 不做任何筛选")
    print("=" * 124)
    FINAL = {
        "UP": ("prox>=0.95 W15 A2 T6", lambda s: (s["prox"] >= 0.95) & BASE_W15(s)),
        "CHOP": ("prox>=0.90 W15 A2 T6", lambda s: (s["prox"] >= 0.90) & BASE_W15(s)),
        "DOWN": ("prox>=0.85 W15 A2 T6", lambda s: (s["prox"] >= 0.85) & BASE_W15(s)),
    }
    rows = []
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[(ev["regime"] == r)]
        nm, fn = FINAL[r]
        rows.append(row(f"【分周期政策】[{r}] {nm}", s[fn(s)]))
        # 单一政策：v1 原版
        rows.append(row(f"【全局单一】[{r}] v1 prox>=0.90 W25 A4",
                        s[(s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8)]))
    show(rows, "最终政策对照")

    print("=" * 124)
    print("Q. 组合层面的「按日持仓」收益：每个交易日等权买入当日命中票，看净值")
    print("=" * 124)
    for policy, nm in [(FINAL, "分周期政策"), (None, "全局v1")]:
        for r in ["UP", "CHOP", "DOWN"]:
            s = ev[ev["regime"] == r].copy()
            if policy is None:
                s["sel"] = ((s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8))
            else:
                s["sel"] = policy[r][1](s)
            g = s[s["sel"]].groupby("date").agg(n=("fwd60", "size"), ret=("fwd60", "mean"))
            if len(g) < 20:
                continue
            r60 = g["ret"]
            print(f"  {nm:8s} [{r:4s}] 交易日{len(g):5d}  日均票{g['n'].mean():6.1f}  "
                  f"多空无关的日均绝对收益 {r60.mean()*100:6.2f}%  "
                  f"日胜率 {(r60>0).mean()*100:5.1f}%  "
                  f"最差单日 {r60.min()*100:6.2f}%  最好单日 {r60.max()*100:6.2f}%")
    print()

    print("=" * 124)
    print("R. CHOP 裁决")
    print("=" * 124)
    print(pd.DataFrame([{"参数集": k, "全样本%": v[0], "2019-2022%": v[1], "2023-2026%": v[2], "判决": v[3]}
                        for k, v in verdict.items()]).to_string(index=False))


if __name__ == "__main__":
    main(load())
