# -*- coding: utf-8 -*-
"""
周期分层校准 IV（终局）：稳住结论
  第三段暴露了两个问题：
   1) 「去追高+控换手」在 2019-2022 与 2023-2026 两半样本里结论相反 → 不可靠，弃用
   2) UP 周期里 prox 越高越好（0.95~1.00 段 beta 调整 +4.06%），DOWN 周期里没有任何档位有正 alpha
本段做两件事：
   A) 对每一个候选参数集强制做 前后半样本 稳健性检验，单样本不稳的一律淘汰
   B) 量化「分周期 vs 不分周期」到底值不值
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


# ---- 候选参数集（全程只用「入场时可知」的量：prox/width/amt/turn） ----
CAND = {
    "DOWN": {
        "D0 v1 prox>=0.90 W25 A4": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8),
        "D1 prox>=0.85 W25 A2": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8),
        "D2 D1+换手上限定1.5%": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 1.5),
        "D3 D1+换手上限定3.0%": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 3.0),
        "D4 D1+换手上限定6.0%": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
        "D5 prox0.85~0.95 W25 A2 T<=6": lambda s: (s["prox"] >= 0.85) & (s["prox"] < 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
        "D6 严箱体15% A2 T<=6": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
    },
    "UP": {
        "U0 v1 prox>=0.90 W25 A4": lambda s: (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8),
        "U1 prox>=0.95 W25 A4": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8),
        "U2 prox>=0.95 W25 A2": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8),
        "U3 U2+换手上限定6.0%": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
        "U4 prox>=0.85 W25 A2 T<=6": lambda s: (s["prox"] >= 0.85) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
        "U5 prox>=0.95 W15 A2 T<=6": lambda s: (s["prox"] >= 0.95) & (s["width30"] <= 0.15) & (s["amt30"] >= 2e8) & (s["turn30"] <= 6.0),
    },
}


def main(ev):
    print("=" * 124)
    print("K. 稳健性筛选：每个候选参数集必须同时通过【全样本 / 2019-2022 / 2023-2026】三段检验")
    print("=" * 124)
    verdict = {}
    for r, cands in CAND.items():
        for nm, fn in cands.items():
            a = ev[ev["regime"] == r]
            h1 = ev[(ev["regime"] == r) & (ev["year"] <= "2022")]
            h2 = ev[(ev["regime"] == r) & (ev["year"] >= "2023")]
            rows = [row(f"[{r}] {nm} 全样本", a[fn(a)])]
            if len(h1) > 500:
                rows.append(row(f"[{r}] {nm} 2019-2022", h1[fn(h1)]))
            if len(h2) > 500:
                rows.append(row(f"[{r}] {nm} 2023-2026", h2[fn(h2)]))
            show(rows, f"[{r}] {nm}")
            va, la, ha = boot_ci(a[fn(a)], "ab_fwd60")
            v1, l1, h1c = boot_ci(h1[fn(h1)], "ab_fwd60")
            v2, l2, h2c = boot_ci(h2[fn(h2)], "ab_fwd60")
            ok = (np.sign(v1) == np.sign(v2)) and (np.sign(va) == np.sign(v2))
            verdict[(r, nm)] = (va * 100, v1 * 100, v2 * 100, "✔稳定" if ok else "✘翻转")
            print(f"    判决：全样本 {va*100:+.2f}% | 前半 {v1*100:+.2f}% | 后半 {v2*100:+.2f}%  → {'两段同号（稳定）' if ok else '两段异号（不稳定，弃用）'}\n")

    print("=" * 124)
    print("L. 最终裁决表")
    print("=" * 124)
    t = pd.DataFrame([{"周期": k[0], "参数集": k[1], "全样本beta调整%": round(v[0], 2),
                       "2019-2022%": round(v[1], 2), "2023-2026%": round(v[2], 2), "判决": v[3]}
                      for k, v in verdict.items()])
    print(t.to_string(index=False))
    print()

    print("=" * 124)
    print("M. 值不值？分周期 vs 不分周期")
    print("=" * 124)
    # 用通过稳健性检验的参数集构建「分周期政策」，与「全局单一政策」对比
    policy = {"UP": "U3 U2+换手上限定6.0%", "DOWN": "D4 D1+换手上限定6.0%", "CHOP": None}
    single = "U0 v1 prox>=0.90 W25 A4"

    def eval_policy(ev, pick, label):
        rows = []
        for r in ["UP", "CHOP", "DOWN"]:
            s = ev[ev["regime"] == r]
            if isinstance(pick, dict):
                nm = pick.get(r)
                if nm is None:
                    continue
                sub = s[CAND[r][nm](s)]
            else:
                fn = CAND["UP"][pick]
                # 同一套口径套到三个周期
                sub = s[(s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8)]
                nm = pick
            if len(sub) < 100:
                continue
            e, lo, hi = boot_ci(sub, "ex_fwd60")
            ae, alo, ahi = boot_ci(sub, "ab_fwd60")
            rows.append({"政策": label, "周期": r, "参数集": nm, "日均票": round(len(sub) / sub["date"].nunique(), 1),
                         "去均值超额%": round(e * 100, 2), "beta调整超额%": round(ae * 100, 2),
                         "beta调整CI": f"[{alo*100:.1f},{ahi*100:.1f}]",
                         "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
                         "大亏<=-20%": round((sub["fwd60"] <= -0.20).mean() * 100, 1)})
        return rows

    rows = eval_policy(ev, single, "全局单一参数(v1原版)")
    rows += eval_policy(ev, {"UP": "U3 U2+换手上限定6.0%", "DOWN": "D4 D1+换手上限定6.0%"}, "分周期参数")
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=" * 124)
    print("N. 换手率天花板是否有效？——不设/设6%/设3% 的三周期对照（这是唯一跨周期同号的修项）")
    print("=" * 124)
    for r in ["UP", "CHOP", "DOWN"]:
        a = ev[ev["regime"] == r]
        b = (a["prox"] >= 0.85) & (a["width30"] <= 0.25) & (a["amt30"] >= 2e8)
        rows = []
        for nm, extra in [("不设天花板", pd.Series(True, index=a.index)),
                          ("<=10%(排最疯的)", a["turn30"] <= 10),
                          ("<=6%", a["turn30"] <= 6),
                          ("<=3%", a["turn30"] <= 3),
                          ("1.5~6%(活跃但不疯)", (a["turn30"] >= 1.5) & (a["turn30"] <= 6))]:
            rows.append(row(f"[{r}] {nm}", a[b & extra]))
        show(rows, f"[{r}] 换手率天花板")


if __name__ == "__main__":
    main(load())
