# -*- coding: utf-8 -*-
"""横盘持续天数（base_days）探针

用户诉求（2026-11-12）：
  「我不是想捕捉 横盘→深坑→修复→横盘，只是德尔恰好长这样。
    主要是横盘 20-30 天 + 满足其他条件。现在主池条件里没有 20-30 天横盘的要求？」

核心问题：现有 width10 / width20 / width30 只度量「最近 k 天有多窄」，
不度量「已经窄了多久」。一只票可以横 3 天（w10 很窄）也能过门。

base_days 定义：最大的 k（5~60），使得过去 k 个交易日收盘价的
  (max/min - 1) <= thr
即「价格已经在 thr 宽度以内的箱体里待了 k 个交易日」。
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from yh_screen import PARAMS  # noqa: E402


def base_days(closes: np.ndarray, i: int, thr: float = 0.20,
              kmin: int = 5, kmax: int = 60) -> int:
    """过去 k 日收盘区间宽度 <= thr 的最大 k。宽度随 k 单调不减，可提前 break。"""
    best = 0
    for k in range(kmin, kmax + 1):
        if i - k + 1 < 0:
            break
        seg = closes[i - k + 1:i + 1]
        if seg.max() / seg.min() - 1.0 <= thr:
            best = k
        else:
            break
    return best


def main() -> None:
    k = pickle.load(open("cache/kline_320.pkl", "rb"))
    pool = pd.read_csv("output/v2/候选池_v2.csv")
    sat = pd.read_csv("output/v2/坑后平台池_v2.csv")
    names = dict(zip(pool["code"].astype(str).str.zfill(6), pool["name"]))
    names.update(dict(zip(sat["code"].astype(str).str.zfill(6), sat["name"])))

    # ---------- 1. 全市场 base_days 分布（11-11 收盘） ----------
    print("=" * 100)
    print(f"① 全市场 base_days 分布（thr=20pct，11-11 收盘，{len(k)} 只）")
    print("=" * 100)
    vals = []
    for code, bars in k.items():
        cl = np.array([b["close"] for b in bars], float)
        if len(cl) < 70:
            continue
        vals.append(base_days(cl, len(cl) - 1, 0.20))
    s = pd.Series(vals)
    print(f"  中位 {s.median():.0f} 日 | 均值 {s.mean():.1f} 日")
    bins = [0, 5, 10, 15, 20, 25, 30, 40, 60]
    lab = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-39", "40-59"]
    cut = pd.cut(s, bins=[-1] + bins[1:], labels=lab)
    dist = cut.value_counts().reindex(lab)
    for lb, n in dist.items():
        bar = "█" * int(n / max(dist.max(), 1) * 40)
        print(f"  {lb:<7s} {n:5d}  {n/len(s)*100:5.1f}% {bar}")
    print(f"  其中 base_days >= 20 的占比 {(s>=20).mean()*100:.1f}%")

    # ---------- 2. 主池 / 卫星池逐票 base_days ----------
    print()
    print("=" * 100)
    print("② 当前主池 16 只 + 卫星池 8 只的 base_days（11-11）")
    print("=" * 100)
    rows = []
    for tag, df in (("主池", pool), ("卫星", sat)):
        for _, r in df.iterrows():
            code = str(r["code"]).zfill(6)
            bars = k.get(code)
            if not bars:
                continue
            cl = np.array([b["close"] for b in bars], float)
            rows.append({
                "池": tag, "代码": code, "名称": r["name"],
                "b15": base_days(cl, len(cl) - 1, 0.15),
                "b20": base_days(cl, len(cl) - 1, 0.20),
                "b25": base_days(cl, len(cl) - 1, 0.25),
                "w10": r.get("width10"), "w30": r.get("width30"),
                "ct": r.get("contraction"),
            })
    out = pd.DataFrame(rows).sort_values(["池", "b20"], ascending=[True, False])
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for tag in ("主池", "卫星"):
        sub = out[out["池"] == tag]
        print(f"\n  {tag}: b20 中位 {sub['b20'].median():.0f} 日 | "
              f">=20 日的有 {int((sub['b20'] >= 20).sum())}/{len(sub)} 只 | "
              f">=15 日的有 {int((sub['b20'] >= 15).sum())}/{len(sub)} 只")

    # ---------- 3. 德尔平台期逐日 base_days ----------
    print()
    print("=" * 100)
    print("③ 德尔股份 2026-09-01 ~ 11-11 逐日 base_days（看平台期到底是多少天）")
    print("=" * 100)
    bars = k.get("300473")
    if bars:
        cl = np.array([b["close"] for b in bars], float)
        rec = []
        for i in range(len(cl) - 55, len(cl)):
            rec.append({
                "date": bars[i]["date"], "close": round(cl[i], 2),
                "b15": base_days(cl, i, 0.15),
                "b20": base_days(cl, i, 0.20),
                "b25": base_days(cl, i, 0.25),
            })
        print(pd.DataFrame(rec).to_string(index=False))
    else:
        print("  300473 不在 kline_320")


if __name__ == "__main__":
    main()
