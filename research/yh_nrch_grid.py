# -*- coding: utf-8 -*-
"""N/R/C/H 取舍矩阵 · 2026-11-12

用户指令：
  「感觉可以删 R，双顶考虑下，R 和 H 不删」
  → 最终口径：R、H 保留；N（防双顶）单独评估；C（逐级收缩）倾向删除。

本脚本把 N/R/C/H 的 16 种组合全跑一遍（在 prox≥0.88 & amt≥2亿 & VCP 之上），
用四口径 + 德尔命中率双重标尺选配置。

判据：
  收益质量  ab_fwd60 / 大亏率 / β
  票量      日均只数（太少没法用，太多不像 VCP）
  match     德尔平台段（2026-10-13~10-23，9 个交易日）命中几天
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_relax4 import load_events, _lines, P                       # noqa: E402
from yh_blockboot_all import make_dmap                             # noqa: E402
from yh_shape_precise import build_structure, cond_N, cond_R, cond_C, cond_H  # noqa: E402
from yh_factors import features_at                                  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "nrch_grid_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
BAD = -0.20


def delta_row(tag: str, d: pd.DataFrame) -> str:
    n = len(d)
    if n == 0:
        return f"  {tag:<18} {0:>7} {'—':>6} | {'—':>8} {'—':>8} {'—':>8} {'—':>6} {'—':>5}"
    return "  %-18s %7d %6.1f | %+7.2f%% %+7.2f%% %+7.2f%% %5.1f%% %5.2f" % (
        tag, n, d.groupby("date").size().mean(),
        d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        d["fwd60"].mean() * 100, (d["ab_fwd60"] < BAD).mean() * 100,
        d["beta"].median())


def ddl_hits(tag: str) -> str:
    """德尔平台段逐日命中（prox/w10/ct/w30/amt 恒过；只差 N/R/C/H 的判断）。"""
    k = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))
    b = k.get("300473")
    if not b:
        return f"  {tag:<18} 无缓存"
    cl = np.array([x["close"] for x in b], float)
    prm = {"high_days": 250, "box_days": 30, "tight_days": 10}
    days, mark = 0, []
    for i in range(len(b) - 30, len(b)):
        ds = str(b[i]["date"])
        if not ("2026-10-13" <= ds <= "2026-10-23"):
            continue
        f = features_at(b, prm, idx=i)
        if not f:
            continue
        sub = cl[:i + 1]
        w20 = sub[-20:].max() / sub[-20:].min() - 1
        ago = int(np.argmax(sub[-250:][::-1])) + 1
        hi10, hi30 = sub[-10:].max(), sub[-30:].max()
        rec = {
            "N": ago <= 60,
            "R": (sub[-1] / sub[-11] - 1) <= 0.08,
            "C": ((f["width10"] <= w20 * 0.70 and w20 <= f["width30"] * 0.90)
                  or f["width10"] <= 0.04),
            "H": hi10 >= hi30 * 0.95,
        }
        days += 1
        mark.append("".join(kk if rec[kk] else "·" for kk in "NRCH"))
    return "  %-18s 平台 %d 日，逐日 NRCH：%s" % (tag, days, " ".join(mark))


def main() -> None:
    t0 = time.time()
    P("=" * 108)
    P("N/R/C/H 取舍矩阵 · 2026-11-12 · base = prox≥0.88 & amt≥2亿 & VCP")
    P("=" * 108)

    ev = load_events()
    dmap = make_dmap(ev)
    st = build_structure()
    st["date"] = pd.to_datetime(st["date"])
    ev = ev.merge(st, on=["code", "date"], how="left")
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]

    base = (ev["prox_now"] >= 0.88) & (ev["amt30"] >= AMT) & \
           (ev["width10"] <= 0.10) & (ev["contraction"] <= 0.45) & (ev["width30"] <= 0.50)
    conds = {"N": cond_N(ev).fillna(False), "R": cond_R(ev).fillna(False),
             "C": cond_C(ev).fillna(False), "H": cond_H(ev).fillna(False)}

    P("  %-18s %7s %6s | %8s %8s %8s %6s %5s" % (
        "配置", "n", "日均", "ab_fwd60", "ex_fwd60", "fwd60", "大亏率", "β"))
    P("  " + "-" * 104)
    rows = []
    for k in range(0, 5):
        for combo in itertools.combinations("NRCH", k):
            tag = "+".join(combo) if combo else "无附加(v2.2)"
            m = base.copy()
            for c in combo:
                m &= conds[c]
            d = ev[m]
            P(delta_row(tag, d))
            rows.append({"cfg": tag, "n": len(d),
                         "daily": d.groupby("date").size().mean() if len(d) else 0,
                         "ab": d["ab_fwd60"].mean(), "bad": (d["ab_fwd60"] < BAD).mean(),
                         "beta": d["beta"].median()})
        P("")

    P("=" * 108)
    P("德尔股份平台段（2026-10-13 ~ 10-23）逐日 N/R/C/H 判定")
    P("=" * 108)
    P(ddl_hits("德尔"))

    # ---- 追加：用趋势前提 T（站上 120 日线）替代 N 来杀「下跌中继」 ----
    #   背景：删 N 后中炬高新、九州通这类「下跌中的收敛平台」会回到池里（用户曾点名质疑）。
    #   T 与 T2（均线多头）不同：德尔平台期 close(34.5) > ma120(约 28)，T 不会误杀。
    P("")
    P("=" * 108)
    P("补充：T 趋势前提（close > ma120）能否替代 N 杀下跌中继")
    P("=" * 108)
    tend = (ev["close"] > ev["ma120"]).fillna(False)
    extra = {
        "R+C+H": conds["R"] & conds["C"] & conds["H"],
        "R+C+H+T": conds["R"] & conds["C"] & conds["H"] & tend,
        "N+R+C+H": conds["N"] & conds["R"] & conds["C"] & conds["H"],
        "N+R+C+H+T": conds["N"] & conds["R"] & conds["C"] & conds["H"] & tend,
    }
    for tag, m in extra.items():
        d = ev[base & m]
        P(delta_row(tag, d))
        rows.append({"cfg": tag, "n": len(d),
                     "daily": d.groupby("date").size().mean() if len(d) else 0,
                     "ab": d["ab_fwd60"].mean(), "bad": (d["ab_fwd60"] < BAD).mean(),
                     "beta": d["beta"].median()})

    # ---- 追加：N 阈值扫描（R+C+H 固定）----
    #   用户要「双顶考虑下」：60 日太严（砍票 33%），全删又放进一年前同价位的双顶。
    P("")
    P("=" * 108)
    P("补充：N 阈值扫描（R+C+H 固定，看「新高新鲜度」放宽到多少天最划算）")
    P("=" * 108)
    rch = conds["R"] & conds["C"] & conds["H"]
    for thr in (40, 60, 90, 120, 150, 200, 250, None):
        m = rch.copy()
        if thr is not None:
            m &= (ev["ago250"] <= thr).fillna(False)
        d = ev[base & m]
        P(delta_row("N<=%s" % (thr if thr else "不限"), d))

    P("")
    P("=" * 108)
    P("按「ab 高 / 大亏低 / 票量够」排序（日均 >= 6 只）")
    P("=" * 108)
    df = pd.DataFrame(rows)
    df = df[df["daily"] >= 6].copy()
    df["score"] = df["ab"] * 100 - df["bad"] * 20
    df = df.sort_values("score", ascending=False)
    P("  %-18s %6s %6s %8s %8s" % ("配置", "日均", "ab", "大亏", "β"))
    for _, r in df.head(10).iterrows():
        P("  %-18s %6.1f %+7.2f%% %7.1f%% %6.2f" % (
            r["cfg"], r["daily"], r["ab"] * 100, r["bad"] * 100, r["beta"]))

    P("")
    P("耗时 %.1f 秒" % (time.time() - t0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("已写", OUT)


if __name__ == "__main__":
    main()
