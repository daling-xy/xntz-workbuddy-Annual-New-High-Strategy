# -*- coding: utf-8 -*-
"""旧高/追涨票复核 · 2026-11-12

用户质疑：「为什么会有新产业和中炬高新」。
诊断发现两只票都是「一年前旧高点附近的收敛平台」(hi250_ago 242 / 138 日)，
且新产业近 20 日 +14.4%、近 60 日 +23.1%（急涨后高位整理）。

本脚本回答：在【v2.6 已含 F 横盘门】的前提下，
恢复 N（双顶门）／加 ret20／ret60／箱顶位置门，各自能剔掉多少、收益代价多大，
以及会不会误杀德尔 10-23 的理想形态。

底座 = prox>=0.88 & amt30>=2亿 & VCP(w10<=10% & ct<=0.45 & w30<=50%)
v2.6 = 底座 + R + C + H + F(base_days<=35)
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_blockboot_all import make_dmap, daily, block_boot, NB, CI   # noqa: E402
from yh_shape_precise import cond_R, cond_C, cond_H                  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "doubletop_recheck_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
BAD = -0.20
_lines: list[str] = []


def P(s: str = "") -> None:
    print(s)
    _lines.append(s)


def row(tag: str, d: pd.DataFrame) -> str:
    if len(d) == 0:
        return "  %-24s %7d %6.1f |      —        —        —      —     —" % (tag, 0, 0)
    return "  %-24s %7d %6.1f | %+7.2f%% %+8.2f%% %+7.2f%% %6.1f%% %5.2f" % (
        tag, len(d), d.groupby("date").size().mean(),
        d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        d["fwd60"].mean() * 100, (d["ab_fwd60"] < BAD).mean() * 100,
        d["beta"].median())


def main() -> None:
    t0 = time.time()
    P("=" * 112)
    P("旧高/追涨票复核 · 2026-11-12 · 底座 = prox>=0.88 & 额>=2亿 & VCP；v2.6 = 底座 + R/C/H/F")
    P("=" * 112)

    ev = pickle.load(open(os.path.join(CACHE, "events_fixed.pkl"), "rb"))
    ev["date"] = ev["date"].astype(str).str[:10]
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    P("事件表 %,d 行".replace("%,d", f"{len(ev):,}"))
    dmap = make_dmap(ev)          # 必须用完整交易日历对齐 60 日区块

    st = pickle.load(open(os.path.join(CACHE, "flat_struct.pkl"), "rb"))
    st["date"] = st["date"].astype(str).str[:10]
    st["code"] = st["code"].astype(str).str.zfill(6)
    P("结构表 %,d 行（缓存）".replace("%,d", f"{len(st):,}"))

    # 先筛底座再 merge，控制内存
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]
    base = ((ev["prox_now"] >= 0.88) & (ev["amt30"] >= AMT)
            & (ev["width10"] <= 0.10) & (ev["contraction"] <= 0.45)
            & (ev["width30"] <= 0.50))
    ev = ev[base].copy()
    P("底座 %s 行" % f"{len(ev):,}")

    ev = ev.merge(st, on=["code", "date"], how="left")
    P("合并后 %s 行（耗时 %.0fs）" % (f"{len(ev):,}", time.time() - t0))
    if "w20" in ev.columns and "w20_y" in ev.columns:
        ev["w20"] = ev["w20_y"]

    R = cond_R(ev).fillna(False)
    C = cond_C(ev).fillna(False)
    H = cond_H(ev).fillna(False)
    F = (ev["base_days"] <= 35).fillna(False)
    A = R & C & H & F

    N = (ev["ago250"] <= 60).fillna(False)
    r20 = (ev["ret20"] <= 0.10).fillna(False)
    r60 = (ev["ret60"] <= 0.30).fillna(False)
    top = ((ev["close"] / ev["hi30"]) >= 0.95).fillna(False)

    P("")
    P("① 各配置四口径（A = 现 v2.6）")
    P("")
    P("  %-24s %7s %6s | %8s %8s %8s %6s %5s" % (
        "配置", "n", "日均", "ab_fwd60", "ex_fwd60", "fwd60", "大亏率", "β"))
    P("  " + "-" * 100)
    arms = {
        "A  v2.6 (R/C/H/F)": A,
        "B  +N 年高<=60日": A & N,
        "C  +ret20<=10%": A & r20,
        "D  +ret60<=30%": A & r60,
        "E  +收盘距30日高>=-5%": A & top,
        "F  +N +ret20": A & N & r20,
        "G  +N +ret60": A & N & r60,
    }
    for tag, m in arms.items():
        P(row(tag, ev[m]))

    P("")
    P("② 单门命中率（在 A 之上，被各门剔掉的比例）")
    P("")
    for nm, m in (("N 年高>60日", ~N), ("ret20>10%", ~r20),
                  ("ret60>30%", ~r60), ("收盘距30日高<-5%", ~top)):
        P("  %-24s 剔 %6d / %6d = %5.1f%%" % (
            nm, int((A & m).sum()), int(A.sum()), (A & m).sum() / max(A.sum(), 1) * 100))

    P("")
    P("③ 配对区块 bootstrap（60日区块 / %d次 / %.0f%%CI）—— 变体 − A，正数=变体更好" % (NB, CI * 100))
    ref = daily(ev[A])
    for tag, m in list(arms.items())[1:]:
        cur = daily(ev[m])
        diff = (cur - ref.reindex(cur.index)).dropna()
        if len(diff) < 3:
            P("  %-24s 共同交易日不足" % tag)
            continue
        est, lo, hi, nbk = block_boot(diff, dmap, NB, CI)
        sig = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
        P("  %-24s %+7.2fpp  90%%CI [%+6.2f, %+6.2f]  %s  (%d 区块)" % (
            tag, est * 100, lo * 100, hi * 100, sig, nbk))

    P("")
    P("④ 逐年一致性（每年日均 ab_fwd60，单位 %）")
    P("")
    yr = pd.to_datetime(ev["date"]).dt.year
    P("  %-8s %10s %10s %10s %10s" % ("年份", "A v2.6", "B +N", "C +ret20", "D +ret60"))
    for y in sorted(yr.unique()):
        vals = [ev[m & (yr == y)]["ab_fwd60"].mean() * 100 for m in (A, A & N, A & r20, A & r60)]
        P("  %-8s %+9.2f%% %+9.2f%% %+9.2f%% %+9.2f%%" % (y, *vals))

    P("")
    P("⑤ 德尔股份 10-23（理想形态）会被各门怎样处理 —— 实测值")
    P("")
    P("  prox 0.901 | hi250_ago 14日 | ret10 -3.7% | ret20 +14.2% | ret60 +62.2%")
    P("  收盘距30日高 -9.9% | base_days 19 | w10 9.3% | ct 0.21")
    P("  → N 通过（14<=60）；ret20>10% 被剔；ret60>30% 被剔；收盘距箱顶<-5% 被剔")
    P("  → 也就是说：用「20/60日涨幅」或「必须贴近箱顶」去剔新产业/中炬，会先误杀德尔。")

    P("")
    P("⑥ 11-12 主池 5 只逐票（哪些门会命中）")
    P("")
    P("  伊利股份   hi250_ago 8 / ret20 +10.3% / ret60 +19.6% / 距箱顶 -1.5%  → ret20 门会误伤")
    P("  惠泰医疗   hi250_ago 248 / ret20 +7.6% / ret60 +28.9% / 距箱顶 -2.4%  → N 门会剔")
    P("  新产业     hi250_ago 242 / ret20 +14.4% / ret60 +23.1% / 距箱顶 -7.7%  → N、ret20、箱顶门都剔")
    P("  金石亚药   hi250_ago 25 / ret20 -4.9% / ret60 +5.3% / 距箱顶 -6.8%  → 仅箱顶门会剔")
    P("  中炬高新   hi250_ago 138 / ret20 +5.6% / ret60 -1.4% / 距箱顶 -6.5%  → N、箱顶门会剔")

    P("")
    P("耗时 %.0fs" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入", OUT)


if __name__ == "__main__":
    main()
