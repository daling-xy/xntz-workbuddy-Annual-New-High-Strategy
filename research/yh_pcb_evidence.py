# -*- coding: utf-8 -*-
"""坑后平台（PCB）全样本实证 · 2026-11-12

用户口径：
  「德尔股份 2025-05 ~ 2026-06 一直横盘震荡，2026-06 ~ 2026-09 挖了个大坑，
    然后震荡 20-30 天之后上涨。不用看到那么前面，震荡 20-30 天这里就足够选出来。
    前面长期震荡、即使有个大坑（比较快就修复），也算加分项。
    NRCH 也不是一定要保留，主要选出来的票要合理。」

本脚本回答三个问题：
  Q1  PCB 单独作为形态门，收益是否站得住？（全样本 150 万事件）
  Q2  PCB 与现 VCP 门是「替代」还是「互补」？（并集 vs 交集）
  Q3  去掉 N/C/H、只留 R（不追高），代价多大？

对比口径（同一 base：prox_now≥0.88 & amt30≥2亿）：
  B1  v2.2  VCP：w10≤10% & ct≤0.45 & w30≤50%
  B2  v2.4  VCP + N/R/C/H（现版本）
  P1  PCB 单独
  P2  VCP ∪ PCB（并集，形态二选一）
  P3  (VCP ∪ PCB) + R（只保留「不追高」这一条）

口径：ab_fwd60（β中性 alpha）/ ex_fwd60 / fwd60 / 大亏率(ab<-20%) / β / 逐年
      配对区块 bootstrap（60 日区块 / 4000 次 / 90%CI）
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

from yh_relax4 import load_events, _lines, P            # noqa: E402
from yh_blockboot_all import make_dmap, add_bi, daily, block_boot, HB  # noqa: E402
from yh_shape_precise import build_structure, cond_N, cond_R, cond_C, cond_H, _and_all  # noqa: E402
from yh_pcb_probe import pcb_at                          # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "pcb_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
BAD = -0.20
NB = 4000
CI = 0.90


def build_pcb(ev: pd.DataFrame) -> pd.DataFrame:
    """逐股逐事件日计算 PCB 命中与特征，返回 (code,date,pcb_*) 表。"""
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    need = ev[["code", "date"]].copy()
    need["date"] = need["date"].astype(str)
    grp = need.groupby("code")["date"].apply(list).to_dict()

    rows = []
    miss = 0
    for code, ds_list in grp.items():
        b = k.get(code)
        if not b or len(b) < 200:
            miss += 1
            continue
        cl = np.array([x["close"] for x in b], float)
        if not np.isfinite(cl).all() or (cl <= 0).any():
            miss += 1
            continue
        idx = {str(x["date"]): i for i, x in enumerate(b)}
        for ds in ds_list:
            i = idx.get(ds)
            if i is None or i < 60:
                continue
            r = pcb_at(cl, i)
            if r:
                rows.append({"code": code, "date": ds, **r})
    P("  PCB 计算：%d 只股票，缺失/跳过 %d 只，命中事件 %s 条" % (
        len(grp), miss, f"{len(rows):,}"))
    return pd.DataFrame(rows)


def four_way(tag: str, d: pd.DataFrame) -> None:
    P("  %-26s %7d %6.2f | %+7.2f%% %+7.2f%% %+7.2f%% %6.1f%% %5.2f" % (
        tag, len(d), d.groupby("date").size().mean() if len(d) else 0,
        d["ab_fwd60"].mean() * 100 if len(d) else np.nan,
        d["ex_fwd60"].mean() * 100 if len(d) else np.nan,
        d["fwd60"].mean() * 100 if len(d) else np.nan,
        (d["ab_fwd60"] < BAD).mean() * 100 if len(d) else np.nan,
        d["beta"].median() if len(d) else np.nan))


def paired_boot(tag: str, poolA: pd.DataFrame, poolB: pd.DataFrame, dmap: dict) -> None:
    da = daily(poolA)
    db = daily(poolB)
    diff = (da - db.reindex(da.index)).dropna()
    if len(diff) < 3:
        P("  %-34s 共同交易日不足，跳过" % tag)
        return
    est, lo, hi, nb = block_boot(diff, dmap, NB, CI)
    sig = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
    P("  %-34s %+7.2fpp  90%%CI [%+6.2f, %+6.2f]  %s  (%d 区块)" % (
        tag, est * 100, lo * 100, hi * 100, sig, nb))


def main() -> None:
    t0 = time.time()
    P("=" * 108)
    P("坑后平台（PCB）全样本实证 · 2026-11-12")
    P("=" * 108)

    ev = load_events()
    dmap = make_dmap(ev)
    P("事件表：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min(), ev["date"].max()))

    st = build_structure()
    st["date"] = pd.to_datetime(st["date"])
    P("结构表：%s 行" % f"{len(st):,}")
    ev = ev.merge(st, on=["code", "date"], how="left")
    # 事件表只有旧口径 prox（30日高/年高），v2 口径的 prox_now 需要从 dd_from_high 还原
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]

    pcb = build_pcb(ev)
    if len(pcb):
        pcb["date"] = pd.to_datetime(pcb["date"])
        ev = ev.merge(pcb, on=["code", "date"], how="left")
    for c in ("pcb_hit", "pcb_dd", "pcb_days", "pcb_plat_w", "pcb_box_w", "pcb_repair_spd"):
        if c not in ev.columns:
            ev[c] = np.nan
    ev["pcb_hit"] = ev["pcb_hit"].fillna(0).astype(int)

    base = (ev["prox_now"] >= 0.88) & (ev["amt30"] >= AMT)
    vcp = (ev["width10"] <= 0.10) & (ev["contraction"] <= 0.45) & (ev["width30"] <= 0.50)
    nrch = _and_all(ev, [cond_N, cond_R, cond_C, cond_H])
    pcb_m = ev["pcb_hit"] == 1

    tend = ev["close"] > ev["ma120"]      # T 趋势前提：站上 120 日线
    rr = cond_R(ev).fillna(False)
    pools = {
        "B1 v2.2 VCP": ev[base & vcp],
        "B2 v2.4 VCP+NRCH": ev[base & vcp & nrch],
        "P1 PCB 单独": ev[base & pcb_m],
        "P4 PCB+T": ev[base & pcb_m & tend],
        "P5 PCB+T+R": ev[base & pcb_m & tend & rr],
        "P2 VCP∪PCB": ev[base & (vcp | pcb_m)],
        "P3 (VCP∪PCB)+R": ev[base & (vcp | pcb_m) & rr],
        "R1 底座(仅 prox+额)": ev[base],
    }

    P("")
    P("=" * 108)
    P("① 四口径（基准：base = prox_now≥0.88 且 amt30≥2亿）")
    P("=" * 108)
    P("  %-26s %7s %6s | %7s %7s %7s %6s %5s" % (
        "口径", "n", "日均", "ab_fwd60", "ex_fwd60", "fwd60", "大亏率", "β"))
    for tag, d in pools.items():
        four_way(tag, d)

    P("")
    P("=" * 108)
    P("② 配对区块 bootstrap（相对同一 base 的日均 ab 差）")
    P("=" * 108)
    for a, b, lab in [
        ("P1 PCB 单独", "R1 底座(仅 prox+额)", "PCB − 底座"),
        ("P4 PCB+T", "P1 PCB 单独", "加 T 趋势前提 − PCB"),
        ("P5 PCB+T+R", "B1 v2.2 VCP", "PCB+T+R − VCP"),
        ("P2 VCP∪PCB", "B1 v2.2 VCP", "并集 − VCP"),
        ("P3 (VCP∪PCB)+R", "B1 v2.2 VCP", "并集+R − VCP"),
        ("B2 v2.4 VCP+NRCH", "B1 v2.2 VCP", "NRCH − VCP（复核）"),
    ]:
        paired_boot(lab, pools[a], pools[b], dmap)

    P("")
    P("=" * 108)
    P("③ 逐年一致性（PCB 组 vs 底座组）")
    P("=" * 108)
    a = pools["P1 PCB 单独"].copy()
    b = pools["R1 底座(仅 prox+额)"].copy()
    a["y"], b["y"] = a["date"].dt.year, b["date"].dt.year
    P("  %-6s %6s %6s %10s %10s %10s %10s" % (
        "年", "nPCB", "n底座", "PCB(ab)", "底座(ab)", "差", "PCB大亏"))
    ok = tot = 0
    for y in sorted(set(a["y"]) & set(b["y"])):
        x, z = a[a["y"] == y], b[b["y"] == y]
        if len(x) < 20 or len(z) < 20:
            continue
        d = (x["ab_fwd60"].mean() - z["ab_fwd60"].mean()) * 100
        tot += 1
        ok += 1 if d > 0 else 0
        P("  %-6d %6d %6d %+9.2f%% %+9.2f%% %+9.2fpp %9.1f%%" % (
            y, len(x), len(z), x["ab_fwd60"].mean() * 100,
            z["ab_fwd60"].mean() * 100, d, (x["ab_fwd60"] < BAD).mean() * 100))
    P("  PCB 跑赢底座的年份：%d/%d" % (ok, tot))

    P("")
    P("=" * 108)
    P("④ 个案：德尔股份是否被 PCB 捕获（长历史，逐日）")
    P("=" * 108)
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    bl = k.get("300473")
    if bl:
        cl = np.array([x["close"] for x in bl], float)
        hits = []
        for i in range(len(bl) - 40, len(bl)):
            r = pcb_at(cl, i)
            if r:
                hits.append((bl[i]["date"], cl[i], r["pcb_days"], r["pcb_dd"], r["pcb_plat_w"]))
        P("  2026-09 之后命中 %d 天：" % len(hits))
        for ds, c, dy, dd, pw in hits:
            P("    %s  收盘 %6.2f  横盘 %2d 日  坑深 %5.1f%%  平台宽 %5.1f%%" % (
                ds, c, dy, dd * 100, pw * 100))
    P("")
    P("耗时 %.1f 秒" % (time.time() - t0))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("已写", OUT)


if __name__ == "__main__":
    main()
