# -*- coding: utf-8 -*-
"""横盘持续天数（base_days）实证 · 2026-11-12

背景：用户指出主池缺「横盘 20-30 天」这条要求。现有 width10/20/30 只度量
「最近 k 天有多窄」，不度量「已经窄了多久」——一只票只横 3 天也能过门。

base_days = 最大的 k（5~60），使过去 k 日收盘区间宽度 (max/min-1) <= 20%

本脚本回答三件事：
  ① base_days 在底座（prox≥0.88 & amt≥2亿）上是否单调有效 → 分箱四口径
  ② 在主池 v2.5（+R/C/H）上叠加 base_days 上下限，池子怎么变
  ③ 德尔平台段（10-13~10-23）在各阈值下能否命中
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

from yh_relax4 import load_events, _lines, P                        # noqa: E402
from yh_shape_precise import cond_R, cond_C, cond_H                 # noqa: E402
from yh_factors import features_at                                  # noqa: E402
from yh_blockboot_all import make_dmap, daily, block_boot, NB, CI   # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "flat_days_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
BAD = -0.20
THR = 0.20
KMIN, KMAX = 5, 60


def build_flat() -> pd.DataFrame:
    """从长历史重建 (code,date) -> base_days / w20 / hi10 / hi30 / ret10 / ago250。"""
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    from numpy.lib.stride_tricks import sliding_window_view
    frames = []
    for code, b in k.items():
        n = len(b)
        if n < 260:
            continue
        cl = np.array([x["close"] for x in b], float)
        if not np.isfinite(cl).all() or (cl <= 0).any():
            continue
        cs = pd.Series(cl)
        d = pd.DataFrame({"code": code, "date": [x["date"] for x in b]})

        # base_days：对 k=5..60 逐一遍历，保留最大满足 k（宽度随 k 单调增）
        bd = np.zeros(n, dtype=np.int16)
        for kk in range(KMIN, KMAX + 1):
            if kk > n:
                break
            w = (cs.rolling(kk).max() / cs.rolling(kk).min() - 1.0).values
            bd = np.where(np.nan_to_num(w, nan=9.9) <= THR, kk, bd)
        d["base_days"] = bd

        d["w20"] = (cs.rolling(20).max() / cs.rolling(20).min() - 1).values
        d["hi10"] = cs.rolling(10).max().values
        d["hi30"] = cs.rolling(30).max().values
        d["ret10"] = cl / np.concatenate([np.full(10, np.nan), cl[:-10]]) - 1
        w = sliding_window_view(cl, 250)
        ago = np.full(n, np.nan)
        ago[249:] = np.argmax(w[:, ::-1], axis=1) + 1
        d["ago250"] = ago
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def frow(tag: str, d: pd.DataFrame) -> str:
    n = len(d)
    if n == 0:
        return "  %-22s %7d %6s | %8s %8s %8s %7s %6s" % (tag, 0, "—", "—", "—", "—", "—", "—")
    return "  %-22s %7d %6.1f | %+7.2f%% %+7.2f%% %+7.2f%% %6.1f%% %6.2f" % (
        tag, n, d.groupby("date").size().mean(),
        d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        d["fwd60"].mean() * 100, (d["ab_fwd60"] < BAD).mean() * 100,
        d["beta"].median())


HDR = ("  %-22s %7s %6s | %9s %9s %9s %8s %7s" %
       ("口径", "事件数", "日均", "ab_fwd60", "ex_fwd60", "fwd60", "大亏率", "β"))
SEP = "  " + "-" * 94


def main() -> None:
    t0 = time.time()
    P("=" * 100)
    P("横盘持续天数 base_days 实证 · 2026-11-12")
    P("base_days = 过去 k 日收盘区间宽度 <= 20% 的最大 k（k 上限 60）")
    P("=" * 100)

    ev = load_events()
    ev["date"] = ev["date"].astype(str).str[:10]
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    P(f"事件表 {len(ev):,} 行")

    fcache = os.path.join(CACHE, "flat_struct.pkl")
    if os.path.exists(fcache):
        st = pickle.load(open(fcache, "rb"))
        P(f"结构表（缓存）{len(st):,} 行")
    else:
        st = build_flat()
        pickle.dump(st, open(fcache, "wb"))
        P(f"结构表 {len(st):,} 行（耗时 {time.time()-t0:.0f}s，已缓存）")

    st["date"] = st["date"].astype(str).str[:10]
    st["code"] = st["code"].astype(str).str.zfill(6)
    ev = ev.merge(st, on=["code", "date"], how="left")
    P(f"合并后缺失 base_days {ev['base_days'].isna().sum():,} 行")

    base = (ev["prox"] >= 0.88) & (ev["amt30"] >= AMT) & ev["base_days"].notna()
    vcp = base & (ev["width10"] <= 0.10) & (ev["contraction"] <= 0.45) & (ev["width30"] <= 0.50)
    rch = vcp & cond_R(ev).fillna(False) & cond_C(ev).fillna(False) & cond_H(ev).fillna(False)

    # ---------- ① 底座上的 base_days 分箱 ----------
    P("")
    P("① 底座（prox≥0.88 & 额≥2亿）上按 base_days 分箱 —— 看形态新鲜度是否影响收益")
    P(HDR)
    P(SEP)
    bins = [(0, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 39), (40, 59), (60, 60)]
    for lo, hi in bins:
        m = base & (ev["base_days"] >= lo) & (ev["base_days"] <= hi)
        P(frow(f"base_days {lo}-{hi}" if lo != hi else f"base_days {lo}+(满格)", ev[m]))
    P(SEP)
    P(frow("底座全体", ev[base]))

    # ---------- ② VCP 上的基线 ----------
    P("")
    P("② 在主池 v2.5 上叠加 base_days 下限 / 区间约束")
    P(HDR)
    P(SEP)
    P(frow("v2.5 现状（无天数）", ev[rch]))
    for lo in (10, 15, 20, 25):
        P(frow(f"v2.5 + bd>={lo}", ev[rch & (ev["base_days"] >= lo)]))
    P(SEP)
    for lo, hi in ((10, 40), (15, 35), (15, 40), (20, 40), (20, 35)):
        P(frow(f"v2.5 + bd {lo}~{hi}", ev[rch & (ev["base_days"] >= lo) & (ev["base_days"] <= hi)]))
    P(SEP)
    for hi in (35, 40, 45, 50):
        P(frow(f"v2.5 + bd<={hi}", ev[rch & (ev["base_days"] <= hi)]))
    P(SEP)
    for lo, hi in ((10, 25), (15, 30)):
        P(frow(f"VCP(不加RCH) + bd {lo}~{hi}",
               ev[vcp & (ev["base_days"] >= lo) & (ev["base_days"] <= hi)]))

    # ---------- ③ 德尔命中 ----------
    P("")
    P("③ 德尔股份平台段（2026-10-13 ~ 10-23）逐日 base_days 与各阈值命中")
    P("")
    k2 = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))
    b = k2.get("300473")
    if b:
        cl = np.array([x["close"] for x in b], float)
        prm = {"high_days": 250, "box_days": 30, "tight_days": 10}
        P("  %-12s %8s %7s %7s %7s | %s" % ("日期", "收盘", "b20", "w10", "prox", "bd>=15 / >=20 / 15~35"))
        rec = []
        for i in range(len(b) - 30, len(b)):
            ds = str(b[i]["date"])
            if not ("2026-10-13" <= ds <= "2026-10-23"):
                continue
            f = features_at(b, prm, idx=i)
            if not f:
                continue
            sub = cl[:i + 1]
            bd = 0
            for kk in range(KMIN, KMAX + 1):
                if kk > len(sub):
                    break
                seg = sub[-kk:]
                if seg.max() / seg.min() - 1 <= THR:
                    bd = kk
                else:
                    break
            rec.append((ds, sub[-1], bd, f["width10"],
                        sub[-1] / sub[-250:].max(),
                        bd >= 15, bd >= 20, 15 <= bd <= 35))
        for ds, clp, bd, w10, px, g15, g20, rg in rec:
            mk = lambda x: "✓" if x else "✗"  # noqa: E731
            P("  %-12s %8.2f %7d %6.1f%% %7.3f | %s %s %s" % (
                ds, clp, bd, w10 * 100, px, mk(g15), mk(g20), mk(rg)))

    # ---------- ④ 配对区块 bootstrap：加天数门是否显著 ----------
    P("")
    P("④ 配对区块 bootstrap（60日区块 / 4000次 / 90%CI）—— 差值是对数交易日日均 ab_fwd60 之差")
    dmap = make_dmap(ev)
    ref = daily(ev[rch])
    arms = {
        "v2.5 + bd>=15": rch & (ev["base_days"] >= 15),
        "v2.5 + bd>=20": rch & (ev["base_days"] >= 20),
        "v2.5 + bd 15~35": rch & (ev["base_days"] >= 15) & (ev["base_days"] <= 35),
        "v2.5 + bd 20~35": rch & (ev["base_days"] >= 20) & (ev["base_days"] <= 35),
        "v2.5 + bd<=35": rch & (ev["base_days"] <= 35),
    }
    for tag, m in arms.items():
        cur = daily(ev[m])
        diff = (cur - ref.reindex(cur.index)).dropna()
        if len(diff) < 3:
            P("  %-24s 共同交易日不足" % tag)
            continue
        est, lo, hi, nbk = block_boot(diff, dmap, NB, CI)
        sig = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
        P("  %-24s %+7.2fpp  90%%CI [%+6.2f, %+6.2f]  %s  (%d 区块)" % (
            tag, est * 100, lo * 100, hi * 100, sig, nbk))

    # ---------- ⑤ 逐年一致性 ----------
    P("")
    P("⑤ 逐年一致性（每年日均 ab_fwd60，看加天数门是否年年更差）")
    P("")
    yr = pd.to_datetime(ev["date"]).dt.year
    P("  %-8s %10s %12s %12s %12s" % ("年份", "v2.5", "bd>=15", "15~35", "20~35"))
    for y in sorted(yr.unique()):
        a = ev[rch & (yr == y)]["ab_fwd60"].mean() * 100
        b = ev[rch & (ev["base_days"] >= 15) & (yr == y)]["ab_fwd60"].mean() * 100
        c = ev[rch & (ev["base_days"] >= 15) & (ev["base_days"] <= 35) & (yr == y)]["ab_fwd60"].mean() * 100
        d2 = ev[rch & (ev["base_days"] >= 20) & (ev["base_days"] <= 35) & (yr == y)]["ab_fwd60"].mean() * 100
        P("  %-8s %+9.2f%% %+11.2f%% %+11.2f%% %+11.2f%%" % (y, a, b, c, d2))

    # ---------- 落盘 ----------
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    P("")
    P(f"已保存 → {OUT}")
    P(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
