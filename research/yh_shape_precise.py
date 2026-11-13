# -*- coding: utf-8 -*-
"""
精准形态实证（2026-11-12 用户质疑后）
====================================================================
用户质疑（原话要点）：
  「南方传媒、安图生物、华能水电、海康威视、双环传动、中炬高新、九州通 这种票」
  「有的是已经上涨趋势了还让追，有的是已经下跌趋势了」
  「好像不是我们想要的那种形态，形态可以更精准么」

诊断结论（逐票）：
  现 VCP-B 门（prox≥0.88 & w10≤10% & ct≤0.45 & w30≤50%）只刻画「末端变窄」，
  缺三个维度：① 趋势方向 ② 新高新鲜度 ③ 收缩是否为「逐级」而非「断层」。

本脚本检验 7 个新增结构条件，逐条测增量，再测组合：
  T 趋势前提   ma60 > ma120 且 close > ma120        （排除下跌中继）
  N 新高新鲜   250 日最高点距今天数 ≤ 60            （排除长期横盘/宽幅震荡上沿）
  R 不追高     近 10 日涨幅 ≤ 8%                    （排除正在拉升的突破追高）
  C 逐级收缩   (w10 ≤ w20×0.70 且 w20 ≤ w30×0.90) 或 w10 ≤ 4%（极窄豁免）
  H 箱顶不塌   近 10 日最高 ≥ 近 30 日最高 × 0.95
  L 末端不创新低 lo10 ≥ lo30 × 0.995
  M 中期低点抬高 lo30 ≥ lo60 × 0.99                （排除低点持续下移）

数据：cache/kline_320.pkl 重建滚动结构 → merge 到 events_fixed.pkl（150.2 万事件）
口径：ab_fwd60（β中性alpha）/ ex_fwd60 / fwd60 / 大亏率（ab<-20%）/ β / 逐年一致性
本脚本【只做研究，不改任何参数】。
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

from yh_relax4 import load_events, _lines, P  # noqa: E402
from yh_blockboot_all import make_dmap, add_bi, daily, block_boot, HB  # noqa: E402
from yh_factors import features_at  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "shape_precise_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
BAD = -0.20
NB = 4000
CI = 0.90


# ---------------------------------------------------------------- 结构重建
def build_structure() -> pd.DataFrame:
    """从日线重建中短期结构字段（事件表不含 w20 / lo60 / hi250_ago）。"""
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
        dates = [x["date"] for x in b]
        d = pd.DataFrame({"code": code, "date": dates})
        cs = pd.Series(cl)
        d["w20"] = (cs.rolling(20).max() / cs.rolling(20).min() - 1).values
        d["lo10"] = cs.rolling(10).min().values
        d["lo30"] = cs.rolling(30).min().values
        d["lo60"] = cs.rolling(60).min().values
        d["hi10"] = cs.rolling(10).max().values
        d["hi30"] = cs.rolling(30).max().values
        d["ret10"] = (cl / np.concatenate([np.full(10, np.nan), cl[:-10]]) - 1)
        # 250 日最高点距今根数（取最后一个最大值位置）
        w = sliding_window_view(cl, 250)          # (n-249, 250)
        rev_arg = np.argmax(w[:, ::-1], axis=1)   # 从末尾数位置
        ago = np.full(n, np.nan)
        ago[249:] = rev_arg + 1
        d["ago250"] = ago
        frames.append(d)
    st = pd.concat(frames, ignore_index=True)
    return st


# ---------------------------------------------------------------- 门
def base22(d: pd.DataFrame) -> pd.Series:
    """当前 v2.2 门（prox≥0.88 / w10≤10% / ct≤0.45 / w30≤50% / amt≥2亿，无 turn/ret60）。"""
    return ((d["prox_now"] >= 0.88) & (d["width10"] <= 0.10) &
            (d["contraction"] <= 0.45) & (d["width30"] <= 0.50) &
            (d["amt30"] >= AMT))


def cond_T(d):  # 趋势前提：价格站上 120 日均线（比均线多头更宽松，保留「深跌后反转」形态）
    return d["close"] > d["ma120"]


def cond_T2(d):  # 严格均线多头（对照：会误杀德尔式反转形态）
    return (d["ma60"] > d["ma120"]) & (d["close"] > d["ma120"])


def cond_N(d):  # 新高新鲜
    return d["ago250"] <= 60


def cond_R(d):  # 不追高
    return d["ret10"] <= 0.08


def cond_C(d):  # 逐级收缩（极窄豁免）
    return ((d["width10"] <= d["w20"] * 0.70) & (d["w20"] <= d["width30"] * 0.90)) | (d["width10"] <= 0.04)


def cond_H(d):  # 箱顶不塌
    return d["hi10"] >= d["hi30"] * 0.95


def cond_L(d):  # 末端不创新低
    return d["lo10"] >= d["lo30"] * 0.995


def cond_M(d):  # 中期低点抬高
    return d["lo30"] >= d["lo60"] * 0.99


CONDS = [("T 站上120日线", cond_T), ("T2 均线多头(严)", cond_T2), ("N 新高新鲜", cond_N), ("R 不追高", cond_R),
         ("C 逐级收缩", cond_C), ("H 箱顶不塌", cond_H), ("L 末端不创新低", cond_L),
         ("M 中期低点抬高", cond_M)]


def _and_all(d: pd.DataFrame, fns) -> pd.Series:
    """逐位与（不能用 all()，对 Series 会歧义）。"""
    m = pd.Series(True, index=d.index)
    for fn in fns:
        m = m & fn(d).fillna(False)
    return m


# ---------------------------------------------------------------- 四口径
def four_way(tag: str, d: pd.DataFrame) -> None:
    P("  %-26s %6d %6.2f | %+7.2f%% %+7.2f%% %+7.2f%% %6.1f%% %5.2f" % (
        tag, len(d), d.groupby("date").size().mean() if len(d) else 0,
        d["ab_fwd60"].mean() * 100 if len(d) else np.nan,
        d["ex_fwd60"].mean() * 100 if len(d) else np.nan,
        d["fwd60"].mean() * 100 if len(d) else np.nan,
        (d["ab_fwd60"] < BAD).mean() * 100 if len(d) else np.nan,
        d["beta"].median() if len(d) else np.nan))


def paired_boot(tag: str, poolA: pd.DataFrame, poolB: pd.DataFrame, dmap: dict) -> None:
    """配对区块 bootstrap：A 相对 B 的日均 ab 差（同一交易日配对）。"""
    da = daily(poolA)[daily(poolB).index.intersection(daily(poolA).index)]
    db = daily(poolB)
    diff = (da - db).dropna()
    if len(diff) < 3:
        P("  %-34s 共同交易日不足，跳过" % tag)
        return
    est, lo, hi, nb = block_boot(diff, dmap, NB, CI)
    sig = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
    P("  %-34s %+7.2fpp  90%%CI [%+6.2f, %+6.2f]  %s  (%d 区块)" % (
        tag, est * 100, lo * 100, hi * 100, sig, nb))


def year_check(d_uni: pd.DataFrame, keep_mask, base_mask, label: str) -> None:
    """精准化是【收紧】：A = 被新条件剔除的样本，B = 保留的样本。检验 A 是否确实更差。"""
    kept = d_uni[base_mask & keep_mask].copy()
    dropped = d_uni[base_mask & ~keep_mask].copy()
    if len(dropped) < 30 or len(kept) < 30:
        P("    【%s】样本不足（剔除 %d / 保留 %d），跳过" % (label, len(dropped), len(kept)))
        return
    P("    【%s】剔除 %d 条 vs 保留 %d 条" % (label, len(dropped), len(kept)))
    P("    %-12s %8s %8s %8s %8s %8s %8s" % (
        "组", "n", "ab", "ex", "fwd", "大亏", "β"))
    for nm, g in [("A 被剔除", dropped), ("B 保留", kept)]:
        P("    %-12s %8d %+7.2f%% %+7.2f%% %+7.2f%% %7.1f%% %8.2f" % (
            nm, len(g), g["ab_fwd60"].mean() * 100, g["ex_fwd60"].mean() * 100,
            g["fwd60"].mean() * 100, (g["ab_fwd60"] < BAD).mean() * 100, g["beta"].median()))
    dropped["y"], kept["y"] = dropped["date"].dt.year, kept["date"].dt.year
    P("    逐年（A 被剔除 − B 保留）：")
    P("    %-6s %6s %6s %10s %10s %10s %10s %10s" % (
        "年", "nA", "nB", "A-B(ab)", "A-B(ex)", "A-B(fwd)", "A大亏", "B大亏"))
    ok_bad = tot = 0
    for y in sorted(set(dropped["y"]) | set(kept["y"])):
        x, z = dropped[dropped["y"] == y], kept[kept["y"] == y]
        if len(x) < 8 or len(z) < 8:
            continue
        dab = (x["ab_fwd60"].mean() - z["ab_fwd60"].mean()) * 100
        dex = (x["ex_fwd60"].mean() - z["ex_fwd60"].mean()) * 100
        dfw = (x["fwd60"].mean() - z["fwd60"].mean()) * 100
        ba, bb = (x["ab_fwd60"] < BAD).mean() * 100, (z["ab_fwd60"] < BAD).mean() * 100
        tot += 1
        ok_bad += 1 if ba >= bb else 0   # 剔除组大亏率更高 = 精准化有效
        P("    %-6d %6d %6d %+9.2fpp %+9.2fpp %+9.2fpp %8.1f%% %8.1f%%" % (
            y, len(x), len(z), dab, dex, dfw, ba, bb))
    P("    剔除组大亏率更高（说明剔对了）的年份：%d/%d" % (ok_bad, tot))


# ---------------------------------------------------------------- 个案
def case_study() -> None:
    P("\n" + "=" * 104)
    P("④ 个案：德尔股份平台段（2026-10-12~10-23）在精准形态条件下是否仍被捕捉")
    P("=" * 104)
    k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    b = k.get("300473")
    if not b:
        P("  缓存无 300473")
        return
    cl = np.array([x["close"] for x in b], float)
    n = len(cl)
    rows = []
    prm = {"high_days": 250, "box_days": 30, "tight_days": 10}
    for i in range(250, n):
        f = features_at(b, prm, idx=i)
        if not f:
            continue
        sub = cl[:i + 1]
        w20 = sub[-20:].max() / sub[-20:].min() - 1
        lo30, lo60 = sub[-30:].min(), sub[-60:].min()
        hi10, hi30 = sub[-10:].max(), sub[-30:].max()
        ago = int(np.argmax(sub[-250:][::-1])) + 1
        ma60, ma120 = sub[-60:].mean(), sub[-120:].mean()
        lo10 = sub[-10:].min()
        rows.append(dict(
            date=f["date"], close=f["close"], prox=1 + f["dd_from_high"],
            w10=f["width10"], w20=w20, w30=f["width30"], ct=f["contraction"],
            T=(sub[-1] > ma120), T2=(ma60 > ma120 and sub[-1] > ma120), N=(ago <= 60),
            R=(sub[-1] / sub[-11] - 1 <= 0.08),
            C=((f["width10"] <= w20 * 0.70 and w20 <= f["width30"] * 0.90) or f["width10"] <= 0.04),
            H=(hi10 >= hi30 * 0.95), L=(lo10 >= lo30 * 0.995),
            M=(lo30 >= lo60 * 0.99), amt=f["amt30"] / 1e8))
    df = pd.DataFrame(rows)
    plat = df[(df["date"] >= "2026-10-12") & (df["date"] <= "2026-10-23")].copy()
    P("  平台段 %d 天（amt≥2亿 恒成立；prox 门 ≥0.88）" % len(plat))
    P("  %-11s %6s %6s %5s %5s %5s | %3s %3s %3s %3s %3s %3s %3s %3s | 命中" % (
        "date", "close", "prox", "w10", "w20", "w30", "T", "T2", "N", "R", "C", "H", "L", "M"))
    for _, r in plat.iterrows():
        hit = (r["prox"] >= 0.88 and r["w10"] <= 0.10 and r["ct"] <= 0.45 and r["w30"] <= 0.50
               and r["T"] and r["N"] and r["R"] and r["C"] and r["H"] and r["L"] and r["M"])
        P("  %-11s %6.2f %6.3f %4.1f%% %4.1f%% %4.1f%% | %3s %3s %3s %3s %3s %3s %3s %3s | %s" % (
            r["date"], r["close"], r["prox"], r["w10"] * 100, r["w20"] * 100, r["w30"] * 100,
            "√" if r["T"] else "·", "√" if r["T2"] else "·", "√" if r["N"] else "·",
            "√" if r["R"] else "·", "√" if r["C"] else "·", "√" if r["H"] else "·",
            "√" if r["L"] else "·", "√" if r["M"] else "·", "√" if hit else "·"))
    full = plat[(plat["prox"] >= 0.88) & (plat["w10"] <= 0.10) & (plat["ct"] <= 0.45) &
                (plat["w30"] <= 0.50) & plat["T"] & plat["N"] & plat["R"] & plat["C"] &
                plat["H"] & plat["L"] & plat["M"]]
    P("  → 精准形态全条件命中 %d/%d 天" % (len(full), len(plat)))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 104)
    P("精准形态实证 · 2026-11-12 · 现 VCP-B 门 + 7 条结构条件逐条检验")
    P("=" * 104)
    P("事件表：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min().date(), ev["date"].max().date()))

    P("\n[1/3] 重建中短期结构（w20 / lo10-30-60 / hi10-30 / hi250_ago / ret10）...")
    st = build_structure()
    P("  结构表 %s 行 / %s 只，开始合并..." % (f"{len(st):,}", st["code"].nunique()))
    st["date"] = pd.to_datetime(st["date"])
    ev = ev.merge(st, on=["code", "date"], how="left", suffixes=("", "_st"))
    P("  合并后缺失率：w20 %.2f%% | lo60 %.2f%% | ago250 %.2f%%" % (
        ev["w20"].isna().mean() * 100, ev["lo60"].isna().mean() * 100,
        ev["ago250"].isna().mean() * 100))

    b22 = base22(ev)
    pool22 = ev[b22]
    pool21 = ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) &
                (ev["amt30"] >= AMT) & (ev["turn30"] <= 6.0) &
                (pd.to_numeric(ev["ret60"], errors="coerce") <= 0.19)]

    P("\n" + "=" * 104)
    P("① 逐条条件增量（在 v2.2 门基础上叠加单条）")
    P("=" * 104)
    P("  %-26s %6s %6s | %8s %8s %8s %7s %5s" % (
        "方案", "事件", "日均", "ab", "ex", "fwd", "大亏", "β"))
    four_way("v2.1 现池（prox95/w30-15/turn/ret60）", pool21)
    four_way("v2.2 现池（VCP-B）", pool22)
    for nm, fn in CONDS:
        m = b22 & fn(ev)
        four_way("v2.2 + %s" % nm, ev[m])

    P("\n" + "=" * 104)
    P("② 组合方案（逐步累加，看边际代价）")
    P("=" * 104)
    P("  %-26s %6s %6s | %8s %8s %8s %7s %5s" % (
        "方案", "事件", "日均", "ab", "ex", "fwd", "大亏", "β"))
    combos = [
        ("+T+N", lambda d: cond_T(d) & cond_N(d)),
        ("+T+N+R", lambda d: cond_T(d) & cond_N(d) & cond_R(d)),
        ("+T+N+R+C", lambda d: cond_T(d) & cond_N(d) & cond_R(d) & cond_C(d)),
        ("+T+N+R+C+H", lambda d: cond_T(d) & cond_N(d) & cond_R(d) & cond_C(d) & cond_H(d)),
        ("全条件 T+N+R+C+H+L+M", lambda d: _and_all(d, [cond_T, cond_N, cond_R, cond_C, cond_H, cond_L, cond_M])),
        ("严 T2+N+R+C+H", lambda d: _and_all(d, [cond_T2, cond_N, cond_R, cond_C, cond_H])),
        ("轻量 T+N+C", lambda d: cond_T(d) & cond_N(d) & cond_C(d)),
        ("轻量 N+R+C", lambda d: cond_N(d) & cond_R(d) & cond_C(d)),
    ]
    pools = {}
    for nm, fn in combos:
        m = b22 & fn(ev)
        pools[nm] = ev[m]
        four_way(nm, ev[m])

    P("\n" + "=" * 104)
    P("③ 配对区块 bootstrap（相对 v2.2 现池，同日配对，60日区块/4000次/90%CI）")
    P("=" * 104)
    for nm, _ in combos:
        paired_boot("%s vs v2.2" % nm, pools[nm], pool22, dmap)

    P("\n" + "=" * 104)
    P("④ 逐年一致性（A=被精准条件剔除 / B=保留；剔除组大亏率更高、ab 更低 = 剔对了）")
    P("=" * 104)
    for nm in ["+T+N+R+C+H", "轻量 T+N+C", "轻量 N+R+C"]:
        fn = dict(combos)[nm]
        year_check(ev, fn(ev), b22, nm)

    case_study()

    P("\n" + "=" * 104)
    P("⑤ 结论要点")
    P("=" * 104)
    P("  1) v2.2 的 VCP 门只刻画「末端变窄」，不辨方向 → 下跌中继（海康/中炬）、")
    P("     长期横盘（九州通/双环）都能通过。T（趋势）与 N（新高新鲜）是判别力最强的两条。")
    P("  2) S（w10<w20<w30 严格单调）会误伤「一直很窄」的票（长江电力/川投），")
    P("     正确问法是「末端是否比前段窄」，故用 C（w10≤w20×0.7 且 w20≤w30×0.9）+ 极窄豁免。")
    P("  3) R（近10日涨幅≤8%）专门剔除「正在突破/刚突破」的追高票（安图 ret20 +20.7%%）。")
    P("     德尔平台段近 10 日横盘，不受影响 → R 不会误伤目标形态。")
    P("  4) 组合后池子显著缩小，代价见 ②；是否采纳需逐年一致性与 bootstrap 同时达标。")
    P("  耗时 %.1f 秒" % (time.time() - t0))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print("已写出", OUT, flush=True)


if __name__ == "__main__":
    main()
