# -*- coding: utf-8 -*-
"""
【全量区块 bootstrap 复核】把此前所有基于 fwd60 的参数结论重算一遍

背景（方法性修正）：
    fwd60 = 未来 60 个【交易日】的收益。相邻两个事件日的持有窗口重叠约 98%，
    因此「按交易日抽样」的 bootstrap 把相关性当作独立性，CI 系统性偏窄。

正确做法：区块 bootstrap（block bootstrap / moving-block 的简化版）
    把时间轴切成互不重叠的、长度 = 持有期(60 交易日) 的区块，
    以【区块】而非【天】作为重抽样单位。
    区块之间不重叠 → 区块之间的观测近似独立 → CI 反映真实的独立信息量。

本脚本复核五个参数维度（固定其余为 v2 基准）：
    width30 / prox_now / amt30 / turn30上限 / rev_yoy
输出：全样本 + 分周期的点估计、90%CI、显著性、日均票、区块数；
      以及每一档相对【当前生效基准值】的【配对区块 bootstrap】检验。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin  # noqa: E402

RNG = np.random.default_rng(20261027)
HB = 60          # 区块长度 = 持有期（交易日）
NB = 4000        # 重抽样次数
CI = 0.90

# v2 当前生效基准
BASE = dict(prox=0.95, width=0.15, amt=4e8, turn=6.0, rev=None)


# ---------------------------------------------------------------- 区块工具
def make_dmap(ev_all: pd.DataFrame):
    days = pd.DatetimeIndex(sorted(pd.to_datetime(ev_all["date"]).unique()))
    return {dt: i for i, dt in enumerate(days)}


def add_bi(d: pd.DataFrame, dmap: dict) -> pd.DataFrame:
    d = d.copy()
    d["_i"] = pd.to_datetime(d["date"]).map(lambda x: dmap.get(pd.Timestamp(x), np.nan))
    d = d.dropna(subset=["_i"])
    d["_bi"] = (d["_i"] // HB).astype(int)
    return d


def daily(d: pd.DataFrame, col: str = "ab_fwd60") -> pd.Series:
    return d.groupby("date")[col].mean()


def block_boot(daily_s: pd.Series, dmap: dict, n_boot: int = NB, ci: float = CI):
    """对日均值序列做区块 bootstrap。返回 (点估计, lo, hi, 区块数)"""
    if len(daily_s) < 3:
        return np.nan, np.nan, np.nan, 0
    idx = pd.to_datetime(daily_s.index)
    bi = pd.Index(idx).map(lambda x: dmap.get(pd.Timestamp(x), np.nan))
    bi = pd.Series(bi, index=daily_s.index)
    bi = bi.dropna()
    daily_s = daily_s.loc[bi.index]
    vals = daily_s.values.astype(float)
    bis = (bi.values.astype(int) // HB)      # ← 必须整除 HB，否则退化成按天抽样
    uniq = np.unique(bis)
    nb = len(uniq)
    point = float(np.mean(vals))
    if nb < 2:
        return point, np.nan, np.nan, nb
    # 向量化：按区块号排序后用 reduceat 求各块总和与长度
    order = np.argsort(bis, kind="stable")
    vs, bs = vals[order], bis[order]
    starts = np.searchsorted(bs, uniq, side="left")
    ends = np.searchsorted(bs, uniq, side="right")
    sums = np.add.reduceat(vs, starts)
    lens = (ends - starts).astype(float)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        pick = RNG.integers(0, nb, nb)
        boots[k] = sums[pick].sum() / lens[pick].sum()
    lo = float(np.nanpercentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(boots, (1 + ci) / 2 * 100))
    return point, lo, hi, nb


def paired_block_boot(d_a: pd.DataFrame, d_b: pd.DataFrame, dmap: dict,
                      col: str = "ab_fwd60", n_boot: int = NB, ci: float = CI):
    """配对区块 bootstrap：只在两者都有观测的交易日上做差"""
    sa = daily(d_a, col)
    sb = daily(d_b, col)
    common = sa.index.intersection(sb.index)
    if len(common) < 3:
        return np.nan, np.nan, np.nan, 0
    diff = (sa.loc[common] - sb.loc[common])
    point, lo, hi, nb = block_boot(diff, dmap, n_boot, ci)
    return point, lo, hi, nb


def sig(pt, lo, hi):
    if any(pd.isna(x) for x in (pt, lo, hi)):
        return "??"
    if lo > 0:
        return "显著为正"
    if hi < 0:
        return "显著为负"
    return "含零"


def bar(t=""):
    print("\n" + "=" * 100)
    if t:
        print(t)
        print("=" * 100)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    # load_events() 已内含：beta clip、ab_fwd60 重算、prox_now 构造、缺失剔除
    ev_all = load_events()

    fin = load_fin()
    ev_all = attach_fin(ev_all, fin)
    ev_all = ev_all.sort_values(["code", "date"])
    ev_all["YSTZ_prev"] = ev_all.groupby("code")["YSTZ"].shift(1)

    dmap = make_dmap(ev_all)
    ndays = len(dmap)
    print(f"事件表 {len(ev_all):,} 行 / {ev_all['date'].nunique()} 交易日")
    print(f"区块长度 = {HB} 交易日 → 全局共 {int(np.ceil(ndays / HB))} 个区块")
    print("指标口径：beta 调整 60 日超额（ab_fwd60），按交易日截面平均后再按区块抽样")
    print(f"重抽样 {NB} 次，置信水平 {int(CI*100)}%")

    # 三个周期 + 全样本
    SCOPES = [("全样本", ev_all)]
    for rg in ["UP", "CHOP", "DOWN"]:
        SCOPES.append((rg, ev_all[ev_all["regime"] == rg]))

    # ---------------- 各参数扫描 ----------------
    GRIDS = [
        ("箱宽 width30 <=", "width", [0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25], 0.15),
        ("距年高 prox >=",  "prox",  [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98], 0.95),
        ("日均额 amt30 >=", "amt",   [1e8, 2e8, 3e8, 4e8, 5e8, 6e8, 8e8], 4e8),
        ("换手 turn30 <=",  "turn",  [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 99.0], 6.0),
        ("营收同比 YSTZ >=", "rev",  [-999.0, 0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0], 30.0),
    ]

    def apply_gate(d: pd.DataFrame, dim: str, v: float) -> pd.DataFrame:
        """在 v2 基准上，只改 dim 这一维"""
        dd = d[(d["prox_now"] >= BASE["prox"]) & (d["width30"] <= BASE["width"]) &
               (d["amt30"] >= BASE["amt"]) & (d["turn30"] <= BASE["turn"])]
        if dim == "width":
            dd = d[(d["prox_now"] >= BASE["prox"]) & (d["width30"] <= v) &
                   (d["amt30"] >= BASE["amt"]) & (d["turn30"] <= BASE["turn"])]
        elif dim == "prox":
            dd = d[(d["prox_now"] >= v) & (d["width30"] <= BASE["width"]) &
                   (d["amt30"] >= BASE["amt"]) & (d["turn30"] <= BASE["turn"])]
        elif dim == "amt":
            dd = d[(d["prox_now"] >= BASE["prox"]) & (d["width30"] <= BASE["width"]) &
                   (d["amt30"] >= v) & (d["turn30"] <= BASE["turn"])]
        elif dim == "turn":
            dd = d[(d["prox_now"] >= BASE["prox"]) & (d["width30"] <= BASE["width"]) &
                   (d["amt30"] >= BASE["amt"]) & (d["turn30"] <= v)]
        elif dim == "rev":
            dd = dd[dd["YSTZ"].notna()]
            if v > -900:
                dd = dd[dd["YSTZ"] >= v]
        return dd

    for title, dim, grid, basev in GRIDS:
        bar(f"A. {title}   （基准值 {basev}，其余四维固定为 v2 基准）")
        for name, sub in SCOPES:
            print(f"\n  【{name}】")
            print("    %-14s %7s %7s %7s %9s %18s %10s" %
                  ("取值", "事件", "交易日", "区块", "日均票", "β调整超额", "判定"))
            ref = None
            for v in grid:
                d = apply_gate(sub, dim, v)
                if len(d) < 10:
                    print("    %-14s (样本不足)" % fmtv(dim, v))
                    continue
                s = daily(d)
                pt, lo, hi, nb = block_boot(s, dmap)
                per = float(d.groupby("date").size().mean())
                mark = "  ←基准" if abs(v - basev) < 1e-9 else ""
                print("    %-14s %7d %7d %7d %9.2f %+8.2f%% CI[%+6.2f,%+6.2f] %s%s" %
                      (fmtv(dim, v), len(d), d["date"].nunique(), nb, per,
                       pt * 100, lo * 100, hi * 100, sig(pt, lo, hi), mark))
                if abs(v - basev) < 1e-9:
                    ref = apply_gate(sub, dim, v)
            # 配对检验：每档 vs 基准
            if ref is not None and len(ref) >= 10:
                print("    --- 配对区块检验（各档 减 基准）---")
                print("      %-14s %9s %18s %10s" % ("取值", "配对日", "差值", "判定"))
                for v in grid:
                    if abs(v - basev) < 1e-9:
                        continue
                    d = apply_gate(sub, dim, v)
                    if len(d) < 10:
                        continue
                    pt, lo, hi, nb = paired_block_boot(d, ref, dmap)
                    if pd.isna(pt):
                        continue
                    print("      %-14s %9d %+8.2fpp CI[%+6.2f,%+6.2f] %s" %
                          (fmtv(dim, v), nb, pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))

    # ---------------- 营收：全周期最终裁决 ----------------
    bar("B. 营收门槛最终裁决：分周期 vs 全市场统一")
    print("\n  说明：结构门 = prox>=0.95 + 箱宽<=15% + 额>=4亿 + 换手<=6%（不含营收）\n")
    for name, sub in SCOPES:
        struct = apply_gate(sub, "rev", -999.0)   # 结构门，营收只要求有数据
        if len(struct) < 10:
            print(f"  【{name}】样本不足")
            continue
        s = daily(struct)
        pt0, lo0, hi0, nb0 = block_boot(s, dmap)
        print(f"  【{name}】结构门基准: n={len(struct)} 日={struct['date'].nunique()} "
              f"区块={nb0} 日均票={struct.groupby('date').size().mean():.2f} "
              f"→ {pt0*100:+.2f}% CI[{lo0*100:+.2f},{hi0*100:+.2f}] {sig(pt0,lo0,hi0)}")
        print("      %-14s %7s %7s %9s %20s %10s" %
              ("营收门槛", "事件", "区块", "日均票", "相对结构门增量", "判定"))
        for v in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
            d = struct[struct["YSTZ"] >= v]
            if len(d) < 10:
                print("      %-14s (样本不足)" % (f">={v:.0f}%",))
                continue
            pt, lo, hi, nb = paired_block_boot(d, struct, dmap)
            if pd.isna(pt):
                continue
            print("      %-14s %7d %7d %9.2f %+9.2fpp CI[%+6.2f,%+6.2f] %s" %
                  (f">={v:.0f}%", len(d), nb, d.groupby("date").size().mean(),
                   pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))
        print()

    # ---------------- 营收：分档形态（全样本，看是否单调） ----------------
    bar("C. 营收分档形态（结构门内，全样本）")
    struct = apply_gate(ev_all, "rev", -999.0)
    bins = [(-np.inf, 0), (0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, np.inf)]
    labs = ["<=0%", "0~10%", "10~20%", "20~30%", "30~50%", "50~100%", ">100%"]
    print("  %-10s %7s %7s %8s %9s %18s %10s" %
          ("分档", "事件", "区块", "β中位", "日均票", "β调整超额", "判定"))
    for (l, h), lb in zip(bins, labs):
        d = struct[(struct["YSTZ"] > l) & (struct["YSTZ"] <= h)]
        if len(d) < 10:
            continue
        s = daily(d)
        pt, lo, hi, nb = block_boot(s, dmap)
        print("  %-10s %7d %7d %8.2f %9.2f %+8.2f%% CI[%+6.2f,%+6.2f] %s" %
              (lb, len(d), nb, d["beta"].median(), d.groupby("date").size().mean(),
               pt * 100, lo * 100, hi * 100, sig(pt, lo, hi)))


def fmtv(dim, v):
    if dim == "amt":
        return f"{v/1e8:.0f}亿"
    if dim == "width":
        return f"{v*100:.0f}%"
    if dim == "prox":
        return f"{v:.2f}"
    if dim == "turn":
        return "不设" if v >= 90 else f"{v:.0f}%"
    if dim == "rev":
        return "不设" if v <= -900 else f"{v:.0f}%"
    return str(v)


if __name__ == "__main__":
    main()
