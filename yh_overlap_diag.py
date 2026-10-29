# -*- coding: utf-8 -*-
"""
【重叠问题的直接证据】fwd60 的自相关诊断

命题：事件日 t 的 fwd60 覆盖 [t+1, t+60]；事件日 t+1 覆盖 [t+2, t+61]。
      两者共享 59 天 → 重叠率 59/60 = 98.3% → 高度相关，不是独立观测。

三种实证（全部按【交易日】对齐，不是按事件序号——这是上一版的错误）：
  A. 个股层面：同一只票，相隔 k 个交易日的两次 fwd60 的相关系数
  B. 组合层面：日截面均值序列的自相关 ACF(k)
  C. 后果：按天 bootstrap vs 区块 bootstrap 的 CI 宽度对比
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, base_v2  # noqa: E402

RNG = np.random.default_rng(20261027)
HB = 60
NB = 4000


def bar(t=""):
    print("\n" + "=" * 100)
    if t:
        print(t)
        print("=" * 100)


def lag_corr_many(mat: pd.DataFrame, k: int, min_obs: int = 15):
    """mat: index=交易日(连续), columns=股票, values=fwd60
    返回每只票在滞后 k 个【交易日】上的相关系数的中位数"""
    a = mat.iloc[:-k].to_numpy() if k > 0 else mat.to_numpy()
    b = mat.iloc[k:].to_numpy() if k > 0 else mat.to_numpy()
    out = []
    for j in range(a.shape[1]):
        x, y = a[:, j], b[:, j]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < min_obs:
            continue
        sx, sy = x[ok].std(), y[ok].std()
        if sx > 0 and sy > 0:
            out.append(np.corrcoef(x[ok], y[ok])[0, 1])
    return float(np.median(out)) if out else np.nan


def acf(s: np.ndarray, max_lag: int):
    v = s - s.mean()
    denom = (v * v).sum()
    out = {}
    for k in range(0, max_lag + 1):
        if k == 0:
            out[k] = 1.0
            continue
        a, b = v[:-k], v[k:]
        out[k] = float((a * b).sum() / denom) if denom > 0 else np.nan
    return pd.Series(out)


def main() -> None:
    ev = load_events()
    ev["_d"] = pd.to_datetime(ev["date"])
    print(f"事件表 {len(ev):,} 行 / {ev['date'].nunique()} 交易日 / {ev['code'].nunique()} 股")

    b = base_v2(ev)

    # ------------------------------------------------------------------ A
    bar("A. 个股层面：同一只票，相隔 k 个【交易日】的两次 fwd60 相关系数")
    print("\n  （v2 基准池内、有 ≥60 个事件日的票；各自算滞后相关后取中位数）\n")
    cnt = b.groupby("code").size()
    codes = cnt[cnt >= 60].index
    print(f"  符合条件的股票 {len(codes)} 只\n")
    sub = b[b["code"].isin(codes)]
    # pivot 到完整交易日网格（缺失=NaN），保证「k 个交易日」的对齐正确
    all_days = pd.DatetimeIndex(sorted(ev["_d"].unique()))
    mat = sub.pivot_table(index="_d", columns="code", values="fwd60", aggfunc="mean")
    mat = mat.reindex(all_days)
    print("  %8s %12s   %s" % ("间隔k(交易日)", "相关系数中位", "含义"))
    for k in [1, 2, 5, 10, 20, 30, 40, 50, 59, 60, 61, 80]:
        r = lag_corr_many(mat, k)
        note = ""
        if k == 1:
            note = "← 相邻两日：窗口重叠 59/60 = 98.3%"
        elif k == 30:
            note = "← 重叠 30/60 = 50%"
        elif k == 59:
            note = "← 仍重叠 1 天"
        elif k == 60:
            note = "← 刚好不重叠"
        elif k == 61:
            note = "← 完全不重叠"
        print("  %8d %12.3f   %s" % (k, r, note))
    print("\n  → 相关随间隔衰减；到 k≈60（=持有期长度）才降到噪声水平。")
    print("    这就是「相邻两天的 fwd60 不是独立观测」的直接证据。")

    # ------------------------------------------------------------------ B
    bar("B. 组合层面：日截面均值序列的自相关（最关键）")
    print("\n  构造：每个交易日等权买入当日基准池全部股票，持有 60 交易日。")
    print("        该策略的日收益序列 = 每日 fwd60 的截面均值。\n")
    daily = b.groupby("date")["fwd60"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    v = daily.to_numpy(dtype=float)
    a = acf(v, 130)
    print("  %8s %12s   %s" % ("间隔k(交易日)", "自相关", "含义"))
    for k in [1, 2, 5, 10, 20, 30, 40, 50, 59, 60, 61, 90, 120]:
        if k in a.index:
            note = ""
            if k == 1:
                note = "← 相邻两日"
            elif k == 60:
                note = "← 持有期长度，理论上应≈0"
            print("  %8d %12.3f   %s" % (k, a.loc[k], note))
    print("\n  → 自相关在 k=1 时接近 1，随 k 单调衰减，到 k≈60 才消失。")
    print("    说明这条日收益序列里，真正独立的信息块约等于 交易日数/60。")

    # ------------------------------------------------------------------ C
    bar("C. 后果：两种 bootstrap 的 CI 宽度差多少")
    n_days = len(v)
    dmap = {d: i for i, d in enumerate(all_days)}
    bis = np.array([dmap[d] // HB for d in daily.index])
    uniq = np.unique(bis)
    n_blocks = len(uniq)
    print(f"\n  基准池：{n_days} 个有观测的交易日 → 切进 {n_blocks} 个非重叠区块（每块 {HB} 日）\n")

    bd = np.empty(NB)
    for i in range(NB):
        bd[i] = RNG.choice(v, n_days, replace=True).mean()
    pt = float(v.mean())
    lo_d, hi_d = np.percentile(bd, [5, 95])

    order = np.argsort(bis, kind="stable")
    vs, bs = v[order], bis[order]
    starts = np.searchsorted(bs, uniq, side="left")
    ends = np.searchsorted(bs, uniq, side="right")
    sums = np.add.reduceat(vs, starts)
    lens = (ends - starts).astype(float)
    bb = np.empty(NB)
    for i in range(NB):
        pick = RNG.integers(0, n_blocks, n_blocks)
        bb[i] = sums[pick].sum() / lens[pick].sum()
    lo_b, hi_b = np.percentile(bb, [5, 95])

    print("  %-24s %12s %26s %12s" % ("方法", "有效样本", "90%CI", "CI宽度"))
    print("  %-24s %12d %26s %12.4f" % ("按交易日抽样(旧)", n_days,
                                        f"[{lo_d*100:+.2f}%, {hi_d*100:+.2f}%]", hi_d - lo_d))
    print("  %-24s %12d %26s %12.4f" % ("区块抽样(新)", n_blocks,
                                        f"[{lo_b*100:+.2f}%, {hi_b*100:+.2f}%]", hi_b - lo_b))
    print(f"\n  → CI 宽度放大 {(hi_b-lo_b)/(hi_d-lo_d):.2f} 倍")
    print(f"  → 理论上 SE ∝ 1/√n_eff：n_eff 从 {n_days} 降到 {n_blocks}，"
          f"应放大约 √({n_days}/{n_blocks}) = {np.sqrt(n_days/max(n_blocks,1)):.2f} 倍")

    # ------------------------------------------------------------------ D
    bar("D. 为什么区块 bootstrap 是对的（原理）")
    print("""
  1) 问题不在【有多少数据】，而在【有多少独立信息】。
     fwd60 的相关长度 = 60 个交易日（A/B 段实测：相关系数到 k≈60 才衰减到噪声）。

  2) 按天 bootstrap 把 N 个交易日当成 N 个独立观测，实际上独立信息块只有 N/60 个。
     → 标准误被低估约 √(N/60) 倍 → CI 系统性偏窄 → 假阳性（把噪声当 alpha）。

  3) 区块 bootstrap 以【长度 = 相关长度】的不重叠区块为抽样单位：
     · 区块【内部】的相关结构被完整保留（不对块内做任何独立性假设）
     · 区块【之间】因不重叠，其 fwd60 窗口几乎无交集 → 近似独立
     → 以块为单位重抽样，估计出的变异性反映真实独立信息量，CI 才可信。

  4) 它是【保守】的：块数少时估计本身有噪声，且丢弃了跨块的部分信息，
     真实精度介于两种口径之间。但假阳性（实施一个无效参数）的代价
     远高于假阴性（漏掉一个弱效应），所以宁可保守。

  5) 本项目在此基础上再做【配对】：只在两个配置都有观测的交易日上做差，
     先消掉共同的日期/市场效应，再按区块抽样 → 最严格口径。

  6) ⚠️ 实现陷阱（本项目踩过的坑）：
     区块号必须是【全局交易日索引 // 60】。若误用「事件序号」或直接拿交易日
     数当区块数，区块数会等于交易日数，方法就悄悄退化回按天 bootstrap。
""")


if __name__ == "__main__":
    main()
