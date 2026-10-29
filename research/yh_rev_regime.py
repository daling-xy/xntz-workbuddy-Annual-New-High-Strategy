# -*- coding: utf-8 -*-
"""
营收纪律的【分周期】增量价值 —— 严格验证

要回答的问题（用户提出）：
  「DOWN 周期不加营收纪律更好、UP 周期加营收纪律更好」是否成立？

方法要点（避免上一轮踩过的坑）：
 1. 财报 join 用【法定披露截止日】，不用 UPDATE_DATE（后者是修正日，会造成前视）
 2. 有效样本量看【独立交易日数】，不看事件数（同一天的多只票不独立）
 3. 配对 bootstrap：只在同一批交易日上比较（控制了市场环境）
 4. 分年拆解：检查效应是否被 1~2 个年份主导
 5. 多重检验：跨 3 周期 × 多门槛多次比较，需降低对单个 p 值的信任
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin, base_v2  # noqa: E402

RNG = np.random.default_rng(20261027)
N_BOOT = 4000


# ---------------------------------------------------------------- 工具
def _daily_mean(d: pd.DataFrame, col: str = "ab_fwd60") -> pd.Series:
    """先按日截面平均，再跨日 —— 关键：同一天的多只票只算一个观测"""
    return d.groupby("date")[col].mean()


def boot_ci_daily(d: pd.DataFrame, col: str = "ab_fwd60", n_boot: int = N_BOOT,
                  ci: float = 0.90) -> tuple:
    """按交易日聚类 bootstrap"""
    dm = _daily_mean(d, col).dropna().values
    if len(dm) < 5:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(dm), size=(n_boot, len(dm)))
    means = dm[idx].mean(axis=1)
    pt = float(dm.mean())
    return (pt, float(np.percentile(means, (1 - ci) / 2 * 100)),
            float(np.percentile(means, (1 + ci) / 2 * 100)))


def paired_boot(a: pd.DataFrame, b: pd.DataFrame, col: str = "ab_fwd60",
                n_boot: int = N_BOOT, ci: float = 0.90) -> tuple:
    """配对 bootstrap：a - b，只在共同交易日上比较"""
    da = _daily_mean(a, col)
    db = _daily_mean(b, col)
    common = da.index.intersection(db.index)
    if len(common) < 5:
        return (np.nan, np.nan, np.nan, 0)
    x = da.loc[common].values
    y = db.loc[common].values
    diff = x - y
    idx = RNG.integers(0, len(diff), size=(n_boot, len(diff)))
    bs = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(bs, (1 - ci) / 2 * 100)),
            float(np.percentile(bs, (1 + ci) / 2 * 100)), len(common))


def sig(pt, lo, hi):
    if np.isnan(pt):
        return "样本不足"
    if lo > 0:
        return "显著为正"
    if hi < 0:
        return "显著为负"
    return "含零·无效"


def describe(tag, d, base):
    n = len(d)
    nd = d["date"].nunique()
    per = n / max(nd, 1)
    pt, lo, hi = boot_ci_daily(d)
    beta = d["beta"].median()
    win = float((d["fwd60"] > 0).mean() * 100)
    big = float((d["fwd60"] < -0.20).mean() * 100)
    print("  %-16s n=%5d  交易日=%4d  日均%5.2f只  β中位%5.2f  %+7.2f%% CI[%+6.2f,%+6.2f]  胜率%5.1f%%  大亏%5.1f%%  %s"
          % (tag, n, nd, per, beta, pt * 100, lo * 100, hi * 100, win, big, sig(pt, lo, hi)))
    return dict(tag=tag, n=n, nd=nd, per=per, pt=pt, lo=lo, hi=hi, beta=beta, win=win, big=big)


def bar(t):
    print("=" * 118)
    print(t)
    print("=" * 118)


def main():
    ev = load_events()
    fin = load_fin()
    e = attach_fin(ev, fin)
    print("事件表: %s 行 / %s 股 / %s 交易日" % (f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique()))
    print("日期区间: %s ~ %s" % (ev["date"].min(), ev["date"].max()))
    print("无前视 join：YSTZ 覆盖率 %.1f%%" % (e["YSTZ"].notna().mean() * 100))
    print()

    # ---------- A. 周期本身的样本结构（必须先看清这个） ----------
    bar("A. 三个周期的【独立交易日】结构 —— 有效样本量到底有多大？")
    print("  用 β 调整后 60 日超额（ab_fwd60）；v2 四纪律基准池")
    print()
    print("  %-8s %8s %10s %12s %12s" % ("周期", "事件数", "独立交易日", "日均事件", "交易日占比"))
    tot_nd = e["date"].nunique()
    for rg in ["UP", "CHOP", "DOWN"]:
        sub = e[e["regime"] == rg]
        nd = sub["date"].nunique()
        print("  %-8s %8d %10d %12.2f %11.1f%%"
              % (rg, len(sub), nd, len(sub) / max(nd, 1), nd / tot_nd * 100))
    print()
    print("  ⚠️ 关键：bootstrap 按交易日聚类，所以真正的有效自由度是【独立交易日数】，不是事件数。")
    print("     FIX: 后续所有 CI 均基于交易日聚类抽样。")

    # ---------- B. 各周期：营收门槛扫描 + 配对增量检验 ----------
    for rg in ["DOWN", "UP", "CHOP"]:
        sub = e[e["regime"] == rg]
        base = base_v2(sub[sub["YSTZ"].notna()])
        # 基准：完全不看营收（>=0% 等价于只要求有数）
        b0 = base
        bar("B.%s  【%s 周期】营收门槛 vs 基准（日均 beta 调整超额）" % (rg, rg))
        print("  基准池事件 %d / 交易日 %d" % (len(base), base["date"].nunique()))
        print()
        describe("不看营收(基准)", b0, base)
        for thr in [10, 20, 25, 30, 40, 50]:
            d = base[base["YSTZ"] >= thr]
            if len(d) < 20 or d["date"].nunique() < 10:
                print("  %-16s n=%5d → 样本不足，跳过" % ("营收>=%d%%" % thr, len(d)))
                continue
            describe("营收>=%d%%" % thr, d, base)
        print()
        print("  --- 配对 bootstrap：加纪律 vs 基准（同一批交易日，差值>0 表示加纪律更优）---")
        for thr in [10, 20, 25, 30, 40, 50]:
            d = base[base["YSTZ"] >= thr]
            if len(d) < 20 or d["date"].nunique() < 10:
                continue
            pt, lo, hi, nday = paired_boot(d, b0)
            print("    >=%2d%% - 基准: 差 %+7.2fpp  CI[%+6.2f,%+6.2f]  (%3d 日)  → %s"
                  % (thr, pt * 100, lo * 100, hi * 100, nday, sig(pt, lo, hi)))
        print()

    # ---------- C. UP 期效应是否被个别年份主导 ----------
    bar("C. UP 期：营收效应是否被 1~2 个年份主导？（防过度拟合）")
    up = e[e["regime"] == "UP"]
    base_up = base_v2(up[up["YSTZ"].notna()])
    base_up = base_up.copy()
    base_up["yr"] = pd.to_datetime(base_up["date"]).dt.year
    print("  %-8s %10s %12s %12s %10s" % ("年份", "事件数", ">=30%超额", "基准超额", "30%的增量"))
    for y in sorted(base_up["yr"].unique()):
        sy = base_up[base_up["yr"] == y]
        if len(sy) < 15:
            continue
        d30 = sy[sy["YSTZ"] >= 30]
        if len(d30) < 5:
            print("  %-8d %10d %12s %12s %10s" % (y, len(sy), "-", "-", " >=30样本不足"))
            continue
        pt, lo, hi, nday = paired_boot(d30, sy)
        e0 = boot_ci_daily(sy)[0]
        e30 = boot_ci_daily(d30)[0]
        print("  %-8d %10d %11.2f%% %11.2f%%   %+8.2fpp  CI[%+6.2f,%+6.2f] (%d日)"
              % (y, len(sy), e30 * 100, e0 * 100, pt * 100, lo * 100, hi * 100, nday))
    print()
    print("  解读：若增量只在某一年显著，说明是【该轮风格行情】而非稳定规律。")

    # ---------- D. DOWN 期：负增速票是不是反而更好？（反常识，需证伪） ----------
    bar("D. DOWN 期：按 rev_yoy 分档（非累计），检验是否存在 U 型")
    dn = e[e["regime"] == "DOWN"]
    base_dn = base_v2(dn[dn["YSTZ"].notna()])
    bins = [(-np.inf, 0), (0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
    labs = ["<0%", "0~10%", "10~20%", "20~30%", "30~50%", ">50%"]
    print("  %-10s %8s %8s %12s %10s %10s" % ("分档", "n", "β中位", "超额", "胜率", "大亏率"))
    for (lo_, hi_), lb in zip(bins, labs):
        d = base_dn[(base_dn["YSTZ"] > lo_) & (base_dn["YSTZ"] <= hi_)]
        if len(d) < 10:
            print("  %-10s %8d → 样本不足" % (lb, len(d)))
            continue
        pt, l, h = boot_ci_daily(d)
        print("  %-10s %8d %8.2f %+11.2f%% %9.1f%% %9.1f%%  %s"
              % (lb, len(d), d["beta"].median(), pt * 100,
                 float((d["fwd60"] > 0).mean() * 100),
                 float((d["fwd60"] < -0.20).mean() * 100), sig(pt, l, h)))
    print()

    # ---------- E. 换手/箱宽等其他纪律的联动：营收是否在 DOWN 期与低波重复 ----------
    bar("E. 交叉检验：DOWN 期营收>=30% 与【箱宽】是否重复筛选同一批票")
    for w in [0.10, 0.15]:
        d = base_dn[base_dn["width30"] <= w]
        d0 = d
        d30 = d[d["YSTZ"] >= 30]
        print("  箱宽<=%d%%: 基准 n=%d (日均%.2f只)  >=30%% n=%d (日均%.2f只)"
              % (w * 100, len(d0), len(d0) / max(d0["date"].nunique(), 1),
                 len(d30), len(d30) / max(d30["date"].nunique(), 1)))
        if len(d30) >= 20 and d30["date"].nunique() >= 10:
            pt, lo, hi, nday = paired_boot(d30, d0)
            print("             配对差 %+.2fpp CI[%+.2f,%+.2f] (%d日) → %s"
                  % (pt * 100, lo * 100, hi * 100, nday, sig(pt, lo, hi)))
    print()

    # ---------- F. 全周期一次性交叉：给最终结论用 ----------
    bar("F. 汇总：加营收纪律的增量（pp），按周期")
    print("  %-8s %10s %10s %10s %10s %10s" % ("周期", ">=10%", ">=20%", ">=30%", ">=40%", ">=50%"))
    for rg in ["DOWN", "UP", "CHOP"]:
        sub = e[e["regime"] == rg]
        base = base_v2(sub[sub["YSTZ"].notna()])
        b0 = base
        cells = []
        for thr in [10, 20, 30, 40, 50]:
            d = base[base["YSTZ"] >= thr]
            if len(d) < 20 or d["date"].nunique() < 10:
                cells.append("    -   ")
                continue
            pt, lo, hi, nday = paired_boot(d, b0)
            mark = "*" if (lo > 0 or hi < 0) else " "
            cells.append("%+6.2f%s" % (pt * 100, mark))
        print("  %-8s %s" % (rg, " ".join(cells)))
    print()
    print("  * = 90%CI 不含零（显著）。注意跨 3 周期 × 5 门槛 = 15 次比较，")
    print("      按 Bonferroni 校正，单次显著性门槛应从 10% 收紧到约 0.7%，即此处标*的也需谨慎看待。")


if __name__ == "__main__":
    main()
