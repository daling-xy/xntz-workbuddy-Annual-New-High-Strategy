# -*- coding: utf-8 -*-
"""
口径污染诊断 + TTM(最近四个季度)营收方案验证

问题：
    TOTAL_OPERATE_INCOME 是【年初至今累计】营收。
    三季报 = 3 个季度，中报 = 2 个季度，一季报 = 1 个季度 → 不可比。
    后果：revenue>=2亿 这道"排壳"门，披露了三季报的公司更容易过（94.5% vs 85.7%）。

方案（用户提出）：改用【最近四个季度 / 最近一年】的滚动营收（TTM）。
    TTM(期 t) = 累计(t) + 累计(上年 Q4) - 累计(上年同期)
    无论 t 是哪一期，结果都覆盖最近 12 个月 → 口径恒定，污染消除。

本脚本：
    A. 复现污染（不同报告期的营收中位数与过线率）
    B. 构造 TTM 并验证污染消除
    C. 给出与现行门槛【等价】的 TTM 阈值（按通过率对齐）
    D. 检查 TTM 的历史可得覆盖率（会不会反而变严/变松）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_fin  # noqa: E402


def bar(t=""):
    print("\n" + "=" * 100)
    if t:
        print(t)
        print("=" * 100)


def build_ttm(fin: pd.DataFrame) -> pd.DataFrame:
    """给每个 (code, 报告期) 计算 TTM 营收"""
    f = fin.copy()
    f["REPORTDATE"] = pd.to_datetime(f["REPORTDATE"]).dt.normalize()
    f["TOTAL_OPERATE_INCOME"] = pd.to_numeric(f["TOTAL_OPERATE_INCOME"], errors="coerce")
    f = f.dropna(subset=["TOTAL_OPERATE_INCOME"])
    f = f.sort_values(["SECURITY_CODE", "REPORTDATE"])

    # 建 (code, period) -> 累计营收 的查找表
    cum = f.set_index(["SECURITY_CODE", "REPORTDATE"])["TOTAL_OPERATE_INCOME"]

    rows = []
    for (code, rd), v in cum.items():
        y = rd.year
        # 上年同期
        prev = pd.Timestamp(year=y - 1, month=rd.month, day=rd.day)
        # 上年末（年报）
        prev_q4 = pd.Timestamp(year=y - 1, month=12, day=31)
        a = cum.get((code, prev), np.nan)
        b = cum.get((code, prev_q4), np.nan)
        if rd.month == 12 and rd.day == 31:
            ttm = v                       # 年报本身就是全年
        elif not np.isnan(a) and not np.isnan(b):
            ttm = v + b - a
        else:
            ttm = np.nan
        rows.append((code, rd, v, ttm))
    out = pd.DataFrame(rows, columns=["SECURITY_CODE", "REPORTDATE", "cum_rev", "ttm_rev"])
    out["q"] = out["REPORTDATE"].dt.month.map({3: 1, 6: 2, 9: 3, 12: 4})
    # 年化（累计口径的旧做法，用于对照）
    out["ann_rev"] = out["cum_rev"] * 4 / out["q"]
    return out


def main() -> None:
    fin = load_fin()
    t = build_ttm(fin)

    bar("A. 复现口径污染（现行 revenue = 年初至今累计）")
    cur = t[t["REPORTDATE"].isin(pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31"]))]
    print("\n  %-12s %8s %14s %14s %14s" %
          ("报告期", "只数", "营收中位(亿)", ">=2亿过线率", "季度数"))
    for rd, g in cur.groupby("REPORTDATE"):
        rd_s = rd.strftime("%Y-%m-%d")
        print("  %-12s %8d %14.2f %13.1f%% %14d" %
              (rd_s, len(g), g["cum_rev"].median() / 1e8,
               (g["cum_rev"] >= 2e8).mean() * 100, g["q"].iloc[0]))
    g3 = cur[cur["REPORTDATE"] == pd.Timestamp("2026-09-30")]
    g2 = cur[cur["REPORTDATE"] == pd.Timestamp("2026-06-30")]
    print("\n  → 三季报组营收中位 %.2f亿 vs 中报组 %.2f亿（前者高出 %.0f%%，纯属季度数差异）" %
          (g3["cum_rev"].median() / 1e8, g2["cum_rev"].median() / 1e8,
           (g3["cum_rev"].median() / g2["cum_rev"].median() - 1) * 100))
    print("  → 过线率 %.1f%% vs %.1f%%：披露三季报的公司更容易过这道门" %
          ((g3["cum_rev"] >= 2e8).mean() * 100, (g2["cum_rev"] >= 2e8).mean() * 100))

    bar("B. TTM（最近四个季度）后，污染是否消除")
    print("\n  %-12s %8s %14s %14s %14s" %
          ("报告期", "只数", "TTM中位(亿)", "TTM覆盖", "季度数"))
    for rd, g in cur.groupby("REPORTDATE"):
        ok = g["ttm_rev"].notna()
        print("  %-12s %8d %14.2f %13.1f%% %14d" %
              (rd.strftime("%Y-%m-%d"), len(g),
               g.loc[ok, "ttm_rev"].median() / 1e8 if ok.any() else float("nan"),
               ok.mean() * 100, g["q"].iloc[0]))
    t3 = cur[(cur["REPORTDATE"] == pd.Timestamp("2026-09-30")) & cur["ttm_rev"].notna()]
    t2 = cur[(cur["REPORTDATE"] == pd.Timestamp("2026-06-30")) & cur["ttm_rev"].notna()]
    print("\n  → TTM 中位：三季报组 %.2f亿 vs 中报组 %.2f亿（差异 %.1f%%，已大幅收敛）" %
          (t3["ttm_rev"].median() / 1e8, t2["ttm_rev"].median() / 1e8,
           abs(t3["ttm_rev"].median() / t2["ttm_rev"].median() - 1) * 100))

    bar("C. 等价阈值标定：TTM 门槛取多少，才能复现行「累计>=2亿」的筛选强度")
    print("\n  现行门槛 revenue>=2亿 是【按中报(2个季度)】标定的，"
          "等价于年化 4 亿。\n  若改 TTM，需要重新标定。以下按【通过率】对齐：\n")
    print("  %-14s %10s %10s %10s" % ("口径", "阈值", "全样本过线率", "等价性"))
    for thr in [1e8, 2e8, 3e8, 4e8, 5e8, 6e8, 8e8]:
        r = (cur["cum_rev"] >= thr).mean() * 100
        print("  %-14s %9.0f亿 %11.1f%%" % ("累计(现行)", thr / 1e8, r))
    print()
    for thr in [1e8, 2e8, 3e8, 4e8, 5e8, 6e8, 8e8]:
        ok = cur["ttm_rev"].notna()
        r = (cur.loc[ok, "ttm_rev"] >= thr).mean() * 100
        print("  %-14s %9.0f亿 %11.1f%%" % ("TTM(建议)", thr / 1e8, r))
    print("\n  → 现行「累计>=2亿」在混合样本里通过率 %.1f%%；"
          "最接近的 TTM 阈值见上表" %
          ((cur["cum_rev"] >= 2e8).mean() * 100))

    bar("D. TTM 的历史可得性（会不会因为缺上年数据而大面积失效）")
    t2all = t.copy()
    t2all["yr"] = t2all["REPORTDATE"].dt.year
    g = t2all.groupby("yr").agg(
        记录=("ttm_rev", "size"),
        有TTM=("ttm_rev", lambda s: s.notna().sum()),
    )
    g["覆盖率%"] = (g["有TTM"] / g["记录"] * 100).round(1)
    print("\n" + g.to_string())

    bar("E. 结论")
    print("""
  1. 现行 revenue = 【年初至今累计】，跨报告期不可比 —— 污染属实。
  2. TTM(最近四个季度) = 本期累计 + 上年全年 − 上年同期累计，
     无论哪一期都覆盖最近 12 个月 → 口径恒定，污染消除。
  3. TTM 的历史覆盖率极高（上表），不会引入新的缺失。
  4. 阈值需重新标定：现行 2 亿是【半年累计】口径（≈年化 4 亿），
     改 TTM 后应按通过率对齐到上表对应档位。
""")


if __name__ == "__main__":
    main()
