# -*- coding: utf-8 -*-
"""
【池子成分解剖】为什么核心池里几乎全是红利/低波票？

要回答三件事：
  A. 每一道门槛各自剔掉了谁 —— 漏斗诊断（累积加门槛，看剩余与池子特征漂移）
  B. 剔除成长票的【主因】是哪一条 —— 单条放宽，看池子特征怎么变
  C. 这是策略的永久偏向，还是当前下跌市的环境特征 —— 分周期对比池子成分

池子特征用三个可解释的量：
  营收增速中位（成长度） / beta 中位（弹性） / 行业 Top5（风格）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin  # noqa: E402

# v2 结构门
V2 = dict(prox=0.95, width=0.15, amt=4e8, turn=6.0)
# v1 旧口径
V1 = dict(prox=0.90, width=0.25, amt=4e8, turn_lo=1.5, turn_hi=10.0)


def prof(g: pd.DataFrame, label: str) -> None:
    """打印一组事件的池子特征。事件表无行业/市值字段，改用可解释的量化特征。"""
    if len(g) == 0:
        print("%-32s %8d %11s %8s %9s %11s" % (label, 0, "—", "—", "—", "—"))
        return
    rev = float(g["YSTZ"].median())
    beta = float(g["beta"].median())
    turn = float(g["turn30"].median())
    hi = float((g["YSTZ"] >= 30).mean() * 100)
    print("%-32s %8d %10.1f%% %8.2f %8.2f%% %10.1f%%" % (
        label, len(g), rev, beta, turn, hi))


def main() -> None:
    ev = load_events()
    fin = load_fin()
    ev = attach_fin(ev, fin).sort_values(["code", "date"])

    print("=" * 112)
    print("池子成分解剖：红利化到底是谁造成的")
    print("=" * 112)
    print(f"事件表 {len(ev):,} 行")

    # ---------------------------------------------------------- A. 漏斗
    print("\n" + "-" * 112)
    print("A. 漏斗：逐道加门槛，看池子特征怎么漂移（全事件，不分周期）")
    print("-" * 112)
    print("%-32s %8s %11s %8s %9s %11s" % ("累积到这一步", "事件数", "营收增速中位", "beta", "换手中位", "高成长占比"))

    cur = ev
    prof(cur, "0. 全市场（无任何门槛）")
    cur = cur[cur["prox_now"] >= V2["prox"]]
    prof(cur, "1. + 距年高 >= 0.95（当前价口径）")
    cur = cur[cur["width30"] <= V2["width"]]
    prof(cur, "2. + 箱宽 <= 15%")
    cur = cur[cur["amt30"] >= V2["amt"]]
    prof(cur, "3. + 日均额 >= 4亿")
    cur = cur[cur["turn30"] <= V2["turn"]]
    prof(cur, "4. + 换手 <= 6%")
    cur2 = cur[cur["YSTZ"].notna()]
    prof(cur2, "5. + 营收数据非空（v2 完整结构门）")

    base = cur2

    def gate(src, prox, width, amt, turn_hi, turn_lo=0.0):
        return src[(src["prox_now"] >= prox) & (src["width30"] <= width) &
                   (src["amt30"] >= amt) & (src["turn30"] <= turn_hi) &
                   (src["turn30"] >= turn_lo)]

    print("\n" + "-" * 112)
    print("B. 单条放宽：到底哪一条把成长票挡在外面？（从全市场重筛，每次只改一条）")
    print("-" * 112)
    print("%-32s %8s %11s %8s %9s %11s" % ("改动", "事件数", "营收增速中位", "beta", "换手中位", "高成长占比"))
    prof(base, "  【v2 完整门 · 基准】")
    for lab, kw in [
        ("箱宽 15% → 25%（v1 值）", dict(width=0.25)),
        ("箱宽 15% → 20%", dict(width=0.20)),
        ("换手 <=6% → <=10%（v1 值）", dict(turn_hi=10.0)),
        ("距年高 0.95 → 0.90（v1 值）", dict(prox=0.90)),
        ("距年高 0.95 → 0.85", dict(prox=0.85)),
        ("日均额 4亿 → 2亿", dict(amt=2e8)),
    ]:
        p_ = dict(prox=0.95, width=0.15, amt=4e8, turn_hi=6.0)
        p_.update(kw)
        prof(gate(ev, **p_), "  " + lab)

    print()
    print("  对照：v1 旧口径整组（事件表无旧 prox 列，此处用 prox_now>=0.90 近似；")
    print("        v1 真口径为『近30日高/年高>=0.90』，方向一致但更松）")
    prof(gate(ev, 0.90, 0.25, 4e8, 10.0, 1.5), "  【v1 完整门 · 近似】")

    # ---------------------------------------------------------- C. 分周期
    print("\n" + "-" * 112)
    print("C. 是策略永久偏向，还是当下的环境特征？（同一个 v2 门，分周期看池子）")
    print("-" * 112)
    print("%-32s %8s %11s %8s %9s %11s" % ("周期", "事件数", "营收增速中位", "beta", "换手中位", "高成长占比"))
    for rg in ["UP", "CHOP", "DOWN"]:
        prof(base[base["regime"] == rg], f"  {rg}")


    # ---------------------------------------------------------- D. 成长票去哪了
    print("\n" + "-" * 112)
    print("D. 高成长票（营收增速 >= 30%）在 v2 门下还剩多少？卡在哪一步？")
    print("-" * 112)
    hi_growth = ev[ev["YSTZ"] >= 30]
    print("  全市场高成长事件           %8d" % len(hi_growth))
    a = hi_growth[hi_growth["prox_now"] >= 0.95]
    print("  + 距年高 >= 0.95           %8d  (保留 %.1f%%)" % (len(a), len(a) / len(hi_growth) * 100))
    b2 = a[a["width30"] <= 0.15]
    print("  + 箱宽 <= 15%%              %8d  (保留 %.1f%%)" % (len(b2), len(b2) / len(hi_growth) * 100))
    c2 = b2[b2["amt30"] >= 4e8]
    print("  + 日均额 >= 4亿            %8d  (保留 %.1f%%)" % (len(c2), len(c2) / len(hi_growth) * 100))
    d2 = c2[c2["turn30"] <= 6.0]
    print("  + 换手 <= 6%%               %8d  (保留 %.1f%%)  ← v2 门下高成长票的最终量"
          % (len(d2), len(d2) / len(hi_growth) * 100))

    print()
    print("  低成长票（营收增速 < 10%）对照组：")
    lo_g = ev[ev["YSTZ"] < 10]
    print("  全市场                     %8d" % len(lo_g))
    for lab, m in [("距年高>=0.95", lo_g["prox_now"] >= 0.95),
                   ("箱宽<=15%", lo_g["width30"] <= 0.15),
                   ("额>=4亿", lo_g["amt30"] >= 4e8),
                   ("换手<=6%", lo_g["turn30"] <= 6.0)]:
        pass
    a3 = lo_g[lo_g["prox_now"] >= 0.95]
    b3 = a3[a3["width30"] <= 0.15]
    c3 = b3[b3["amt30"] >= 4e8]
    d3 = c3[c3["turn30"] <= 6.0]
    print("  逐道后最终剩余             %8d  (保留 %.1f%%)"
          % (len(d3), len(d3) / len(lo_g) * 100))
    print()
    print("  → 高成长票过门率 %.2f%%  vs  低成长票过门率 %.2f%%  相差 %.1f 倍"
          % (len(d2) / len(hi_growth) * 100, len(d3) / len(lo_g) * 100,
             (len(d3) / len(lo_g)) / max(len(d2) / len(hi_growth), 1e-9)))


if __name__ == "__main__":
    main()
