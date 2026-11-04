# -*- coding: utf-8 -*-
"""
【行业剔除清单审计 · 信息性】—— 了解黑名单在当前 v2 口径下的实际作用与代价

定位（2026-11-02 用户确认）：
  行业剔除是【原始策略要求】（原描述写明剔除基建/银行/交通/电信四类），
  属于策略定义的组成部分，不做质疑与增删。
  本脚本回答的问题是「这张清单在 v2 下实际过滤了什么、代价多大」，
  【不】回答「该不该剔」，输出不驱动任何参数变更。

方法：
  在【结构门】内（prox_now>=0.95 + 箱宽<=15% + amt30>=4亿 + turn30<=6%，不含营收门槛），
  按行业分组，用区块 bootstrap（区块=60交易日）算各行业的 beta 调整 60 日超额与 90%CI。
  同时给出「黑名单合计」vs「非黑名单合计」的对照。

局限（必须说明）：
  行业分类取自【当前快照】的 BOARD_NAME，回溯历史存在轻微前视偏差
  （个股可能因重组/主业变更而改行业）。行业分类本身稳定性较高，影响有限。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events                    # noqa: E402
from yh_blockboot_all import make_dmap, daily, block_boot, HB  # noqa: E402
from yh_screen import PARAMS, PARAMS_BY_REGIME          # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_board_map() -> dict:
    """从当前截面面板取 code -> board（申万二级行业名）"""
    p = os.path.join(CACHE, "panel_320_0.pkl")
    df = pd.read_pickle(p)
    df.index = df.index.astype(str).str.zfill(6)
    return df["board"].to_dict()


def main() -> None:
    ev = load_events()
    p = dict(PARAMS)
    p.update(PARAMS_BY_REGIME["DOWN"])          # 结构门三周期统一，取任一套即可
    ex = set(p["exclude_industries"])

    bmap = load_board_map()
    ev["board"] = ev["code"].astype(str).str.zfill(6).map(bmap)

    # ---- 结构门（不含营收门槛）----
    g = ev[(ev["prox_now"] >= p["prox_now_min"]) &
           (ev["width30"] <= p["width30_max"]) &
           (ev["amt30"] >= p["amt30_min"]) &
           (ev["turn30"] <= p["turn30_max"])]
    print("=" * 104)
    print("行业剔除清单审计 · v2 当前口径")
    print("=" * 104)
    print("结构门：prox_now>=%.2f  箱宽<=%.0f%%  amt30>=%.0f亿  turn30<=%.0f%%  （营收门槛：全周期关闭）"
          % (p["prox_now_min"], p["width30_max"] * 100, p["amt30_min"] / 1e8, p["turn30_max"]))
    print("回测区间：%s ~ %s ；门内事件 %d 条" % (g["date"].min(), g["date"].max(), len(g)))
    print("指标：beta 调整 60 日超额（ab_fwd60）；区块 bootstrap 区块=%d 交易日，90%%CI" % HB)
    print()

    dmap = make_dmap(ev)
    covered = g[g["board"].notna()]
    print("有行业标签的门内事件：%d / %d（%.1f%%）"
          % (len(covered), len(g), 100 * len(covered) / max(len(g), 1)))
    print()

    # ---- 1. 黑名单各行业逐一 ----
    print("-" * 104)
    print("【A】当前黑名单 9 类 —— 在 v2 口径下各自的门内表现")
    print("-" * 104)
    print("  %-14s %8s %8s %6s %8s %10s %20s %10s" %
          ("行业", "事件", "交易日", "区块", "日均票", "β中位", "β调整超额", "判定"))
    rows = []
    for ind in sorted(ex):
        d = covered[covered["board"] == ind]
        if len(d) < 10:
            print("  %-14s %8d   (样本不足 10，无法检验)" % (ind, len(d)))
            rows.append((ind, len(d), np.nan, np.nan, np.nan))
            continue
        s = daily(d)
        pt, lo, hi, nb = block_boot(s, dmap)
        per = float(d.groupby("date").size().mean())
        beta = float(d["beta"].median())
        verdict = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
        print("  %-14s %8d %8d %6d %8.3f %10.2f %+8.2f%% CI[%+6.2f,%+6.2f] %10s" %
              (ind, len(d), d["date"].nunique(), nb, per, beta,
               pt * 100, lo * 100, hi * 100, verdict))
        rows.append((ind, len(d), pt, lo, hi))

    # ---- 2. 黑名单 vs 非黑名单 ----
    print()
    print("-" * 104)
    print("【B】黑名单合计 vs 非黑名单合计（配对区块检验）")
    print("-" * 104)
    d_ex = covered[covered["board"].isin(ex)]
    d_ok = covered[~covered["board"].isin(ex)]
    for nm, d in (("黑名单合计", d_ex), ("非黑名单合计", d_ok)):
        s = daily(d)
        pt, lo, hi, nb = block_boot(s, dmap)
        verdict = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
        print("  %-12s 事件%7d  区块%3d  日均票%6.2f  %+7.2f%% CI[%+6.2f,%+6.2f]  %s"
              % (nm, len(d), nb, d.groupby("date").size().mean(),
                 pt * 100, lo * 100, hi * 100, verdict))

    from yh_blockboot_all import paired_block_boot
    pt, lo, hi, nb = paired_block_boot(d_ex, d_ok, dmap)
    verdict = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
    print("  差值（黑名单 − 非黑名单）：%+7.2fpp CI[%+6.2f,%+6.2f]  配对区块%3d  → %s"
          % (pt * 100, lo * 100, hi * 100, nb, verdict))
    print("  读法：差值显著为负 = 黑名单确实更差（支持剔除）；含零 = v2 口径下无证据支持剔除")

    # ---- 3. 全市场行业排名：哪些非黑名单行业反而最差？ ----
    print()
    print("-" * 104)
    print("【C】门内事件数 >= 30 的全部行业，按 β 调整超额升序（找「该剔未剔」的候选）")
    print("-" * 104)
    stat = []
    for ind, d in covered.groupby("board"):
        if len(d) < 30:
            continue
        s = daily(d)
        pt, lo, hi, nb = block_boot(s, dmap)
        stat.append((ind, len(d), nb, pt, lo, hi, ind in ex))
    stat.sort(key=lambda x: x[3])
    print("  %-14s %8s %6s %20s %10s %8s" %
          ("行业", "事件", "区块", "β调整超额", "判定", "在黑名单"))
    for ind, n, nb, pt, lo, hi, in_ex in stat:
        verdict = "显著为正" if lo > 0 else ("显著为负" if hi < 0 else "含零")
        print("  %-14s %8d %6d %+8.2f%% CI[%+6.2f,%+6.2f] %10s %8s" %
              (ind, n, nb, pt * 100, lo * 100, hi * 100, verdict, "★是" if in_ex else ""))
    print()
    print("  说明：样本量普遍很小，单行业 CI 绝大多数含零 —— 本表用于「看方向」，不做显著性裁决。")


if __name__ == "__main__":
    main()
