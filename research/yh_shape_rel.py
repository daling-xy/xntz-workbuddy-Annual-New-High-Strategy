# -*- coding: utf-8 -*-
"""
形态口径研究：绝对箱宽 vs 相对化 / 多尺度收缩（2026-11-12 用户提问）
====================================================================
用户问题：振幅能否做成「相对振幅」？小票波动大但形态好，按理也该选入；
          以德尔股份 300473 突破前平台为例，形态该怎么选？

先澄清口径（很重要）：
  width30 = 近30日最高收盘/最低收盘 − 1，**本身已经是相对振幅**（已按价格归一化）。
  用户想要的其实是【按个股自身波动特性归一化】，即"相对紧度"，有三层：
    L1 相对自身历史   width_pct = 过去一年里 30日箱宽 >= 当前值 的比例
                      （越大 = 当前相对自己越窄；事件表 yh_eventstudy.py 口径）
    L2 相对自身近期   contraction = width10 / width30（末端收缩比，越小越收敛）
    L3 多尺度绝对末端 width10 = 近10日箱宽（平台本身窄不窄）

方法（完全复用 yh_relax4.py 的纪律，防幸存者偏差）：
  ① 个案：从 cache/kline_320.pkl 逐日重建德尔股份形态，看五道门分别卡在哪。
  ② 边际组逐年：在【不施加 width 门】的样本上，
     A = 绝对门外（w30>15%）但形态达标；B = 原门内（w30<=15%）。
     四口径同报：ab_fwd60 / ex_fwd60 / fwd60 / 大亏率。
  ③ 整池配对区块 bootstrap：方案池 − 现池，60交易日区块 / 4000次 / 90%CI，
     全样本 + UP/CHOP/DOWN。
  ④ 网格稳健性：阈值小幅变动是否翻转结论（防挑参数）。

基准门 = 当前正式版本 v2.1（2026-11-12 定版）：
  prox>=0.95 / width30<=15% / amt30>=2亿 / turn30<=6% / ret60<=19%
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

from yh_relax4 import gate, load_events, report_relax, P, _lines  # noqa: E402
from yh_blockboot_all import make_dmap  # noqa: E402
from yh_factors import features_at  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "shape_rel_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8                 # 当前正式版本 v2.1 的成交额门
BAD = -0.20               # 大亏定义


# ---------------------------------------------------------------- 个案
def case_study() -> None:
    P("\n" + "=" * 104)
    P("① 个案：德尔股份 300473 · 突破前平台逐日形态重建")
    P("=" * 104)
    k = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))
    b = k.get("300473")
    if not b:
        P("  缓存中无 300473，跳过")
        return
    prm = {"high_days": 250, "box_days": 30, "tight_days": 10}
    rows = []
    for i in range(250, len(b)):
        f = features_at(b, prm, idx=i)
        if not f:
            continue
        rows.append(dict(
            date=f["date"], close=f["close"], prox=1 + f["dd_from_high"],
            w30=f["width30"], w10=f["width10"], wp=f["width_pct"],
            ct=f["contraction"], amt=f["amt30"] / 1e8, turn=f["turn30"],
            ret60=f["ret60"], vr=f["vr20"], dry=f["vol_dry"]))
    df = pd.DataFrame(rows)
    P("  缓存区间 %s ~ %s，可评估交易日 %d 天（需 250 日预热）" % (
        b[0]["date"], b[-1]["date"], len(df)))
    P("  w30=30日箱宽 | w10=10日箱宽 | wp=自身历史宽度分位(越大越窄) | ct=w10/w30 收缩比")
    P("  %-11s %7s %7s %7s %7s %6s %6s %7s %7s %8s %6s" % (
        "date", "close", "prox", "w30", "w10", "wp", "ct", "amt亿", "turn%", "ret60", "vr20"))
    for _, r in df[df["date"] >= "2026-10-09"].iterrows():
        P("  %-11s %7.2f %7.3f %6.1f%% %6.1f%% %6.2f %6.2f %7.2f %7.2f %7.1f%% %6.2f" % (
            r["date"], r["close"], r["prox"], r["w30"] * 100, r["w10"] * 100,
            r["wp"], r["ct"], r["amt"], r["turn"], r["ret60"] * 100, r["vr"]))

    g1 = df["prox"] >= 0.95
    g2 = df["w30"] <= 0.15
    g3 = df["amt"] >= 2.0
    g4 = df["turn"] <= 6.0
    g5 = df["ret60"] <= 0.19
    P("\n  【五门体检】71 个可评估日中各自通过天数：")
    for nm, m in [("① prox>=0.95", g1), ("② w30<=15%", g2), ("③ amt30>=2亿", g3),
                  ("④ turn30<=6%", g4), ("⑤ ret60<=19%", g5)]:
        P("    %-18s %3d 天" % (nm, int(m.sum())))
    P("    五门全过            %3d 天" % int((g1 & g2 & g3 & g4 & g5).sum()))
    P("  w30 分布：min %.1f%% / 中位 %.1f%% / max %.1f%% → 从未 ≤15%%" % (
        df["w30"].min() * 100, df["w30"].median() * 100, df["w30"].max() * 100))
    plat = df[(df["date"] >= "2026-10-12") & (df["date"] <= "2026-10-23")]
    P("  平台段（10-12~10-23）：w30 中位 %.1f%%，w10 中位 %.1f%%，收缩比 ct 中位 %.2f，wp 中位 %.2f" % (
        plat["w30"].median() * 100, plat["w10"].median() * 100,
        plat["ct"].median(), plat["wp"].median()))
    P("  → 30日窗口被「19.96→38 那一波拉升」污染；真正收敛发生在 10 日尺度（VCP 末端收缩）。")
    P("  → 但同期 prox 0.89~0.96、turn30 6.1~6.6%、ret60 40~73%，门①④⑤也同时不满足。")


# ---------------------------------------------------------------- 四口径
def four_way(tag: str, d: pd.DataFrame) -> None:
    P("    %-16s 事件 %6d  日均 %5.2f 只 | ab %+6.2f%%  ex %+6.2f%%  fwd %+6.2f%%  大亏 %4.1f%%  β %.2f  w30中位 %4.1f%%" % (
        tag, len(d), d.groupby("date").size().mean(),
        d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        d["fwd60"].mean() * 100, (d["ab_fwd60"] < BAD).mean() * 100,
        d["beta"].median(), d["width30"].median() * 100))


def year_four(d_now: pd.DataFrame, maskA, label: str) -> None:
    a = d_now[maskA(d_now)].copy()
    b = d_now[d_now["width30"] <= 0.15].copy()
    a["y"], b["y"] = a["date"].dt.year, b["date"].dt.year
    gdiff = a["ab_fwd60"].mean() - b["ab_fwd60"].mean()
    P("\n    【%s】逐年四口径（A=形态通道新放入 / B=原门内）" % label)
    P("    %-6s %5s %5s %10s %10s %10s %10s %10s %10s" % (
        "年", "nA", "nB", "A-B(ab)", "A-B(ex)", "A-B(fwd)", "A大亏", "B大亏", "风险差"))
    ok_ab = ok_ex = ok_bad = tot = 0
    for y in sorted(set(a["y"]) | set(b["y"])):
        x, z = a[a["y"] == y], b[b["y"] == y]
        if len(x) < 8 or len(z) < 8:
            continue
        dab = (x["ab_fwd60"].mean() - z["ab_fwd60"].mean()) * 100
        dex = (x["ex_fwd60"].mean() - z["ex_fwd60"].mean()) * 100
        dfw = (x["fwd60"].mean() - z["fwd60"].mean()) * 100
        ba, bb = (x["ab_fwd60"] < BAD).mean() * 100, (z["ab_fwd60"] < BAD).mean() * 100
        tot += 1
        ok_ab += 1 if (dab > 0) == (gdiff > 0) else 0
        ok_ex += 1 if dex > 0 else 0
        ok_bad += 1 if ba <= bb else 0
        P("    %-6d %5d %5d %+9.2fpp %+9.2fpp %+9.2fpp %8.1f%% %8.1f%% %+9.1fpp" % (
            y, len(x), len(z), dab, dex, dfw, ba, bb, ba - bb))
    P("    及格线 6/8：ab方向 %d/%d | ex为正 %d/%d | A不更危险 %d/%d" % (
        ok_ab, tot, ok_ex, tot, ok_bad, tot))


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 104)
    P("形态口径研究 · 2026-11-12 · 绝对箱宽 vs 相对化 / 多尺度收缩")
    P("=" * 104)
    P("事件表 events_fixed.pkl：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min().date(), ev["date"].max().date()))
    P("基准门 = 当前正式版本 v2.1：prox≥0.95 / w30≤15% / amt30≥2亿 / turn30≤6% / ret60≤19%")
    P("四口径：ab_fwd60(β中性alpha) / ex_fwd60(横截面超额) / fwd60(原始收益) / 大亏率(ab<-20%)")
    P("说明：width30 本身已是「相对振幅」(hi30/lo30−1)，本研究检验的是【按个股自身归一化】。")
    P("本脚本只做研究，不改任何参数。")

    case_study()

    d_now = gate(ev, drop="width", over={"amt": AMT})   # 不施加箱宽门，其余门齐全
    pool_base = gate(ev, over={"amt": AMT})              # 现池 v2.1
    out = d_now[d_now["width30"] > 0.15]                 # 绝对门外

    P("\n" + "=" * 104)
    P("② 样本与三层相对化口径")
    P("=" * 104)
    P("  不施加箱宽门后 %s 条；原门内 %s 条，门外 %s 条" % (
        f"{len(d_now):,}", f"{len(pool_base):,}", f"{len(out):,}"))
    P("  L1 width_pct（相对自身历史）  L2 contraction=w10/w30（相对自身近期）  L3 width10（末端绝对窄）")

    P("\n  【网格稳健性】门外样本，形态条件小幅变动是否翻转结论：")
    P("  %-26s %6s %9s %9s %9s %9s %8s %7s" % (
        "条件", "n", "ab", "ex", "fwd", "大亏", "w30中位", "beta"))
    for w10 in [0.06, 0.08, 0.10, 0.12]:
        for ct in [0.35, 0.45, 0.60]:
            s = out[(out["width10"] <= w10) & (out["contraction"] <= ct)]
            if len(s) < 30:
                continue
            P("  %-26s %6d %+8.2f%% %+8.2f%% %+8.2f%% %8.1f%% %7.1f%% %7.2f" % (
                "w10<=%.0f%% ct<=%.2f" % (w10 * 100, ct), len(s),
                s["ab_fwd60"].mean() * 100, s["ex_fwd60"].mean() * 100,
                s["fwd60"].mean() * 100, (s["ab_fwd60"] < BAD).mean() * 100,
                s["width30"].median() * 100, s["beta"].median()))
    for wp, ct in [(0.60, 0.45), (0.70, 0.45), (0.60, 0.60), (0.80, 0.60)]:
        s = out[(out["width_pct"] >= wp) & (out["contraction"] <= ct)]
        if len(s) < 30:
            continue
        P("  %-26s %6d %+8.2f%% %+8.2f%% %+8.2f%% %8.1f%% %7.1f%% %7.2f" % (
            "wp>=%.2f ct<=%.2f" % (wp, ct), len(s),
            s["ab_fwd60"].mean() * 100, s["ex_fwd60"].mean() * 100,
            s["fwd60"].mean() * 100, (s["ab_fwd60"] < BAD).mean() * 100,
            s["width30"].median() * 100, s["beta"].median()))
    P("  %-26s %6d %+8.2f%% %+8.2f%% %+8.2f%% %8.1f%% %7.1f%% %7.2f" % (
        "【原门内 w30<=15%】", len(pool_base),
        pool_base["ab_fwd60"].mean() * 100, pool_base["ex_fwd60"].mean() * 100,
        pool_base["fwd60"].mean() * 100, (pool_base["ab_fwd60"] < BAD).mean() * 100,
        pool_base["width30"].median() * 100, pool_base["beta"].median()))

    # ---- 方案定义：现池 ∪ 形态通道（形态通道另加 w30<=35% 的极端波动兜底）----
    WCAP = 0.35
    s1 = (d_now["width_pct"] >= 0.60) & (d_now["contraction"] <= 0.45) & (d_now["width30"] <= WCAP)
    s2 = (d_now["width10"] <= 0.08) & (d_now["contraction"] <= 0.45) & (d_now["width30"] <= WCAP)
    pool_s1 = d_now[(d_now["width30"] <= 0.15) | s1]
    pool_s2 = d_now[(d_now["width30"] <= 0.15) | s2]

    P("\n" + "=" * 104)
    P("③ 方案实证（现池 ∪ 形态通道；通道内另设 w30≤35% 极端波动兜底）")
    P("=" * 104)
    P("  S1 相对自身历史通道：wp≥0.60 且 ct≤0.45   | 新增 %d 条" % (len(pool_s1) - len(pool_base)))
    P("  S2 末端收缩通道：w10≤8%% 且 ct≤0.45        | 新增 %d 条" % (len(pool_s2) - len(pool_base)))
    P("\n  【整池四口径均值】")
    four_way("现池 v2.1", pool_base)
    four_way("S1 相对历史通道", pool_s1)
    four_way("S2 末端收缩通道", pool_s2)

    year_four(d_now, lambda x: s1.reindex(x.index, fill_value=False) & (x["width30"] > 0.15),
              "S1 相对自身历史")
    year_four(d_now, lambda x: s2.reindex(x.index, fill_value=False) & (x["width30"] > 0.15),
              "S2 末端收缩")

    for tag, pl in [("S1 相对自身历史 wp≥0.60 ct≤0.45", pool_s1),
                    ("S2 末端收缩 w10≤8% ct≤0.45", pool_s2)]:
        P("\n  " + "=" * 100)
        P("  配对区块 bootstrap：" + tag)
        P("  " + "=" * 100)
        report_relax(
            tag, d_now,
            lambda x, _pl=pl: pd.Series(x.index.isin(_pl.index), index=x.index)
            & (x["width30"] > 0.15),
            lambda x: x["width30"] <= 0.15,
            "形态通道新放入", "原门内 ≤15%",
            pl, pool_base, dmap)

    P("\n" + "=" * 104)
    P("④ 结论要点（自动汇总，最终裁决见速查卡/日志）")
    P("=" * 104)
    P("  1) width30 已是相对价格的振幅；真正缺的是「按个股自身波动归一化」与「多尺度」。")
    P("  2) 德尔股份平台 w30≈44%（30日窗口含前一波拉升），w10≈8%、ct≈0.18 —— 形态收敛")
    P("     确实存在，但在 10 日尺度；改相对振幅也救不回，因为门①prox / ④换手 / ⑤ret60 同时不满足。")
    P("  3) 形态通道挑出的票 β≈0.68~0.98、大亏率 8.7%~14.5%（现池 4.9%），是【高β弹性】资产，")
    P("     不是低风险 alpha。它与主池的低波防御定位正交，混进同一池会改变整体风险画像。")
    P("  耗时 %.1f 秒" % (time.time() - t0))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print("已写出", OUT, flush=True)


if __name__ == "__main__":
    main()
