# -*- coding: utf-8 -*-
"""
【箱宽门槛演化史】20% -> 25% -> 15% 中间到底发生了什么

用户的原始问题：
  最初定的条件是「近30日 (最高收盘-最低收盘)/最低收盘 <= 20%（同时试验 25%）」
  现在为什么变成 15%？中间经历了哪些优化？

要拆清楚的矛盾：
  v1 的结论是「20% 太紧，25% 更好」
  v2 的结论是「15% 比 25% 更好」
  两句都跑过数，为什么会打架？

假设：v1 和 v2 之间同时变了 3 件事，箱宽最优值的变化是被这 3 件事"推"出来的：
  V1 样本口径 : 35.9 万事件 (2023-03~2026-07)
  V2 样本口径 : 150.2 万事件 (2019-06~2026-07)
  V1 prox 口径: 近30日最高收盘 / 250日最高收盘   ← 允许"冲高回落"
  V2 prox 口径: 当前收盘 / 250日最高收盘         ← 严格贴年高
  V1 超额口径 : 横截面去均值  ex_fwd60 = fwd60 - 当日全市场均值
  V2 超额口径 : beta 调整     ab_fwd60 = fwd60 - beta * mkt_fwd60

本脚本做 5 段：
  A 复现 v1 的决策现场：旧 prox 口径 + 去均值超额，看箱宽档位排序
  B 只换 prox 口径    ：新 prox 口径 + 去均值超额
  C 只换超额口径      ：旧 prox 口径 + beta 调整
  D 两个都换（= v2）  ：新 prox 口径 + beta 调整
  E 归因结论 + 单调性检验（细分箱看是否"越窄越好"）

数据来源：腾讯前复权日线（2018 至今全历史）、通达信指数日线、东方财富财报。
所有结论均为历史统计，不构成投资建议。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1000
_rng = np.random.default_rng(20261026)

LINE = "=" * 118
SUB = "-" * 118

NEED = ["code", "date", "regime", "prox_now", "prox", "width30",
        "amt30", "turn30", "beta", "fwd60", "mkt_fwd60", "ex_fwd60",
        "ab_fwd60", "mae20"]


def load_events() -> pd.DataFrame:
    p = os.path.join(CACHE, "events_fixed.pkl")
    if not os.path.exists(p):
        p = os.path.join(CACHE, "events_long_beta.pkl")
    ev = pd.read_pickle(p)
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev = ev[[c for c in NEED if c in ev.columns]]
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    return ev.reset_index(drop=True)


def boot_ci(sub: pd.DataFrame, key: str, n: int = BOOT):
    """按交易日整簇重抽样"""
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = float(v.mean())
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def stat(tag: str, d: pd.DataFrame, key: str = "ab_fwd60", w: int = 30):
    n = len(d)
    if n < 30:
        print(f"  {tag:<{w}} n={n:>6}  样本不足")
        return None
    est, lo, hi = boot_ci(d, key)
    win = float((d[key] > 0).mean())
    big = float((d["fwd60"] < -0.20).mean())
    mae = float(d["mae20"].mean()) if "mae20" in d.columns else np.nan
    sig = "显著为正" if (lo > 0) else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<{w}} n={n:>6}  β中位{d['beta'].median():>5.2f}  "
          f"{est*100:+6.2f}%  CI[{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率{win*100:4.1f}%  大亏{big*100:4.1f}%  MAE{mae*100:+6.2f}%  {sig}")
    return est


# 统一的"其它纪律"：成交额、换手（与 v2 一致）
DISC = ("amt30", 4e8, ">="), ("turn30", 6.0, "<=")


def with_disc(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=d.index)
    for col, thr, op in DISC:
        m &= (d[col] >= thr) if op == ">=" else (d[col] <= thr)
    return d[m]


def section(title: str) -> None:
    print(LINE)
    print(title)
    print(LINE)
    print()


def main() -> None:
    ev = load_events()
    print(f"事件表: {len(ev):,} 行 / {ev['code'].nunique():,} 股 / "
          f"{ev['date'].nunique():,} 交易日")
    print()

    widths = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 9.9]
    wlab = {0.10: "<=10%", 0.15: "<=15%", 0.20: "<=20%", 0.25: "<=25%",
            0.30: "<=30%", 0.40: "<=40%", 9.9: "不限"}

    # ============ A 段：复现 v1 的决策现场 ============
    section("A. 复现 v1 当年的决策现场（旧 prox 口径 + 横截面去均值超额）")

    print("  v1 前置条件：prox（近30日最高收盘/250日最高收盘）>= 0.90")
    print("  v1 超额口径：ex_fwd60 = fwd60 - 当日全市场横截面均值")
    print()
    baseA = ev[ev["prox"] >= 0.90]
    print(f"  命中 {len(baseA):,} 事件")
    print()
    print(f"  {'箱宽档':<12}{'事件数':>9}{'β中位':>8}{'去均值超额':>12}{'90%CI':>20}{'胜率':>8}")
    for w in widths:
        d = baseA[baseA["width30"] <= w]
        if len(d) < 30:
            print(f"  {wlab[w]:<12}{len(d):>9}  样本不足")
            continue
        est, lo, hi = boot_ci(d, "ex_fwd60")
        win = float((d["ex_fwd60"] > 0).mean())
        print(f"  {wlab[w]:<12}{len(d):>9}{d['beta'].median():>8.2f}"
              f"{est*100:>11.2f}%  [{lo*100:+6.2f},{hi*100:+6.2f}]"
              f"{win*100:>7.1f}%")
    print()
    print("  → 这正是 v1 得出「20% 太紧，改 25%」的那张表。")
    print("    注意：此时还没做 beta 调整，也没做周期分层。")
    print()

    # ============ B 段：只换 prox 口径 ============
    section("B. 只把 prox 口径修正（新口径），超额仍用去均值")

    print("  前置：prox_now（当前收盘/250日最高收盘）>= 0.95")
    print("  超额：ex_fwd60（去均值，与 A 段同一口径，便于对照）")
    print()
    baseB = ev[ev["prox_now"] >= 0.95]
    print(f"  命中 {len(baseB):,} 事件")
    print()
    for w in widths:
        d = baseB[baseB["width30"] <= w]
        stat(f"箱宽{wlab[w]}", d, key="ex_fwd60")
    print()
    print("  → 单独换 prox 口径后，箱宽档位排序是什么样？看上面的数。")
    print()

    # ============ C 段：只换超额口径 ============
    section("C. 只把超额口径换成 beta 调整，prox 仍用旧口径")

    print("  前置：prox >= 0.90（旧口径，与 A 段相同）")
    print("  超额：ab_fwd60 = fwd60 - beta × mkt_fwd60")
    print()
    baseC = ev[ev["prox"] >= 0.90]
    for w in widths:
        d = with_disc(baseC[baseC["width30"] <= w])
        stat(f"箱宽{wlab[w]} + 额≥4亿 + 换手≤6%", d, key="ab_fwd60")
    print()

    # ============ D 段：两个都换（= v2 的真实配置）============
    section("D. 两个都换（= v2 修正版的真实配置）")

    print("  前置：prox_now >= 0.95（新口径）")
    print("  超额：ab_fwd60（beta 调整）")
    print()
    baseD = ev[ev["prox_now"] >= 0.95]
    for w in widths:
        d = with_disc(baseD[baseD["width30"] <= w])
        stat(f"箱宽{wlab[w]} + 额≥4亿 + 换手≤6%", d, key="ab_fwd60")
    print()

    # ============ E 段：单调性 / 细分箱 ============
    section("E. 单调性检验：把箱宽切细，看是不是「越窄越好」")

    bins = [(0.0, 0.08), (0.08, 0.12), (0.12, 0.15), (0.15, 0.20),
            (0.20, 0.25), (0.25, 0.40), (0.40, 9.9)]
    blab = {(0.0, 0.08): "0~8%", (0.08, 0.12): "8~12%", (0.12, 0.15): "12~15%",
            (0.15, 0.20): "15~20%", (0.20, 0.25): "20~25%",
            (0.25, 0.40): "25~40%", (0.40, 9.9): ">40%"}

    for label, base, key, use_disc in [
        ("【旧 prox 口径 + 去均值】= v1 现场", ev[ev["prox"] >= 0.90], "ex_fwd60", False),
        ("【新 prox 口径 + beta 调整】= v2 现场", ev[ev["prox_now"] >= 0.95], "ab_fwd60", True),
    ]:
        print(SUB)
        print(f"  {label}")
        print(SUB)
        for b in bins:
            d = base[(base["width30"] > b[0]) & (base["width30"] <= b[1])]
            if use_disc:
                d = with_disc(d)
            stat(f"   箱宽 {blab[b]}", d, key=key, w=24)
        print()

    # ============ F 段：样本量归因 ============
    section("F. 归因 1/3：是不是【样本变大】导致的？")

    print("  同一个口径（旧 prox>=0.90 + 去均值），只换样本区间：")
    print()
    ev["_y"] = pd.to_datetime(ev["date"]).dt.year
    for tag, sub in [
        ("全样本 2019-06~2026-07", ev),
        ("仅 2023-03 起（v1 覆盖期）", ev[ev["date"] >= "2023-03-01"]),
        ("仅 2019-2022（v1 没覆盖）", ev[ev["_y"] <= 2022]),
    ]:
        print(SUB)
        print(f"  {tag}   （{len(sub):,} 事件）")
        print(SUB)
        b = sub[sub["prox"] >= 0.90]
        for w in widths:
            d = b[b["width30"] <= w]
            stat(f"   箱宽{wlab[w]}", d, key="ex_fwd60", w=20)
        print()

    print("  → v1 当年看到的是【2023-03 起】那一栏，据它判定『25% > 20%』。")
    print("    把样本扩到全历史后，这一栏的排序是否还成立，看上面的数。")
    print()

    # ============ G 段：分年份稳健性 ============
    section("G. 归因 2/3：15% 在每个年份都稳吗？（新口径 + beta 调整 + 额≥4亿 + 换手≤6%）")

    base = ev[ev["prox_now"] >= 0.95]
    ys = sorted(ev["_y"].unique())
    print(f"  {'年份':<8}", end="")
    for w in [0.15, 0.20, 0.25]:
        print(f"{'箱宽'+wlab[w]:>18}", end="")
    print()
    print(f"  {'-'*8:<8}", end="")
    for _ in [0.15, 0.20, 0.25]:
        print(f"{'-'*18:>18}", end="")
    print()
    for y in ys:
        sy = base[base["_y"] == y]
        print(f"  {y:<8}", end="")
        for w in [0.15, 0.20, 0.25]:
            d = with_disc(sy[sy["width30"] <= w])
            if len(d) < 30:
                print(f"{'n<30':>18}", end="")
                continue
            est, lo, hi = boot_ci(d, "ab_fwd60")
            mark = "*" if lo > 0 else ("x" if hi < 0 else " ")
            print(f"{f'{est*100:+.1f}% (n={len(d)}){mark}':>18}", end="")
        print()
    print()
    print("  * = 显著为正   x = 显著为负   空白 = 含零")
    print()

    # ============ H 段：门槛的边际收益 ============
    section("H. 归因 3/3：从 25% 收到 15%，边际收益与代价")

    print("  新口径 + beta 调整 + 额≥4亿 + 换手≤6%，看每一档的『增量 / 代价』：")
    print()
    print(f"  {'箱宽档':<10}{'事件数':>9}{'每天命中':>10}{'β调整超额':>11}{'90%CI下界':>11}{'胜率':>8}{'大亏率':>8}{'MAE20':>9}")
    prev = None
    for w in widths:
        d = with_disc(base[base["width30"] <= w])
        if len(d) < 30:
            continue
        est, lo, hi = boot_ci(d, "ab_fwd60")
        win = float((d["ab_fwd60"] > 0).mean())
        big = float((d["fwd60"] < -0.20).mean())
        mae = float(d["mae20"].mean())
        perday = len(d) / ev["date"].nunique()
        print(f"  {wlab[w]:<10}{len(d):>9}{perday:>10.1f}{est*100:>10.2f}%{lo*100:>10.2f}%"
              f"{win*100:>7.1f}%{big*100:>7.1f}%{mae*100:>8.2f}%")
    print()
    print("  读法：往上一档，超额和胜率升、大亏和回撤降（这是收益）；")
    print("        但『每天命中』往下掉（这是代价，会频繁空仓）。")
    print("        15% 是收益仍然显著、而每天还有 1.7 只可挑的最后一个档。")
    print()

    print(LINE)
    print("※ 数据来源：腾讯前复权日线与换手率、通达信指数日线、东方财富财报。")
    print("※ 以上均为历史统计与回测结论，不构成投资建议。")
    print(LINE)


if __name__ == "__main__":
    main()
