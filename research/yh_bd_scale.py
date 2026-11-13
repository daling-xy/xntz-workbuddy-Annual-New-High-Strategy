"""base_days 度量口径与 F 门阈值的关系研究 · 2026-11-12

背景：用户问「把区间撑到 25.8%，是箱体宽度还是波动率？是相对波动率吗？能否放宽？」

需要回答三件事：
  ① base_days 里的 20% 到底是什么统计量（绝对箱宽 / 波动率 / 相对振幅）
  ② 放宽「度量阈值」（20% → 25% / 30% / 35%）会怎样
  ③ 改成「相对口径」（除以该股自身同长度历史中位宽度）会怎样

关键结论预判：在「F 门只有上限」的用法下，放宽度量阈值 = 收紧结果
（更容易判定为横盘 → base_days 变大 → 更多票突破上限被剔），方向是反的。

口径定义
--------
绝对口径 abs(thr)：max{ k∈[5,60] : width_k ≤ thr }              （现用，thr=20%）
相对口径 rel(ratio)：max{ k∈[5,60] : width_k ≤ ratio × med_k }  （med_k = 该股过去一年
                     同长度 k 日滚动宽度的中位数，即「这只票在 k 日尺度上的典型震荡幅度」）

※ 注意：ratio 的语义与绝对口径的 20% 完全不同。绝对口径 20% 是「箱宽不超过 20 个点」，
  相对口径 ratio=1.0 才是「箱宽不超过自己平时的水平」。两者不能照搬同一个数字。

相对口径的实测问题（见 ⑤）：比值在突破瞬间会从 0.8 跳到 6.2，
因为它被自己前期行情污染（德尔刚挖完 47% 的坑，250 日历史中位宽度被坑撑坏），
导致 base_days 在关键节点剧烈跳变、不稳定。
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_screen import PARAMS                                    # noqa: E402
from research.yh_shape_precise import cond_R, cond_C, cond_H     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "bd_scale_20261112.txt")

KMIN, KMAX = 5, 60
HIST = 250          # 相对口径的分母回看窗口

_LINES = []


def P(s=""):
    print(s)
    _LINES.append(s)


def w_at(c, i, k):
    seg = c[i - k + 1: i + 1]
    return float(seg.max() / seg.min() - 1.0)


def bd_abs(c, i, thr):
    best = 0
    for k in range(KMIN, KMAX + 1):
        if k > i + 1:
            break
        if w_at(c, i, k) <= thr:
            best = k
        else:
            break
    return best


def bd_rel(c, i, ratio):
    """width_k ≤ ratio × 该股过去一年同长度 k 日滚动宽度的中位数。"""
    best = 0
    lo = max(0, i - HIST)
    for k in range(KMIN, KMAX + 1):
        if k > i + 1:
            break
        hist = [w_at(c, j, k) for j in range(max(k - 1, lo), i)
                if j - k + 1 >= 0]
        if len(hist) < 60:
            break
        med = float(np.median(hist))
        if med <= 0:
            break
        if w_at(c, i, k) <= ratio * med:
            best = k
        else:
            break
    return best


def main():
    k = pickle.load(open(os.path.join(ROOT, "cache", "kline_320.pkl"), "rb"))
    d = pickle.load(open(os.path.join(ROOT, "cache", "panel_320_0.pkl"), "rb"))
    if "prox_now" not in d:
        d["prox_now"] = 1.0 + d["dd_from_high"]
    p = dict(PARAMS)
    d["w20"] = d["width20"]
    d["ago250"] = d["hi250_ago"]
    d = d[~d["name"].astype(str).str.contains("ST|退", na=False)]

    base = ((d["prox_now"] >= 0.88) & (d["width10"] <= 0.10)
            & (d["contraction"] <= 0.45) & (d["width30"] <= 0.50)
            & (d["amt30"] >= 2e8))
    rch = (base & cond_R(d).fillna(False) & cond_C(d).fillna(False)
           & cond_H(d).fillna(False))
    cand = d[rch]

    P("=" * 104)
    P("base_days 度量口径研究 · 2026-11-12 · 基准日 %s" % d["date"].max())
    P("=" * 104)
    P("")
    P("① 三种统计量的区别（同一只票、同一天，不是同一个数）")
    P("")
    P("  width30      = 近30日收盘 max/min − 1        → 绝对箱体宽度（%），现用于 VCP 门")
    P("  atr_pct      = ATR14 / 收盘价                → 波动率（%），日频振幅")
    P("  width_pct    = 近30日宽度在过去一年滚动30日宽度分布中的分位 → 相对振幅（0~1，越小越窄）")
    P("  base_days    = 最大 k，使近 k 日收盘宽度 ≤ 20% → 横盘持续天数")
    P("  ※ base_days 内部的 20% 是【绝对箱宽】，不是波动率，也不是相对振幅。")
    P("")

    rows = []
    for code, r in cand.iterrows():
        bars = k.get(code)
        if not bars:
            continue
        c = np.array([x["close"] for x in bars], float)
        i = len(c) - 1
        w250 = w_at(c, i, min(250, i + 1))
        rows.append(dict(
            code=code, name=r["name"], board=r["board"], score=r.get("score", np.nan),
            bd20=bd_abs(c, i, 0.20), bd25=bd_abs(c, i, 0.25),
            bd30=bd_abs(c, i, 0.30), bd35=bd_abs(c, i, 0.35),
            rel07=bd_rel(c, i, 0.70), rel10=bd_rel(c, i, 1.00),
            rel12=bd_rel(c, i, 1.20),
            w250=w250 * 100,
            inblk=not (pd.notna(r["board"]) and r["board"] in p["exclude_industries"]),
        ))
    t = pd.DataFrame(rows)
    t["w250"] = t["w250"].round(1)

    P("② 过 R/C/H 的 %d 只在各口径下的 base_days（w250 = 该股自身 250 日箱宽）" % len(t))
    P("")
    P("%-9s %-8s %5s %5s %5s %5s | %5s %5s %5s | %7s" % (
        "名称", "行业", "20%", "25%", "30%", "35%", "r.7", "r1.0", "r1.2", "w250"))
    P("-" * 104)
    for _, r in t.sort_values("bd20", ascending=False).iterrows():
        flag = "" if r["inblk"] else "  [行业黑名单]"
        P("%-9s %-8s %5d %5d %5d %5d | %5d %5d %5d | %6.1f%%%s" % (
            str(r["name"])[:8], str(r["board"])[:8], r["bd20"], r["bd25"], r["bd30"], r["bd35"],
            r["rel07"], r["rel10"], r["rel12"], r["w250"], flag))

    P("")
    P("③ 放宽「度量阈值」（上限固定 35 日）—— 注意方向")
    P("")
    P("  %-10s %8s %8s   %s" % ("度量口径", "过F", "过行业", "名单"))
    P("  " + "-" * 96)
    for col, lab in (("bd20", "abs 20%（现用）"), ("bd25", "abs 25%"),
                     ("bd30", "abs 30%"), ("bd35", "abs 35%"),
                     ("rel07", "rel 0.7"), ("rel10", "rel 1.0"), ("rel12", "rel 1.2")):
        m = t[col] <= 35
        ps = t[m & t["inblk"]]
        P("  %-10s %8d %8d   %s" % (lab, int(m.sum()), len(ps),
                                    "、".join(ps["name"].tolist()) or "（空）"))

    P("")
    P("④ 放宽「上限」（度量固定 abs 20%）—— 看断层在哪")
    P("")
    P("  %-10s %8s %8s   %s" % ("上限", "过F", "过行业", "名单"))
    P("  " + "-" * 96)
    for cap in (20, 25, 30, 35, 40, 45, 50, 59, 60):
        m = t["bd20"] <= cap
        ps = t[m & t["inblk"]]
        P("  %-10s %8d %8d   %s" % ("≤ %d 日" % cap, int(m.sum()), len(ps),
                                    "、".join(ps["name"].tolist()) or "（空）"))

    P("")
    P("⑤ 德尔股份平台期逐日（2026-10-09 ~ 10-27）：口径差异的直接体现")
    P("")
    bars = k.get("300473")
    c = np.array([x["close"] for x in bars], float)
    dates = [x["date"] for x in bars]
    P("  %-12s %6s %6s %6s | %6s %6s   %s" % ("日期", "20%", "25%", "30%", "r1.0", "r1.2", "近20日宽度"))
    P("  " + "-" * 88)
    for i in range(len(c)):
        if not ("2026-10-09" <= dates[i] <= "2026-10-27"):
            continue
        P("  %-12s %6d %6d %6d | %6d %6d   %6.1f%%" % (
            dates[i], bd_abs(c, i, 0.20), bd_abs(c, i, 0.25), bd_abs(c, i, 0.30),
            bd_rel(c, i, 1.00), bd_rel(c, i, 1.20), w_at(c, i, 20) * 100))

    P("")
    P("⑥ 结论")
    P("")
    P("  1. 25.8% 是【绝对箱体宽度】：德尔 20 日窗口 30.45~38.31 的 max/min−1，")
    P("     即 20 日窗口往回够到了修复段低点 30.45，宽度从 19 日的 14.4% 跳到 25.8%。")
    P("  2. 放宽度量阈值（20%→25%/30%）会让池子【变小】，不是变大：")
    P("     更容易判为横盘 → base_days 变大 → 更多票突破上限 35 被剔（6→3→3）。")
    P("     极端情况 abs35% 时 31 只全部 bd=60，池子归零。")
    P("  3. 放宽上限（35→50 日）零变化：31 只的 bd20 在 36~59 之间完全无票（断层），")
    P("     35 与 59 等价；直到 60 才把 17 只低波动红利股全部放回（池子 6→16）。")
    P("  4. 改相对口径会【让 F 门失效】：rel 0.7 时 31 只全部 bd≤35（过 F 31 只 → 过行业 16 只），")
    P("     与不加 F 门完全一样。原因是分母（过去一年同长度中位宽度）在当前普遍上涨的市况下")
    P("     远大于当前宽度，比值多数落在 0~1，算出的 base_days 普遍偏小。")
    P("     更致命的是不稳定：德尔平台期 rel 1.0 逐日为 5→12→7→8→9→0→0→0→0，")
    P("     在 10-20 后归零 —— 因为 250 日历史窗口被前面 47% 的坑污染，中位宽度失真。")
    P("  5. 根本原因：F 门只有上限，绝对口径下 base_days 与波动率负相关")
    P("     （bd=60 的 17 只 w250 多在 16~45%；bd<35 的 6 只 w250 多在 29~65%），")
    P("     所以 F 门实际上在兼任「剔除低波动」——这正是 β 0.80→0.91、大亏率升高的来源。")
    P("  6. 建议：不改。20% 绝对口径在「只用上限」的用法下偏【宽容】：")
    P("     高波动票的箱宽容易超 20%，算出的 base_days 偏小 → 更容易过上限 → 不会被误杀。")
    P("     这正是用户「小票波动大也该选入」诉求想要的方向，无需再放宽。")
    P("     唯一前提：若将来要【加下限】（例如要求已横盘≥15 日），必须同时改相对口径，")
    P("     否则高波动小票的 base_days 会被系统性低估、全部卡在下限之外。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES))
    P("")
    P("已落盘：%s" % OUT)


if __name__ == "__main__":
    main()
