# -*- coding: utf-8 -*-
"""坑后平台（Post-Crash Base, PCB）形态探针

用户口径（2026-11-12）：
  「德尔股份 2025-05 ~ 2026-06 一直横盘震荡，2026-06 ~ 2026-09 挖了一个大坑，
    然后震荡 20-30 天之后上涨。不用看到那么前面，震荡 20-30 天这里就足够选出来。
    前面长期震荡过、即使有个大坑（比较快就修复），也算加分项。」

把这段话拆成可计算的四件事：
  1) 坑：过去 LOOKBACK 日内出现过一次 >= MIN_DD 的回撤（滚动最高 → 其后最低）
  2) 修复：坑底之后收盘价回到坑前最高 × REPAIR 以上
  3) 平台：修复日到基准日之间 >= MIN_DAYS 且 <= MAX_DAYS 个交易日，
          期间高低宽度 <= PLAT_W（横盘而非二次拉升）
  4) 不破位：基准日收盘 >= 平台最低 × 0.95

加分项（不打门，只标注）：
  box_width  —— 修复日之前 250 日的价格区间宽度（长期横盘过 = 窄）
  repair_spd —— 坑底到修复用了多少交易日（修复越快越像 V 型）

用法：
  python research/yh_pcb_probe.py            # 德尔逐日诊断 + 两个基准日全市场扫描
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

CACHE = os.path.join(BASE, "cache")

LOOKBACK = 120     # 回看窗口（交易日），够覆盖 2026-05 的坑前高点
MIN_DD = 0.25      # 坑深至少 25%
REPAIR = 0.90      # 修复到坑前高点的 90% 视为修复
MIN_DAYS = 10      # 修复后横盘至少 10 个交易日
MAX_DAYS = 45      # 最多 45 个交易日（再长就不是「坑后平台」而是长期箱体）
PLAT_W = 0.30      # 平台期高低宽度 <= 30%


def pcb_at(closes: np.ndarray, i: int, lookback: int = LOOKBACK,
           min_dd: float = MIN_DD, repair: float = REPAIR,
           min_days: int = MIN_DAYS, max_days: int = MAX_DAYS,
           plat_w: float = PLAT_W, lookback_high: int = LOOKBACK) -> dict | None:
    """在索引 i 处判定是否构成「坑后平台」。命中返回特征字典，否则 None。

    顺序很重要：**先找坑底，再往坑底之前找坑前高点**。
    反过来（先找窗口最高点）会把坑前高点挤出窗口 —— 德尔股份的坑前高 38.30
    在 2026-05，距离 10-23 已有 100+ 个交易日，120 日窗口内的最高点反而变成
    修复日之后的 9-29（38.31），导致「高点太近」被误杀。
    """
    lo = max(0, i - lookback + 1)
    seg = closes[lo:i + 1]
    if len(seg) < 40:
        return None

    # 1) 坑底：窗口内最低收盘
    l_rel = int(np.argmin(seg))
    low_abs = lo + l_rel
    low = seg[l_rel]
    # 坑底必须离基准日足够远，「修复 + 横盘」才有时间发生
    if i - low_abs < 15:
        return None
    # 2) 坑前高点：坑底之前 lookback_high 日内的最高收盘
    h0 = max(0, low_abs - lookback_high)
    if low_abs - h0 < 30:
        return None
    h_rel = int(np.argmax(closes[h0:low_abs]))
    h_abs = h0 + h_rel
    h = float(closes[h_abs])
    dd = 1.0 - low / h
    if dd < min_dd:
        return None

    rep = None
    for j in range(low_abs, i + 1):
        if closes[j] >= h * repair:
            rep = j
            break
    if rep is None:
        return None

    days = i - rep
    if days < min_days or days > max_days:
        return None

    plat = closes[rep:i + 1]
    pw = plat.max() / plat.min() - 1.0
    if pw > plat_w:
        return None
    if closes[i] < plat.min() * 0.95:
        return None

    # 长期箱体（加分项）：坑前高点之前 250 日的区间宽度。
    #   必须锚在「坑前高点」而不是「修复日」——锚修复日会把坑本身算进去，
    #   德尔股份会得到 89% 的假宽度；锚坑前高点是 43%，才对得上
    #   「2025-05 ~ 2026-05 一直在 26.7~38.3 之间横盘」这个事实。
    b0 = max(0, h_abs - 250)
    box = closes[b0:h_abs + 1]
    box_w = (box.max() / box.min() - 1.0) if len(box) > 60 else np.nan

    return {
        "pcb_hit": 1,
        "pcb_dd": dd,                    # 坑深
        "pcb_days": days,                # 修复后已横盘天数
        "pcb_plat_w": pw,                # 平台宽度
        "pcb_repair_spd": rep - low_abs, # 坑底到修复的交易日数（越快越 V）
        "pcb_box_w": box_w,              # 修复前长期箱体宽度（加分项）
        "pcb_low": low,
        "pcb_high": h,
    }


def probe_one(code: str, kmap: dict, dates: list[str]) -> pd.DataFrame:
    bars = kmap.get(code)
    if not bars:
        return pd.DataFrame()
    closes = np.array([b["close"] for b in bars], dtype=float)
    dmap = {b["date"]: i for i, b in enumerate(bars)}
    rows = []
    for ds in dates:
        i = dmap.get(ds)
        if i is None:
            continue
        r = pcb_at(closes, i) or {}
        rows.append({"date": ds, "close": closes[i], **r})
    return pd.DataFrame(rows)


def scan(asof: int, kmap: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """全市场扫描：对面板里每只票，在其 asof 索引处判 PCB。"""
    recs = []
    for code, row in panel.iterrows():
        bars = kmap.get(code)
        if not bars:
            continue
        i = len(bars) - 1 - asof
        if i < 60:
            continue
        closes = np.array([b["close"] for b in bars], dtype=float)
        r = pcb_at(closes, i)
        if r:
            r.update({"code": code, "name": row["name"], "close": closes[i],
                      "prox_now": row.get("prox_now", np.nan),
                      "w10": row.get("width10", np.nan),
                      "amt30": row.get("amt30", np.nan) / 1e8,
                      "board": row.get("board", "")})
            recs.append(r)
    return pd.DataFrame(recs)


def main():
    kmap = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))

    print("=" * 108)
    print("一、德尔股份（300473）逐日形态状态")
    print("=" * 108)
    bars = kmap["300473"]
    closes = np.array([b["close"] for b in bars], dtype=float)
    print(f"{'日期':<12}{'收盘':>8}{'坑深':>8}{'横盘天数':>9}{'平台宽':>8}{'修复用时':>9}{'长期箱宽':>9}")
    print("-" * 108)
    for i in range(len(bars) - 40, len(bars)):
        r = pcb_at(closes, i)
        if r:
            print(f"{bars[i]['date']:<12}{closes[i]:>8.2f}{r['pcb_dd']:>8.1%}"
                  f"{r['pcb_days']:>9d}{r['pcb_plat_w']:>8.1%}"
                  f"{r['pcb_repair_spd']:>9d}{r['pcb_box_w']:>9.1%}")
        else:
            print(f"{bars[i]['date']:<12}{closes[i]:>8.2f}{'—':>8}{'—':>9}{'—':>8}{'—':>9}{'—':>9}")

    for asof, tag in ((4, "2026-11-05"), (13, "2026-10-23")):
        pf = os.path.join(CACHE, f"panel_320_{asof}.pkl")
        if not os.path.exists(pf):
            print(f"\n[跳过] {tag}：面板未缓存")
            continue
        panel = pickle.load(open(pf, "rb"))
        if "prox_now" not in panel.columns:
            panel["prox_now"] = 1.0 + panel["dd_from_high"]
        res = scan(asof, kmap, panel)
        print()
        print("=" * 108)
        print(f"二、全市场 PCB 扫描 · 基准日 {tag}（asof={asof}）命中 {len(res)} 只")
        print("=" * 108)
        if len(res):
            res = res.sort_values("pcb_days")
            out = res[["code", "name", "close", "prox_now", "w10", "amt30",
                       "pcb_dd", "pcb_days", "pcb_plat_w", "pcb_box_w", "board"]].copy()
            for c in ("pcb_dd", "pcb_plat_w", "pcb_box_w", "w10"):
                out[c] = (out[c] * 100).round(1)
            out["prox_now"] = out["prox_now"].round(3)
            out["amt30"] = out["amt30"].round(1)
            print(out.to_string(index=False))
            hit = "300473" in set(res["code"])
            print(f"\n  德尔股份命中：{'是' if hit else '否'}")

            # 组合口径：PCB + 距年高 0.88 + 成交额 2 亿（模拟接入主池的样子）
            comb = res[(res["prox_now"] >= 0.88) & (res["amt30"] >= 2.0)].copy()
            comb = comb.sort_values(["pcb_days", "pcb_plat_w"])
            print()
            print(f"  ── 叠加 prox≥0.88 & 额≥2亿 后剩 {len(comb)} 只 ──")
            if len(comb):
                o2 = comb[["code", "name", "close", "prox_now", "w10", "amt30",
                           "pcb_dd", "pcb_days", "pcb_plat_w", "pcb_box_w", "board"]].copy()
                for c in ("pcb_dd", "pcb_plat_w", "pcb_box_w", "w10"):
                    o2[c] = (o2[c] * 100).round(1)
                print(o2.to_string(index=False))


if __name__ == "__main__":
    main()
