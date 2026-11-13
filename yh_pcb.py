# -*- coding: utf-8 -*-
"""坑后平台（Post-Crash Base, PCB）形态判定

用户口径（2026-11-12）：
  「德尔 2025-05 ~ 2026-06 一直横盘震荡，2026-06 ~ 2026-09 挖了个大坑，
    然后震荡 20-30 天之后上涨。不用看到那么前面，震荡 20-30 天这里就足够选出来。
    前面长期震荡、即使有个大坑（比较快就修复），也算加分项。」

德尔实测：坑前高 38.30（2026-05）→ 坑底 20.23（07-30，−47.2%）→
修复 34.80（09-21，用时 37 个交易日）→ 平台 09-21~10-23 共 21 个交易日、宽度 14.4%。

定位：**卫星池**，不并入主池。
  全样本实证（output/pcb_20261112.txt）：
    PCB 单独 ab +0.49% / 大亏 15.9% / β 1.02
    主池 v2.4   ab +1.07% / 大亏 11.5% / β 0.79
    VCP∪PCB    ab +0.61% / 大亏 12.7%，配对 bootstrap −0.17pp CI[−0.61,+0.32] 含零
  → 形态有效但弱于主池、且波动更大，故独立成池并列展示，不做主池门。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

PCB_PARAMS = {
    "pcb_lookback": 120,      # 坑底回看窗口（交易日）
    "pcb_lookback_high": 120,  # 坑底之前找坑前高点的回看窗口
    "pcb_min_dd": 0.25,       # 坑深下限
    "pcb_repair": 0.90,       # 修复到坑前高点的比例
    "pcb_min_days": 10,       # 修复后横盘最少交易日
    "pcb_max_days": 45,       # 修复后横盘最多交易日
    "pcb_plat_w": 0.30,       # 平台期高低宽度上限
    "pcb_break": 0.95,        # 不破位：收盘 >= 平台最低 × 该值
}


def pcb_at(closes: np.ndarray, i: int, p: Optional[Dict] = None) -> Dict | None:
    """在索引 i 处判定是否构成「坑后平台」。命中返回特征字典，否则 None。

    顺序很重要：**先找坑底，再往坑底之前找坑前高点**。
    反过来（先找窗口最高点）会把坑前高点挤出窗口 —— 德尔股份的坑前高 38.30
    在 2026-05，距 10-23 已 100+ 个交易日，120 日窗口内的最高点反而变成修复后的
    9-29（38.31），会被「高点太近」误杀。
    """
    q = dict(PCB_PARAMS)
    if p:
        q.update({k: v for k, v in p.items() if k in PCB_PARAMS})

    lookback = q["pcb_lookback"]
    lookback_high = q["pcb_lookback_high"]

    lo = max(0, i - lookback + 1)
    seg = closes[lo:i + 1]
    if len(seg) < 40:
        return None

    # 1) 坑底：窗口内最低收盘
    l_rel = int(np.argmin(seg))
    low_abs = lo + l_rel
    low = float(seg[l_rel])
    if i - low_abs < 15:          # 坑底太近，「修复 + 横盘」还没发生
        return None

    # 2) 坑前高点：坑底之前 lookback_high 日内的最高收盘
    h0 = max(0, low_abs - lookback_high)
    if low_abs - h0 < 30:
        return None
    h_rel = int(np.argmax(closes[h0:low_abs]))
    h_abs = h0 + h_rel
    h = float(closes[h_abs])

    dd = 1.0 - low / h
    if dd < q["pcb_min_dd"]:
        return None

    # 3) 修复：坑底之后首次回到坑前高 × repair
    thr = h * q["pcb_repair"]
    after = closes[low_abs:i + 1]
    hit = np.nonzero(after >= thr)[0]
    if len(hit) == 0:
        return None
    rep = low_abs + int(hit[0])

    # 4) 平台时长与宽度
    days = i - rep
    if days < q["pcb_min_days"] or days > q["pcb_max_days"]:
        return None
    plat = closes[rep:i + 1]
    pw = float(plat.max() / plat.min() - 1.0)
    if pw > q["pcb_plat_w"]:
        return None
    if closes[i] < plat.min() * q["pcb_break"]:
        return None

    # 5) 加分项：长期箱体宽度（锚坑前高点，不能锚修复日——否则坑本身会被算进去）
    b0 = max(0, h_abs - 250)
    box = closes[b0:h_abs + 1]
    box_w = float(box.max() / box.min() - 1.0) if len(box) > 60 else np.nan

    return {
        "pcb_hit": 1,
        "pcb_dd": dd,                        # 坑深
        "pcb_days": days,                    # 修复后已横盘交易日数
        "pcb_plat_w": pw,                    # 平台宽度
        "pcb_repair_spd": rep - low_abs,     # 坑底到修复用时（越快越 V）
        "pcb_box_w": box_w,                  # 修复前长期箱体宽度（加分项）
        "pcb_low": low,
        "pcb_high": h,
    }
