# -*- coding: utf-8 -*-
"""
年度新高选股策略 - 特征/因子层

所有因子都在「任意历史时点 t」上可重算（只取 bars[: idx+1]），
因此同一套代码既用于当日选股，也用于历史回放（参数校准 / 事件研究回测）。

约定：
  bars 为按日期升序的前复权日线 dict 列表，至少含 close / high / low / volume / amount / turnover
  idx  为评估时点下标（含）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ 基础滚动量
def _closes(bars: Sequence[dict]) -> np.ndarray:
    return np.array([b["close"] for b in bars], dtype=float)


def rolling_extreme(x: np.ndarray, win: int, kind: str) -> np.ndarray:
    """滚动极值，长度与 x 对齐，前 win-1 个用可用窗口的极值填充"""
    n = len(x)
    out = np.empty(n, dtype=float)
    f = np.max if kind == "max" else np.min
    for i in range(n):
        s = max(0, i - win + 1)
        out[i] = f(x[s:i + 1])
    return out


def true_range_atr(bars: Sequence[dict], win: int = 14) -> np.ndarray:
    """ATR（真实波幅均值），返回绝对点数数组"""
    n = len(bars)
    tr = np.empty(n, dtype=float)
    for i in range(n):
        h, l = bars[i]["high"], bars[i]["low"]
        pc = bars[i - 1]["close"] if i > 0 else bars[i]["open"]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    atr = np.empty(n, dtype=float)
    for i in range(n):
        s = max(0, i - win + 1)
        atr[i] = tr[s:i + 1].mean()
    return atr


def ma(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        s = max(0, i - win + 1)
        out[i] = x[s:i + 1].mean()
    return out


# ------------------------------------------------------------------ 单时点因子
def features_at(bars: Sequence[dict], p: Dict, idx: Optional[int] = None) -> Dict[str, float]:
    """
    计算评估时点的全部技术结构因子。
    p 需要: high_days(250), box_days(30), tight_days(10)
    """
    if idx is None:
        idx = len(bars) - 1
    if idx < 250:
        return {}

    hd, bd, td = p["high_days"], p["box_days"], p["tight_days"]
    sub = bars[: idx + 1]
    c = _closes(sub)
    amt = np.array([b.get("amount", 0.0) for b in sub], dtype=float)
    vol = np.array([b.get("volume", 0.0) for b in sub], dtype=float)
    turn = np.array([b.get("turnover", 0.0) for b in sub], dtype=float)

    last = c[idx]
    win250 = c[idx - hd + 1: idx + 1]
    hi250 = float(win250.max())
    hi30 = float(c[idx - bd + 1: idx + 1].max())
    lo30 = float(c[idx - bd + 1: idx + 1].min())
    hi20 = float(c[idx - 20 + 1: idx + 1].max())
    lo20 = float(c[idx - 20 + 1: idx + 1].min())
    hi10 = float(c[idx - td + 1: idx + 1].max())
    lo10 = float(c[idx - td + 1: idx + 1].min())

    width30 = hi30 / lo30 - 1.0 if lo30 > 0 else np.nan
    width20 = hi20 / lo20 - 1.0 if lo20 > 0 else np.nan
    width10 = hi10 / lo10 - 1.0 if lo10 > 0 else np.nan
    prox = hi30 / hi250 if hi250 > 0 else np.nan          # 原版条件2：30日高 / 一年高
    dd_from_high = last / hi250 - 1.0                      # 当前价距一年最高收盘的回撤
    pos_in_box = (last - lo30) / (hi30 - lo30) if hi30 > lo30 else 1.0
    contraction = width10 / width30 if width30 and width30 > 0 else np.nan

    # --- 250日最高收盘「新鲜度」（2026-11-12 v2.3 新增）---
    # hi250_ago = 从最新一根往回数、【最近一次】达到 250 日最高收盘的根数（1 = 今日创年高）。
    # 用途：区分「刚创新高的有效平台」与「回到半年前同一价位的双顶/下跌中继」——
    #   南方传媒/双环传动/九州通 的 250 日最高点分别在 199/194/191 日前，
    #   价格虽回到高位区，但那是半年多前的旧高点，不是新鲜突破。
    # 实现：反向取 argmax，等价于取「最后一次出现最大值」的位置。
    _rev = win250[::-1]
    hi250_ago = int(np.argmax(_rev)) + 1

    # --- 横盘持续天数 base_days（2026-11-12 v2.6 新增）---
    # base_days = 最大的 k（5~60），使过去 k 日收盘区间宽度 (max/min-1) <= 20%。
    # 用途：width10/20/30 只度量「最近 k 天有多窄」，不度量「已经窄了多久」。
    #   实测 24.7 万事件分箱：fwd60 在 30~39 天见顶 +2.77%，60+ 回落到 +2.41%；
    #   横盘 <10 天 ab −4.78%、大亏率 30.1%（垃圾区）。
    # 用法：只加上限 base_days <= 35，剔掉「横 60 天以上还趴着」的低波动红利股
    #   （长电/核电/唐山港/华电/国电/美的/九州通 一类），不设下限 ——
    #   下限已被 VCP（w10<=10% & w30<=50%）隐含满足，且设 20 会误杀德尔
    #   （德尔平台期 base_days 峰值仅 19，因 20 日窗口够到 09-18 的修复段）。
    base_days = 0
    for _k in range(5, 61):
        if _k > idx + 1:
            break
        _seg = c[idx - _k + 1: idx + 1]
        if _seg.max() / _seg.min() - 1.0 <= 0.20:
            base_days = _k
        else:
            break

    atr_pct = float(true_range_atr(sub, 14)[idx] / last) if last > 0 else np.nan

    # --- 量能 ---
    amt30 = float(amt[idx - bd + 1: idx + 1].mean())
    amt10 = float(amt[idx - td + 1: idx + 1].mean())
    turn30 = float(turn[idx - bd + 1: idx + 1].mean())
    # 当日量比（Minervini 语义：事件日量能 / 近20日日均量能）
    # 2026-10-28 新增，D3 诊断用（与箱宽/prox 相关性 <0.2，无共线性）
    amt_now = float(amt[idx])
    ma20_amt = float(amt[idx - 19: idx + 1].mean())
    vr20 = amt_now / ma20_amt if ma20_amt > 0 else np.nan
    # 缩量整理度：近10日量 / 近30日量，越小说明整理越"干"
    vol_dry = amt10 / amt30 if amt30 > 0 else np.nan
    # 资金倾向：上涨日与下跌日成交额之比
    chg = np.diff(c[idx - bd: idx + 1], prepend=c[idx - bd])
    up = amt[idx - bd + 1: idx + 1][chg[idx - bd + 1: idx + 1] > 0].sum()
    dn = amt[idx - bd + 1: idx + 1][chg[idx - bd + 1: idx + 1] < 0].sum()
    money_tilt = up / dn if dn > 0 else np.nan

    # --- 趋势质量 ---
    ret10 = last / c[idx - 10] - 1.0
    ret20 = last / c[idx - 20] - 1.0
    ret60 = last / c[idx - 60] - 1.0
    ret120 = last / c[idx - 120] - 1.0
    ret250 = last / c[idx - 250] - 1.0
    ma20 = float(ma(c, 20)[idx]); ma60 = float(ma(c, 60)[idx])
    ma120 = float(ma(c, 120)[idx])
    trend_stack = int(last > ma20 > ma60) + int(ma60 > ma120)

    # --- 突破状态（相对 b 日前的最近 high_days 新高） ---
    # 过去 hd 日内，收盘价创出新高的天数（强势度）
    roll_max_prev = rolling_extreme(c, hd, "max")
    new_high_days = int(sum(1 for i in range(idx - bd + 1, idx + 1)
                            if c[i] >= roll_max_prev[i - 1] * 0.999 if i >= 1))

    # 距离 30 日 & 250 日箱顶的空间（突破触发位）
    trig_box = hi30
    trig_year = hi250
    dist_to_box = last / trig_box - 1.0
    dist_to_year = last / trig_year - 1.0

    # --- 该股自身历史波动基准：过去一年滚动 bd 日宽度的分布 ---
    widths_hist = []
    for j in range(idx - hd + bd, idx + 1):
        if j - bd + 1 < 0:
            continue
        hh = c[j - bd + 1: j + 1].max(); ll = c[j - bd + 1: j + 1].min()
        if ll > 0:
            widths_hist.append(hh / ll - 1.0)
    width_pct = float(np.mean(np.array(widths_hist) >= width30)) if widths_hist else np.nan

    return {
        "date": bars[idx]["date"],
        "close": float(last),
        "hi250": hi250, "hi30": hi30, "lo30": lo30,
        "hi10": hi10, "lo10": lo10, "hi20": hi20, "lo20": lo20,
        "hi250_ago": hi250_ago,
        "base_days": int(base_days),
        "width30": float(width30), "width20": float(width20), "width10": float(width10),
        "prox": float(prox), "dd_from_high": float(dd_from_high),
        "pos_in_box": float(pos_in_box), "contraction": float(contraction),
        "width_pct": width_pct, "atr_pct": atr_pct,
        "amt30": amt30, "amt10": amt10, "turn30": turn30, "vr20": float(vr20),
        "vol_dry": float(vol_dry), "money_tilt": float(money_tilt),
        "ret10": float(ret10), "ret20": float(ret20), "ret60": float(ret60),
        "ret120": float(ret120), "ret250": float(ret250),
        "trend_stack": trend_stack,
        "new_high_days": new_high_days,
        "trig_box": float(trig_box), "trig_year": float(trig_year),
        "dist_to_box": float(dist_to_box), "dist_to_year": float(dist_to_year),
    }


def forward_return(bars: Sequence[dict], idx: int, horizon: int) -> Optional[float]:
    """买入持有 horizon 个交易日的收益（以 idx 收盘为基准）"""
    j = idx + horizon
    if j >= len(bars):
        return None
    return float(bars[j]["close"] / bars[idx]["close"] - 1.0)


def mae_mfe(bars: Sequence[dict], idx: int, horizon: int) -> Dict[str, Optional[float]]:
    """
    持有期内最大不利/有利幅度。用于评估止损设置的合理性。
    mae < 0 表示期间最差回撤；mfe > 0 表示期间最好浮盈。
    """
    j = min(idx + horizon, len(bars) - 1)
    base = bars[idx]["close"]
    lows = [b["low"] / base - 1.0 for b in bars[idx + 1: j + 1]]
    highs = [b["high"] / base - 1.0 for b in bars[idx + 1: j + 1]]
    return {
        "mae": float(min(lows)) if lows else None,
        "mfe": float(max(highs)) if highs else None,
    }
