# -*- coding: utf-8 -*-
"""
年度新高选股策略 —— 筛选引擎（硬性过滤 + 打分排序）

证据来源：yh_eventstudy.py 的 35.9 万个历史事件回放（2023-03 ~ 2026-07），
阈值依据见 yh_calibrate.py 输出与 README_策略说明.md。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================== 参数 ==============================
PARAMS = {
    # --- 结构：接近一年新高 ---
    "high_days": 250,
    "prox_min": 0.90,        # 30日最高收盘 / 一年最高收盘 >= 0.90
    # --- 结构：箱体 ---
    "box_days": 30,
    "width30_max": 0.25,     # 30日收盘价箱宽 <= 25%
    # --- 资金活跃 ---
    "amt30_min": 4e8,        # 30日日均成交额 >= 4亿
    "turn30_min": 1.5,       # 30日日均换手率 >= 1.5%
    "turn30_max": 10.0,      # 30日日均换手率 <= 10%
    # --- 基本面 ---
    "rev_yoy_min": 30.0,     # 最新报告期营收同比 >= 30%
    "rev_yoy_prev_min": 0.0, # 上一年同期营收同比 > 0
    "revenue_min": 2e8,      # 单期营收 >= 2亿元（排除壳股）
    # --- 行业剔除（电信运营商≠通信设备） ---
    "exclude_industries": [
        "银行Ⅱ", "基础建设", "房屋建设Ⅱ", "专业工程", "工程咨询服务Ⅱ",
        "铁路公路", "物流", "航运港口", "航空机场", "通信服务",
    ],
    # --- 剔除规则（移出票池的两大触发） ---
    "overext_ratio": 1.20,   # 收盘价 > 首次突破价 × 1.20 → 移出（不追高）
    "stop_ratio": 0.80,      # 收盘价 < 30日最高收盘 × 0.80 → 移出（破位）
    # --- 亏损票处理：不硬性剔除，降级并标注（用户允许“高预期故事票”） ---
    "allow_loss_stories": True,
}

# 东财行业名（带Ⅱ后缀）的规范化显示名
INDUSTRY_LABEL = {"银行Ⅱ": "银行", "房屋建设Ⅱ": "房屋建设"}


def _ind(board: str) -> str:
    if not isinstance(board, str):
        return ""
    return INDUSTRY_LABEL.get(board, board)


# ============================== 打分 ==============================
def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    五维打分（0~100）：结构 / 增长质量 / 资金活跃 / 趋势健康 / 财务安全
    权重按事件研究中各维度对超额收益的边际贡献设定。
    """
    out = df.copy()

    def clip01(s):
        return np.clip(s, 0, 1)

    # 1) 结构分（27）：离一年新高越近、箱体越窄越好（数据支持：prox 与 width 均为有效单调项）
    s_struct = (0.6 * clip01((out["prox"] - 0.90) / 0.10) +
                0.4 * clip01((0.30 - out["width30"]) / 0.25))

    # 2) 增长分（28）：营收同比 30%~150% 映射 0~1（YSTZ 单位本身是百分数）
    rev = out["rev_yoy"].clip(0, 150)
    s_growth = 0.5 * clip01((rev - 30) / 120)
    # 增长持续性 + 加速
    accel = (out["rev_yoy"] - out["rev_yoy_prev"]).clip(-50, 100)
    s_growth = s_growth + 0.25 * clip01(accel / 100) + 0.25 * clip01(out["profit_yoy"].clip(0, 100) / 100)

    # 3) 资金分（18）：换手 3~6% 最优（区间峰形），成交额对数刻度
    t = out["turn30"]
    s_turn = np.exp(-((np.log(t / 4.0)) ** 2) / 0.9)          # 峰值在 4%，宽幅衰减
    s_amt = clip01((np.log10(out["amt30"]) - np.log10(4e8)) / (np.log10(3e9) - np.log10(4e8)))
    s_money = 0.6 * s_turn + 0.4 * s_amt

    # 4) 趋势分（17）：多头排列 + 近端相对抗跌 + 30日内创新高天数
    s_trend = (0.4 * out["trend_stack"] / 2.0 +
               0.3 * clip01(1.0 + out["ret60"] / 0.2) +
               0.3 * clip01(out["new_high_days"] / 10.0))

    # 5) 财务安全分（10，扣分制）：亏损 / 低 ROE 折价（归一到 0~1）
    s_safe = np.where(out["is_loss"] == 1, 0.2, 1.0)
    s_safe = np.minimum(s_safe, 0.4 + 0.6 * clip01((out["roe_w"].fillna(0)) / 20.0))
    s_safe = pd.Series(s_safe, index=out.index)

    out["score"] = (27 * s_struct + 28 * s_growth + 18 * s_money +
                    17 * s_trend + 10 * s_safe).round(1)
    return out


# ============================== 过滤 ==============================
def _base_mask(df: pd.DataFrame, p: Dict) -> pd.Series:
    m = (
        (df["prox"] >= p["prox_min"]) &
        (df["width30"] <= p["width30_max"]) &
        (df["amt30"] >= p["amt30_min"]) &
        (df["turn30"] >= p["turn30_min"]) &
        (df["turn30"] <= p["turn30_max"]) &
        (df["rev_yoy"] >= p["rev_yoy_min"]) &
        (df["rev_yoy_prev"] > p["rev_yoy_prev_min"]) &
        (df["revenue"] >= p["revenue_min"]) &
        df["board"].notna() &
        (~df["board"].isin(p["exclude_industries"]))
    )
    return m


def screen(panel: pd.DataFrame, params: Optional[Dict] = None,
           holdings: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    主入口。
    返回 dict：
      pool     —— 核心池：通过全部硬性条件（营收同比>=30%），已打分排序
      watch    —— 观察池：结构/资金/行业全过，营收同比 15%~30%
      story    —— 营收高增长但当前亏损/TTM亏损的「故事票」分组（不剔除，单列）
      removed  —— 当前在池持仓中被剔除规则触发的票（含原因）
      all_scored —— 全市场打分明细（含未通过者，便于复核）
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    holdings = holdings or []

    df = panel.copy()
    df["industry"] = df["board"].map(_ind)

    # ST / 名称异常剔除
    st_mask = df["name"].astype(str).str.contains("ST|退", na=False)
    df = df[~st_mask]

    # 亏损口径：最新一期亏损 或 TTM 市盈率为负
    df["is_loss_ttm"] = ((df["is_loss"] == 1) | (df["pe_ttm"] < 0)).astype(int)

    struct = (
        (df["prox"] >= p["prox_min"]) & (df["width30"] <= p["width30_max"]) &
        (df["amt30"] >= p["amt30_min"]) &
        (df["turn30"] >= p["turn30_min"]) & (df["turn30"] <= p["turn30_max"]) &
        (df["revenue"] >= p["revenue_min"]) &
        df["board"].notna() & (~df["board"].isin(p["exclude_industries"]))
    )
    df["pass"] = struct & (df["rev_yoy"] >= p["rev_yoy_min"]) & (df["rev_yoy_prev"] > 0)
    watch_mask = struct & (df["rev_yoy"] >= 15) & (df["rev_yoy"] < p["rev_yoy_min"]) & (df["rev_yoy_prev"] > 0)
    story_mask = struct & (df["rev_yoy"] >= p["rev_yoy_min"]) & (df["rev_yoy_prev"] > 0) & (df["is_loss_ttm"] == 1)

    # --- 剔除规则：对「仍在池中」的持仓做检查 ---
    rem_rows = []
    for c in holdings:
        if c not in df.index:
            continue
        r = df.loc[c]
        reasons = []
        if r["dd_from_high"] < (p["stop_ratio"] - 1.0):
            reasons.append(f"距一年新高回撤 {r['dd_from_high']:.1%} 已超 {1-p['stop_ratio']:.0%} 破位线")
        if r["width30"] > p["width30_max"]:
            reasons.append(f"30日箱宽 {r['width30']:.1%} 超过 {p['width30_max']:.0%}")
        if r["prox"] < p["prox_min"]:
            reasons.append(f"近30日高/一年高 {r['prox']:.2f} 低于 {p['prox_min']:.2f}，已脱离高位区")
        if isinstance(r["board"], str) and r["board"] in p["exclude_industries"]:
            reasons.append(f"行业 {r['industry']} 在剔除清单")
        if r["rev_yoy"] < p["rev_yoy_min"]:
            reasons.append(f"营收同比 {r['rev_yoy']:.1f}% 低于 {p['rev_yoy_min']:.0f}%")
        if reasons:
            rem_rows.append({"code": c, "name": r["name"], "industry": r["industry"],
                             "close": r["close"], "reasons": "；".join(reasons)})
    removed = pd.DataFrame(rem_rows)

    # --- 打分与排序 ---
    scored = score(df)
    pool = scored[scored["pass"] & (scored["is_loss_ttm"] == 0)].sort_values("score", ascending=False)
    watch = scored[watch_mask & (scored["is_loss_ttm"] == 0)].sort_values("score", ascending=False)
    story = scored[story_mask].sort_values("score", ascending=False)

    return {"pool": pool, "watch": watch, "story": story, "removed": removed, "all_scored": scored}


# ============================== 展示列 ==============================
POOL_COLS = ["name", "industry", "close", "score", "prox", "width30", "dd_from_high",
             "amt30", "turn30", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w",
             "pe_ttm", "pb", "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]

# 比例型字段（0~1）→ 显示为百分数
_FRACTION_COLS = {"prox", "width30", "dd_from_high", "pos_in_box", "ret60", "today_pct", "dist_to_trig"}
# 本身就以百分数为单位的字段 → 原样显示
_PCT_UNIT_COLS = {"turn30", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w"}


def fmt(df: pd.DataFrame, cols: List[str] = None, n: int = 40) -> pd.DataFrame:
    cols = cols or POOL_COLS
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    if "amt30" in d:
        d["amt30"] = (d["amt30"] / 1e8).round(1).astype(str) + "亿"
    if "float_mv" in d:
        d["float_mv"] = (d["float_mv"] / 1e8).round(0).astype(str) + "亿"
    for c in _FRACTION_COLS & set(d.columns):
        d[c] = (d[c].astype(float) * 100).round(1)
        d.rename(columns={c: c + "(%)"}, inplace=True)
    for c in _PCT_UNIT_COLS & set(d.columns):
        d[c] = d[c].astype(float).round(1)
    for c in ["pe_ttm", "pb"]:
        if c in d:
            d[c] = d[c].round(1)
    return d.head(n)
