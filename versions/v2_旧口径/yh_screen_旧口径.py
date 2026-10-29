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


# ============================== v3：分周期参数（前置数据修复 + 口径修正后） ==============================
# v2 → v3 的三处修正（详见 README 第四节 4.7 与 output/final_calib*.txt / def_check.txt）：
#   修正1 prox 口径：原用【近30日最高收盘/250日最高收盘】，会让「从年高回落 15% 后横盘」的票通过
#         （中信证券、财通证券就是这么混进来的）。改用 prox_now = 【当前收盘/250日最高收盘】，
#         更贴合「距年高」原意，且三个周期的效果都更好（DOWN 0.95 档 +3.60% vs +2.31%）。
#   修正2 成交额门槛：v2 曾从 4 亿降到 2 亿，修复数据时变性后证据不稳定，【撤回】，维持 4 亿。
#   修正3 样本：补上幸存者偏差量化（2019 年缺口约 7%）与「纯老股子样本」复核。
#
# 结论的重要变化：修正口径后，UP 与 DOWN 收敛到【同一套参数】。
#   → 「按周期切换选股门槛」的价值大幅下降；真正有效的是三项口径修正本身（跨周期通用）。
#   → 周期标签的用途改为【决定参与强度】：CHOP 无稳定 alpha，DOWN 命中数极少，都应降仓。
PARAMS_BY_REGIME = {
    # 上涨：距年高 5% 以内 + 窄箱 + 控换手，beta 调整超额 +4.22%，前后半样本与老股样本均通过
    "UP": {
        "prox_now_min": 0.95,
        "width30_max": 0.15,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
    },
    # 震荡：无「三段全胜」解，说明该环境下本策略没有稳定 alpha。
    # 退而采用相对 v1 稳定为正的宽松版，并明确按【降仓】处理。
    "CHOP": {
        "prox_now_min": 0.90,
        "width30_max": 0.20,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
    },
    # 下跌：与 UP 同参数。注意——不是放宽，反而是【收紧到 0.95】（v1 原为 0.90）。
    # 放宽到 0.90 / 0.85 会使 beta 调整超额从 +3.60% 降到 +1.93% / +1.18%。
    "DOWN": {
        "prox_now_min": 0.95,
        "width30_max": 0.15,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
    },
}

# 各参数集历史事件回测的 beta 调整超额（相对同期 v1 的增量；v1 = 近30日高/年高>=0.90、箱宽<=25%、额>=4亿、换手1.5~10%）
REGIME_EVIDENCE = {
    "UP": "beta调整超额 +4.22%，相对 v1 +3.85pp（全/前/后三段 +3.85/+4.43/+3.69 全胜，老股样本 +2.37%）",
    "CHOP": "【无稳定 alpha】宽松版 +0.29%，相对 v1 +3.44/+1.62/+4.40 全胜但绝对值近零 → 建议降参与",
    "DOWN": "beta调整超额 +3.60%（90%CI 1.5~5.6，胜率 67.2%），相对 v1 +6.75pp（三段 +6.75/+4.73/+7.53 全胜，老股样本 +2.27%）",
}

# 周期 → 建议参与强度（不是选股门槛，是仓位纪律）
REGIME_SIZE = {
    "UP": "正常仓（历史日均命中 4.5 只）",
    "CHOP": "半仓（该环境无稳定 alpha，命中虽多但超额近零）",
    "DOWN": "轻仓（严格口径下命中极少，命中数本身就是仓位信号）",
}


def resolve_regime(asof: Optional[str] = None,
                   cache_path: Optional[str] = None) -> Dict:
    """读取中级周期状态序列，返回最近一日的 regime 判定。"""
    import pickle
    cache_path = cache_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache", "regime_state.pkl")
    with open(cache_path, "rb") as f:
        st = pickle.load(f)
    st = st.sort_values("date")
    if asof:
        st = st[st["date"] <= asof]
    last = st.iloc[-1]
    return {
        "regime": str(last["regime"]),
        "date": str(last["date"]),
        "breadth": round(float(last["breadth"]), 4),
        "breadth_slope": round(float(last["breadth_slope"]), 4),
        "hs300_dd250": round(float(last["沪深300_dd250"]), 4),
    }


def params_for(regime: str) -> Dict:
    """按周期取回完整参数集（未指定的一维沿用 v1 默认值）。"""
    p = dict(PARAMS)
    p.update(PARAMS_BY_REGIME.get(str(regime).upper(), {}))
    p["regime"] = str(regime).upper()
    return p


def _ind(board: str) -> str:
    if not isinstance(board, str):
        return ""
    return INDUSTRY_LABEL.get(board, board)


# ============================== 打分 ==============================
def score(df: pd.DataFrame, p: Optional[Dict] = None) -> pd.DataFrame:
    """
    五维打分（0~100）：结构 / 增长质量 / 资金活跃 / 趋势健康 / 财务安全
    权重按事件研究中各维度对超额收益的边际贡献设定。

    v2 变更（证据见 output/regime_calib*.txt）：
      - 结构分改用「相对当前周期门槛」归一：箱宽按 <=15% 刻度打分（原 30%），
        反映 2019-2026 三周期一致的「窄箱体才是真正有效项」；
      - 资金分的换手峰值从 4.0% 下移到 2.5%，并对 >6% 做额外衰减：
        换手 >6% 在三个周期里 beta 调整超额均为负（-1.4% ~ -7.9%）；
      - 成交额刻度起点随当前周期门槛走（v3 三个周期均回到 4 亿）。
      v3 变更：结构分改用 prox_now（当前价/年高），理由见 PARAMS_BY_REGIME 注释。
    """
    out = df.copy()
    pr = dict(PARAMS)
    if p:
        pr.update(p)
    w_ref = max(pr.get("width30_max", 0.25), 0.15)
    a_ref = pr.get("amt30_min", 4e8)

    def clip01(s):
        return np.clip(s, 0, 1)

    # v3：结构分改用 prox_now（当前价/年高）。旧口径 prox 只作对照，不再入分。
    if "prox_now" not in out.columns:
        out["prox_now"] = 1.0 + out["dd_from_high"]

    # 1) 结构分（27）：离一年新高越近、箱体越窄越好（数据支持：prox_now 与 width 均为有效单调项）
    s_struct = (0.5 * clip01((out["prox_now"] - 0.90) / 0.10) +
                0.5 * clip01((w_ref * 1.0 - out["width30"]) / (w_ref * 0.8)))

    # 2) 增长分（28）：营收同比 30%~150% 映射 0~1（YSTZ 单位本身是百分数）
    rev = out["rev_yoy"].clip(0, 150)
    s_growth = 0.5 * clip01((rev - 30) / 120)
    # 增长持续性 + 加速
    accel = (out["rev_yoy"] - out["rev_yoy_prev"]).clip(-50, 100)
    s_growth = s_growth + 0.25 * clip01(accel / 100) + 0.25 * clip01(out["profit_yoy"].clip(0, 100) / 100)

    # 3) 资金分（18）：换手 2.0~3.0% 最优，>6% 明显衰减（三周期一致，见 regime_calib2 D 段）
    t = out["turn30"]
    s_turn = np.exp(-((np.log(t / 2.5)) ** 2) / 1.2)          # 峰值在 2.5%
    s_turn = np.where(t > 6.0, s_turn * np.exp(-(t - 6.0) / 4.0), s_turn)
    s_amt = clip01((np.log10(out["amt30"]) - np.log10(a_ref)) / (np.log10(3e9) - np.log10(a_ref)))
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
def prox_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """距年高门槛。v3 起统一用「当前价 / 250日最高收盘」口径（prox_now）。

    旧口径 prox = 近30日最高收盘 / 250日最高收盘，会放行「从年高回落
    15% 后横盘」的票（中信证券、财通证券即因此入选），已废弃。
    仅当显式传入 prox_min 且未给 prox_now_min 时才回落到旧口径。
    """
    if p.get("prox_now_min") is not None:
        col = "prox_now"
        if col not in df.columns:
            df[col] = 1.0 + df["dd_from_high"]
        return df[col] >= p["prox_now_min"]
    return df["prox"] >= p.get("prox_min", 0.90)


def _base_mask(df: pd.DataFrame, p: Dict) -> pd.Series:
    m = (
        prox_gate(df, p) &
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

    # v3：距年高的「当前价口径」= 当前收盘 / 250日最高收盘。
    # 与旧的 prox（近30日最高/年高）并存，后者保留仅作对照展示。
    if "prox_now" not in df.columns:
        df["prox_now"] = 1.0 + df["dd_from_high"]

    # 亏损口径：最新一期亏损 或 TTM 市盈率为负
    df["is_loss_ttm"] = ((df["is_loss"] == 1) | (df["pe_ttm"] < 0)).astype(int)

    # 距年高门槛：优先用 v3 的「当前价口径」，未指定则回落到旧口径
    pg = prox_gate(df, p)

    struct = (
        pg & (df["width30"] <= p["width30_max"]) &
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
        if p.get("prox_now_min") is not None:
            if r["prox_now"] < p["prox_now_min"]:
                reasons.append(f"当前价/一年高 {r['prox_now']:.2f} 低于 {p['prox_now_min']:.2f}，已脱离高位区")
        elif r["prox"] < p.get("prox_min", 0.90):
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
    scored = score(df, p)
    pool = scored[scored["pass"] & (scored["is_loss_ttm"] == 0)].sort_values("score", ascending=False)
    watch = scored[watch_mask & (scored["is_loss_ttm"] == 0)].sort_values("score", ascending=False)
    story = scored[story_mask].sort_values("score", ascending=False)

    return {"pool": pool, "watch": watch, "story": story, "removed": removed, "all_scored": scored}


# ============================== 展示列 ==============================
POOL_COLS = ["name", "industry", "close", "score", "prox_now", "prox", "width30", "dd_from_high",
             "amt30", "turn30", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w",
             "pe_ttm", "pb", "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]

# 比例型字段（0~1）→ 显示为百分数
_FRACTION_COLS = {"prox", "prox_now", "width30", "dd_from_high", "pos_in_box", "ret60", "today_pct", "dist_to_trig"}
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
