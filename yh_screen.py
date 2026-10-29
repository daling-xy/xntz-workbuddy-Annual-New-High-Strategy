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
    # 【命名约定】本策略统一使用「箱宽」或「箱体振幅」指代下面这个量，
    #   **绝不要把它叫成「波动率」** —— 统计学的「波动率」专指收益率标准差，是另一个量。
    #   公式：width30 = (近30日最高收盘 − 近30日最低收盘) / 近30日最低收盘
    #   口径：只用【收盘价】，不使用 K 线的盘中 high/low
    #   量纲：区间相对振幅（range），非标准差
    #   两者关系：相关系数仅 0.765，约四分之一信息不重叠
    "width30_max": 0.25,     # 30日箱宽(振幅) <= 25%
    # --- 资金活跃 ---
    "amt30_min": 4e8,        # 30日日均成交额 >= 4亿
    "turn30_min": 1.5,       # 30日日均换手率 >= 1.5%
    "turn30_max": 10.0,      # 30日日均换手率 <= 10%
    # --- 前期涨幅上限（不追已大幅拉升的票）---
    # 2026-10-28 新增。经典「年度新高」三派一致要求：不买已经 extended 的股票。
    #   国盛金工《创新高股票中的Alpha》：前期涨跌幅低 = 反应不足 = 后续超额高
    #     （创新高后20日差 1.02%，240日差 2.09%）。
    #   Minervini / CANSLIM：买入点须在 pivot 上方 5% 以内，追高即失 edge。
    # 本策略实测（结构门内 2867 条，区块 bootstrap）：
    #   近60日涨幅 最低25% 组 +3.33%(显著) / 大亏率 2.2%
    #   近60日涨幅 最高25% 组 +2.09%(含零) / 大亏率 9.9%   ← 4.4 倍
    #   ★ 逐年 8/8 年（2019~2026）高涨幅组大亏率均更高 → 这是【风险】因子，
    #     不是收益因子；收益差异方向一致但不足以单独证明。
    # 阈值 = 历史结构门内 ret60 的 75 分位 18.7%，取 19%（固定值便于复现）。
    # 效果：日均票 4.03→3.48，收益 +2.58%→+2.56%，胜率 51.2%→53.3%，
    #       大亏率 6.0%→4.7%。
    "ret60_max": 0.19,       # 近60日涨幅 <= 19%（None = 不设）
    # --- 基本面 ---
    # 营收增速门槛：2026-10-27 起【全周期关闭】。
    #   区块 bootstrap 复核（见 versions/推导归档_参数实证_20261027.md）显示：
    #   全样本/CHOP/DOWN 设 >=0% 均【显著为负】，UP 期点估计虽大但 CI 含零且
    #   样本仅 0.63 个独立持有期（56 事件/38 交易日，55% 集中在 2025 年）。
    #   保留 None 即不筛选增速；未来若重启只需在此填数值。
    "rev_yoy_min": None,     # None = 不设营收增速门槛
    "rev_yoy_prev_min": 0.0, # 上期营收同比 > 0（仅在 rev_yoy_min 非 None 时生效）
    # 2026-10-27 变更：改用 TTM（最近四个季度）口径，阈值同步 2亿 → 4亿。
    #   原值 2 亿对应【半年累计】，等价于全年 4 亿；改 TTM 后按全年标定。
    #   目的：消除「累计数跨报告期不可比」的口径污染（详见 README 3.4）。
    "revenue_min": 4e8,      # 最近一年(TTM)营收 >= 4亿元（排除壳股）
    # --- 行业剔除（电信运营商≠通信设备） ---
    # 2026-10-26 变更：应户主要求，将「航运港口」移出剔除清单（中远海控 601919 所属）。
    #   原剔除理由：航运/港口属强周期重资产，营收高增多来自运价周期而非成长逻辑，
    #   与本策略「营收同比>=30% 的成长筛选」口径不匹配。
    #   移出后该行业 35 只个股重新进入候选池，其余 9 个行业剔除规则不变。
    "exclude_industries": [
        "银行Ⅱ", "基础建设", "房屋建设Ⅱ", "专业工程", "工程咨询服务Ⅱ",
        "铁路公路", "物流", "航空机场", "通信服务",
    ],
    # --- 剔除规则（移出票池的两大触发） ---
    "overext_ratio": 1.20,   # 收盘价 > 首次突破价 × 1.20 → 移出（不追高）
    "stop_ratio": 0.80,      # 收盘价 < 30日最高收盘 × 0.80 → 移出（破位）
    # --- 亏损票处理：不硬性剔除，降级并标注（用户允许“高预期故事票”） ---
    "allow_loss_stories": True,
}

# 东财行业名（带Ⅱ后缀）的规范化显示名
INDUSTRY_LABEL = {"银行Ⅱ": "银行", "房屋建设Ⅱ": "房屋建设"}


# ============================== v2 修正版：分周期参数（前置数据修复 + 口径修正后） ==============================
# v2 本次修订的三处修正（详见 README 第四节 4.7 与 output/final_calib*.txt / def_check.txt）：
#   修正1 prox 口径：原用【近30日最高收盘/250日最高收盘】，会让「从年高回落 15% 后横盘」的票通过
#         （中信证券、财通证券就是这么混进来的）。改用 prox_now = 【当前收盘/250日最高收盘】，
#         更贴合「距年高」原意，且三个周期的效果都更好（DOWN 0.95 档 +3.60% vs +2.31%）。
#   修正2 成交额门槛：v2 曾从 4 亿降到 2 亿，修复数据时变性后证据不稳定，【撤回】，维持 4 亿。
#   修正3 样本：补上幸存者偏差量化（2019 年缺口约 7%）与「纯老股子样本」复核。
#
# 结论（2026-10-27 修订）：【结构性门槛】UP / CHOP / DOWN 三周期统一
#   （距年高 0.95、箱宽 15%、额 4亿、换手 ≤6%），这几项的分周期差异
#   在区块 bootstrap 下基本都不显著（仅 CHOP 箱宽 10%、DOWN 箱宽 14% 显著更优，
#   增益 <1pp，暂不启用）。
#   → 但【营收增速门槛】存在强周期交互，必须分化：
#     CHOP 设门槛显著为负（取消）、DOWN 无证据支持且票量代价大（取消）、
#     UP 点估计单调递增且幅度大（保留 30%，但未达显著）。
#     详见 README 3.4 与 versions/推导归档_参数实证_20261027.md。
#   → 周期标签同时决定【参与强度】：CHOP 无稳定 alpha，DOWN 命中数极少，都应降仓。
PARAMS_BY_REGIME = {
    # 上涨：距年高 5% 以内 + 窄箱 + 控换手，beta 调整超额 +4.22%，前后半样本与老股样本均通过
    # 上涨：保留营收增速门槛。区块 bootstrap 下点估计单调递增且幅度大
    #   （>=0 -> +0.80pp, >=10 -> +3.93pp, >=20 -> +6.63pp, >=30 -> +10.13pp,
    #    >=40 -> +14.63pp, >=50 -> +17.04pp），但受限于 UP 仅 12 个有效区块，
    #   CI 均含零、未达显著。保留的理由是「方向一致且幅度大」，不是「已证实」。
    "UP": {
        "prox_now_min": 0.95,
        "width30_max": 0.15,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
        "rev_yoy_min": None,      # 2026-10-27 起关闭：证据不足（CI 含零，样本仅 0.63 个持有期）
        "rev_yoy_prev_min": 0.0,
    },
    # 震荡：搜索后【无三段全胜解】，说明该环境下本策略没有稳定 alpha。
    # 因此门槛沿用与 UP/DOWN 相同的统一值（门槛已不是胜负手），
    # 真正的处理是【半仓参与】——见 REGIME_SIZE。
    # 震荡（占全部交易日 67%）：明确【不设】营收增速门槛。
    #   这是本轮唯一站得住的显著结论，且样本最充分（29 个区块）：
    #     >=0%   → −0.59pp CI[−0.94,−0.27] 显著为负
    #     >=10%  → −1.15pp CI[−2.26,−0.16] 显著为负
    #   即：在已通过结构门（贴年高+窄箱+有量+低换手）的票里，
    #   再按营收增速筛选会显著损害超额。
    "CHOP": {
        "prox_now_min": 0.95,
        "width30_max": 0.15,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
        "rev_yoy_min": None,      # 不设增速门槛
        "rev_yoy_prev_min": 0.0,
    },
    # 下跌：与 UP 同参数。注意——不是放宽，反而是【收紧到 0.95】（v1 原为 0.90）。
    # 放宽到 0.90 / 0.85 会使 beta 调整超额从 +3.60% 降到 +1.93% / +1.18%。
    # 下跌：同样【不设】营收增速门槛。
    #   区块 bootstrap 下各档全部含零（>=30% 为 −1.75pp CI[−4.76,+2.43]），
    #   点估计一致为负或约等于零，而代价是砍掉约 2/3 的候选票（日均 3.58 → 1.18）。
    #   「无证据支持 + 点估计不为正 + 票量代价大」→ 取消。
    #   机制：营收增速与 beta 正相关（Spearman 0.347），下跌市里这道门
    #         实际是在挑高波动品种，与低波防守的门互相打架。
    "DOWN": {
        "prox_now_min": 0.95,
        "width30_max": 0.15,
        "amt30_min": 4e8,
        "turn30_min": 0.0,
        "turn30_max": 6.0,
        "rev_yoy_min": None,      # 不设增速门槛
        "rev_yoy_prev_min": 0.0,
    },
}

# 各参数集历史事件回测的 beta 调整超额（相对同期 v1 的增量；v1 = 近30日高/年高>=0.90、箱宽<=25%、额>=4亿、换手1.5~10%）
# ⚠️ 2026-10-27 全量改为【区块 bootstrap】口径（区块=60交易日，共29块）。
#     此前按天 bootstrap 因 fwd60 窗口重叠 98% 而低估标准误，CI 系统性偏窄。
#     旧口径下 DOWN 曾显示为「+3.60% CI[1.5,5.6] 显著」，修正后含零 → 已作废。
REGIME_EVIDENCE = {
    "UP": "β调整超额 +4.22%，90%CI[+1.29,+7.62] 显著为正（区块bootstrap）→ 唯一有显著证据的周期",
    "CHOP": "【无稳定 alpha】+1.92%，90%CI[-0.15,+4.02] 含零（区块bootstrap）→ 半仓",
    "DOWN": "+3.60%，90%CI[-0.22,+8.21] 含零（区块bootstrap；旧口径误判为显著，已作废）→ 降仓",
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


# ---------------------------------------------------------------------------
# v1 旧口径参数：【仅供每日输出的「v1 对照」显示使用】，与 v2 选股参数彻底分离。
#
# 2026-10-27 修复：原先 v1 对照直接复用顶层 PARAMS，而 PARAMS 同时是 v2 的默认值，
#   导致 2026-10-27 关闭营收增速门槛时把 v1 的 30% 门槛也一并关掉 ——
#   「v1 对照」名不副实，且 v1/v2 参数互相污染。
#   现冻结为导入期快照：v2 只走 PARAMS / PARAMS_BY_REGIME，v1 只走这里，互不干扰。
#   注意：这里不是「把策略改回 v1」，v2 的实盘选股参数不受任何影响。
V1_PARAMS = dict(PARAMS)
V1_PARAMS.update({
    "prox_now_min": None,   # v1 用旧口径 prox = 近30日最高收盘 / 250日最高收盘
    "prox_min": 0.90,
    "width30_max": 0.25,
    "amt30_min": 4e8,
    "turn30_min": 1.5,
    "turn30_max": 10.0,
    "ret60_max": None,      # v1 无前期涨幅约束（该门为 v2 于 2026-10-28 新增）
    "rev_yoy_min": 30.0,    # v1 原文条件：营收同比 >= 30%
    "rev_yoy_prev_min": 0.0,
    "regime": "V1",
})


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
      - 成交额刻度起点随当前周期门槛走（修正后三个周期均回到 4 亿）。
      v2 修正：结构分改用 prox_now（当前价/年高），理由见 PARAMS_BY_REGIME 注释。
    """
    out = df.copy()
    pr = dict(PARAMS)
    if p:
        pr.update(p)
    w_ref = max(pr.get("width30_max", 0.25), 0.15)
    a_ref = pr.get("amt30_min", 4e8)

    def clip01(s):
        return np.clip(s, 0, 1)

    # v2 修正：结构分改用 prox_now（当前价/年高）。旧口径 prox 只作对照，不再入分。
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
    """距年高门槛。v2 修正版起统一用「当前价 / 250日最高收盘」口径（prox_now）。

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


def _rev_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """营收增速门槛。p['rev_yoy_min'] 为 None 时不设这道门（不筛选增速）。
    另外 rev_yoy 为 NaN（无财报数据）时一律不通过——这是数据完整性要求，
    与「要不要设增速门槛」是两件事。"""
    thr = p.get("rev_yoy_min", None)
    if thr is None:
        return df["rev_yoy"].notna()
    return df["rev_yoy"].notna() & (df["rev_yoy"] >= thr)


def _rev_prev_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """上期营收同比 > 0。仅在设置了增速门槛时才作为配套条件（避免孤立生效）。"""
    if p.get("rev_yoy_min", None) is None:
        return pd.Series(True, index=df.index)
    return df["rev_yoy_prev"] > p.get("rev_yoy_prev_min", 0.0)


def _rev_size_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """营收规模门槛（排除壳股）。优先用 TTM（最近四个季度）口径；
    若面板没有 TTM 字段则回退到累计口径，并在调用方给出提示。"""
    col = "revenue_ttm" if "revenue_ttm" in df.columns else "revenue"
    return pd.to_numeric(df[col], errors="coerce") >= p["revenue_min"]


def _ret60_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """前期涨幅上限。p['ret60_max'] 为 None 时不设这道门（如 v1 旧口径）。
    ret60 缺失时一律放行——缺失是数据完整性问题，不应静默淘汰样本。"""
    thr = p.get("ret60_max", None)
    if thr is None or "ret60" not in df.columns:
        return pd.Series(True, index=df.index)
    return pd.to_numeric(df["ret60"], errors="coerce") <= thr


def form_tag(df: pd.DataFrame) -> pd.Series:
    """形态三分（仅供展示与人工复核，【不参与筛选】）。

    真突破·创年高 ：prox_now >= 0.999，已站上一年新高
    临界·距年高<0.5%：0.995 <= prox_now < 0.999，一线之隔（避免把只差 0.1% 的
                    票误标为假突破；实证上仍属「未创年高」组，标注仅作提示）
    假突破·贴箱顶  ：贴 30 日箱顶(pos_in_box>=0.8) 但距年高 >0.5% → 诱多形态
    箱内蓄势       ：其余（箱体中部，尚未触及年高）

    实证依据（结构门内 2867 条，区块 bootstrap）：
      贴箱顶且创年高   +4.37% CI[+1.33,+7.88] 显著，大亏率 5.6%
      贴箱顶但未创年高 +1.18% CI[-1.13,+3.75] 含零，大亏率 7.6%
    → 「箱顶」本身不是坏事，坏事是「贴着 30 日箱顶但年高还在上方」的假突破。
      这正好解释了达瓦斯「突破箱顶才买」与本策略实测「箱顶组最差」的表面矛盾。
    """
    prox_now = pd.to_numeric(df.get("prox_now", np.nan), errors="coerce")
    pos = pd.to_numeric(df.get("pos_in_box", np.nan), errors="coerce")
    nh = prox_now >= 0.999
    edge = (prox_now >= 0.995) & (prox_now < 0.999)
    top = pos >= 0.8
    out = np.where(nh, "真突破·创年高",
                   np.where(edge, "临界·距年高<0.5%",
                            np.where(top, "假突破·贴箱顶", "箱内蓄势")))
    return pd.Series(out, index=df.index)


def _base_mask(df: pd.DataFrame, p: Dict) -> pd.Series:
    m = (
        prox_gate(df, p) &
        (df["width30"] <= p["width30_max"]) &
        (df["amt30"] >= p["amt30_min"]) &
        (df["turn30"] >= p["turn30_min"]) &
        (df["turn30"] <= p["turn30_max"]) &
        _rev_gate(df, p) &
        _rev_prev_gate(df, p) &
        _rev_size_gate(df, p) &
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

    # v2 修正：距年高的「当前价口径」= 当前收盘 / 250日最高收盘。
    # 与旧的 prox（近30日最高/年高）并存，后者保留仅作对照展示。
    if "prox_now" not in df.columns:
        df["prox_now"] = 1.0 + df["dd_from_high"]

    # 亏损口径：最新一期亏损 或 TTM 市盈率为负
    df["is_loss_ttm"] = ((df["is_loss"] == 1) | (df["pe_ttm"] < 0)).astype(int)

    # 距年高门槛：优先用 v2 修正版的「当前价口径」，未指定则回落到旧口径
    pg = prox_gate(df, p)

    struct = (
        pg & (df["width30"] <= p["width30_max"]) &
        (df["amt30"] >= p["amt30_min"]) &
        (df["turn30"] >= p["turn30_min"]) & (df["turn30"] <= p["turn30_max"]) &
        _ret60_gate(df, p) &
        _rev_size_gate(df, p) &
        df["board"].notna() & (~df["board"].isin(p["exclude_industries"]))
    )
    # 形态三分：仅标注，不参与筛选（见 form_tag 说明）
    df["form"] = form_tag(df)
    # RS 相对强度（O'Neil RS rating 同义）：当日全市场 ret250 百分位，仅标注。
    # 2026-10-28 诊断：RS>=85 组点估计最高（+4.28%）、胜率 63.3%，但 CI 含零 → 不做门槛。
    if "ret250" in df.columns:
        df["rs_pct"] = pd.to_numeric(df["ret250"], errors="coerce").rank(pct=True) * 100.0
    else:
        df["rs_pct"] = np.nan
    df["pass"] = struct & _rev_gate(df, p) & _rev_prev_gate(df, p)
    # 观察池 / 故事票：按「营收增速门槛」的相对位置定义。
    # 当周期不设增速门槛（CHOP / DOWN）时，改用固定 15% / 30% 这两个参考档，
    # 保证这两类输出仍是"相对更优/更极端"的观察对象，语义不随门槛开关而漂移。
    _thr = p.get("rev_yoy_min", None)
    if _thr is None:
        watch_mask = struct & (df["rev_yoy"] >= 15) & (df["rev_yoy_prev"] > 0)
        story_mask = (struct & (df["rev_yoy"] >= 30) & (df["rev_yoy_prev"] > 0) &
                      (df["is_loss_ttm"] == 1))
    else:
        watch_mask = (struct & (df["rev_yoy"] >= 15) & (df["rev_yoy"] < _thr) &
                      (df["rev_yoy_prev"] > 0))
        story_mask = (struct & (df["rev_yoy"] >= _thr) & (df["rev_yoy_prev"] > 0) &
                      (df["is_loss_ttm"] == 1))

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
        _thr = p.get("rev_yoy_min", None)
        if _thr is not None and pd.notna(r["rev_yoy"]) and r["rev_yoy"] < _thr:
            reasons.append(f"营收同比 {r['rev_yoy']:.1f}% 低于 {_thr:.0f}%")
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
             "amt30", "turn30", "vr20", "rs_pct", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w",
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
    if "vr20" in d:
        d["vr20"] = d["vr20"].astype(float).round(2)      # 当日量比（倍）
    if "rs_pct" in d:
        d["rs_pct"] = d["rs_pct"].astype(float).round(0)  # RS 相对强度（0~100 分）
    return d.head(n)
