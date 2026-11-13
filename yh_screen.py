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
    "prox_min": 0.90,        # 旧口径回落值：30日最高收盘 / 一年最高收盘（prox_now_min 生效时不用）
    # 【2026-11-12 正式定版 · v2.2】距年高门 0.95 → 0.88（当前价/250日高）。
    #   用户原话：德尔平台段「prox 0.89~0.96 其实已经处在合理区间」。
    #   注意：prox≥0.88 意味着策略定位从「创新高前夜」转为「上升趋势中继平台」
    #   （Minervini 原教旨 VCP），β 与大亏率显著抬升，实证见
    #   versions/推导归档_VCP形态捕捉器实证_20261112.md（用户知悉后仍指令采纳）。
    "prox_now_min": 0.88,    # 当前收盘 / 250日最高收盘 >= 0.88
    # --- 结构：箱体 ---
    "box_days": 30,
    # 【命名约定】本策略统一使用「箱宽」或「箱体振幅」指代下面这个量，
    #   **绝不要把它叫成「波动率」** —— 统计学的「波动率」专指收益率标准差，是另一个量。
    #   公式：width30 = (近30日最高收盘 − 近30日最低收盘) / 近30日最低收盘
    #   口径：只用【收盘价】，不使用 K 线的盘中 high/low
    #   量纲：区间相对振幅（range），非标准差
    #   两者关系：相关系数仅 0.765，约四分之一信息不重叠
    # 【2026-11-12 v2.2】width30_max 不再是主门：VCP 多尺度形态门（shape_*）已取代
    #   「w30 ≤ 15%」。本值仅作 _shape_gate 缺列/缺参时的回落默认与 V1 对照兼容。
    "width30_max": 0.25,
    # 【2026-11-12 正式定版 · v2.2】VCP 多尺度形态门（替代原 w30≤15% 绝对箱宽门）：
    #   10日箱宽 ≤ 10%（末端绝对窄）且 收缩比 ct = w10/w30 ≤ 0.45（先宽后窄）且
    #   30日箱宽 ≤ 50%（极端波动兜底，德尔平台段 w30 中位 44.2%）。
    #   依据：德尔股份平台段 w30 被前一波拉升污染（中位 44.2%），真实收敛在 10 日尺度
    #   （w10 中位 9.4%、ct 中位 0.19）；该组合下平台段命中 5/10 天。
    "shape_w10_max": 0.10,   # VCP 形态门：近10日箱宽 <= 10%
    "shape_ct_max": 0.45,    # VCP 形态门：收缩比 width10/width30 <= 0.45
    "shape_w30_cap": 0.50,   # VCP 形态门：30日箱宽 <= 50%（极端波动兜底）
    # 【2026-11-12 正式定版 · v2.3】精准形态四门（在 v2.2 基线之上叠加，逐条）
    #   动机：v2.2 的 VCP 门只回答「末端有没有变窄」，不管趋势方向、高点新不新鲜、
    #   这个窄是逐级收缩还是断层骤降 → 下跌中继（海康威视）、磨底横盘（九州通）、
    #   急反弹（双环传动）、双顶旧高位（南方传媒）都能通过，用户点名质疑。
    #   全样本实证（150.2 万事件，research/yh_shape_precise.py）：
    #     v2.2 基线 日均17.7只 ab +0.75% 大亏11.2%
    #     +N+R+C+H  日均 7.8只 ab +1.07% 大亏11.5% β0.79
    #     ⚠️ 配对 bootstrap 点估计 −0.34pp、90%CI[−0.92,+0.18] 含零；
    #        被剔除组逐年大亏率仅 2/8 年更高 → 形态更「像人话」，但统计优势不显著。
    #        本次为用户指令采纳（「按你的建议跑一下看看」），非统计定论。
    #   注意：**刻意不加 T2（均线多头排列）** —— 德尔股份平台期 ma60(27.90) < ma120(29.63)，
    #        它是深跌反转而非健康上升趋势，均线多头门会永久杀死该目标形态。
    # 【2026-11-12 v2.5 定版】N 门（新高新鲜 hi250_ago≤60）【删除】。
    #   用户原话：「双顶考虑下」。16 组合矩阵实测（output/nrch_grid_20261112.txt）：
    #     无附加(v2.2) 日均17.7 ab+0.75% 大亏11.2%
    #     R+C+H       日均10.4 ab+1.09% 大亏10.5% β0.78   ← 采纳
    #     N+R+C+H     日均 7.9 ab+1.07% 大亏11.5% β0.79   ← 旧版 v2.4
    #   删 N 后票量 +32%、alpha 略升、大亏率反降 1.0pp；被 N 剔掉的「旧高点票」
    #   并不比保留组更差，这条门只是无谓地砍票量。
    #   ⚠️ 副作用：中炬高新、九州通这类低位的收敛平台会回到池里。复核过两者
    #     2026-11-11 收盘分别为 ma120 的 1.048 / 1.039 倍 —— 不是下跌中继，
    #     而是「底部反转后的收敛平台」，形态本身成立；观感偏「弱」源于绝对价位低。
    "shape_hi_age_max": None,  # N 已删；如需恢复填 60
    "shape_ret10_max": 0.08,   # R 不追高：近10日涨幅 <= 8%（防「正在突破」的追高票）
    "shape_c_w20": 0.70,     # C 逐级收缩：width10 <= width20 × 0.70
    "shape_c_w30": 0.90,     # C 逐级收缩：width20 <= width30 × 0.90
    "shape_c_fast": 0.04,    # C 豁免：width10 <= 4% 视为「一直很窄」，不要求逐级收缩
    "shape_h_ratio": 0.95,   # H 箱顶不塌：近10日最高 >= 近30日最高 × 0.95
    # 【2026-11-12 v2.6 定版】F 横盘天数上限 base_days <= 35。
    #   用户诉求：「主要不是捕捉坑后平台，德尔只是恰好那样；主要是横盘 20-30 天 + 其他条件」。
    #   缺口确认：width10/20/30 只度量「最近 k 天有多窄」，不度量「已经窄了多久」。
    #   实证（output/flat_days_20261112.txt，24.7 万事件分箱）：
    #     base_days   fwd60     ab      大亏率
    #       0-9      -1.42%   -4.78%   30.1%   ← 垃圾区
    #      15-19     +1.74%   -0.93%   18.5%
    #      30-39     +2.77%   +0.31%   12.4%   ← fwd60 / ex_fwd60 双峰值
    #      60+       +2.41%   +0.61%    4.9%   ← ab 高仅因 β 0.55 的低波动红利
    #   ⚠️ 只设【上限】，不设下限：
    #     ① 下限已被 VCP 隐含（v2.5 主池 16 只 base_days 最低 13，无一只 <10）；
    #     ② 设 20 会误杀德尔 —— 其平台期 base_days 峰值仅 19（20 日窗口会够到
    #        09-18 的 30.45 修复段，宽度 25.8% 超 20% 阈值）。
    #   11-11 落地：主池 16 → 6 只，剔掉的 10 只均为 base_days=60 满格的
    #     低波动红利股（长电/核电/唐山港/华能/华电/国电/美的/九州通/江苏金租/华测）。
    #   bootstrap：+0.08pp CI[−0.55,+0.73] 含零 —— 属语义修正（从低波动红利切到
    #     中等弹性平台），非显著收益优化；代价是大亏率 10.3%→13.3%、β 0.80→0.91。
    "shape_base_max": 35,
    # --- 资金活跃 ---
    # 【2026-11-12 正式定版 · v2.1】流动性底线 4亿 → 2亿。用户确认「不用管实证，
    #   直接放宽，把这一版作为最新版本」，不再回退；本值即当前唯一生效的成交额门槛。
    #   注意：score() 的成交额对数刻度起点 a_ref 直接读本值，会随之联动 →
    #   池内既有票的资金分与排序会一并重算，不只是「多放进几只票」。
    #   V1_PARAMS 永久冻结在 4亿（v1 旧口径对照，不随 v2.1 定版改动）。
    #   实证留痕：research/yh_amt_relax.py（output/amt_relax_20261112.txt）复核
    #   边际组 2~4亿 更差、配对 CI 全含零 → 统计证据不支持放宽；
    #   本次是【用户主观决策定版】，非回测结论，未来复盘请连同该实证一并阅读。
    "amt30_min": 2e8,        # 30日日均成交额 >= 2亿（当前正式版本）
    # 【2026-11-12 正式定版 · v2.2】换手率门【删除】（用户原话「不要限制换手率，
    #   这个门槛直接删掉」）。None = 不设门；换手仍留在 score() 资金分里作软衰减
    #   （峰值 2.5%、>6% 递减），只影响排序、不再踢人。
    "turn30_min": None,      # None = 不设换手下限
    "turn30_max": None,      # None = 不设换手上限（原 <=6% 已删）
    # --- 前期涨幅上限 ---
    # 【2026-11-12 正式定版 · v2.2】ret60 门【删除】（用户原话「ret60这个60日涨幅
    #   也删掉」）。None = 不设门；ret60 仍留在 score() 趋势分里。
    #   ⚠️ 历史证据备忘（门存在时）：ret60 高 25% 组大亏率 9.9% vs 低 25% 组 2.2%，
    #   8/8 年方向一致——删门后该风险敞口由用户明确接受（VCP 实证边际组
    #   大亏率 10.5%~12.5% 即含此效应）。
    "ret60_max": None,       # None = 不设前期涨幅上限（原 <=19% 已删）
    # --- 基本面（2026-11-12 用户指令：全部删除，不再按基本面筛选）---
    # use_fundamentals=False 时：营收规模门、TTM 盈利门、营收增速门【全部不施加】，
    #   连「必须有财报数据」的数据完整性要求也一并放开（_rev_gate 直接放行）。
    #   保留下来的只有技术面（形态/距年高/成交额）+ 行业黑名单 + ST 剔除。
    #   用户原话：「把基本面的筛选也都删了，不要基本面」。
    "use_fundamentals": False,
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
#     ⚠️ 该「撤回」已于 2026-11-12 被用户指令推翻：门③重新放宽至 2 亿并【正式定版 v2.1】。
#        本节及下方 REGIME_EVIDENCE 的回测数字均按 4 亿口径计算、未重算，仅作沿革参考；
#        当前生效参数一律以 PARAMS / PARAMS_BY_REGIME 中的 2e8 为准。
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
#
# ⚠️ 2026-11-10：日常跑批已【删除市场周期判断】（用户指示）。
#   三周期参数本就完全相同，删除对选股结果零影响；日常流程不再调用 resolve_regime。
#   以下 PARAMS_BY_REGIME / REGIME_EVIDENCE / REGIME_SIZE / resolve_regime
#   仅供 research/ 与 yh_industry_audit.py 的历史分层研究使用，勿再接入日常选股。
#   2026-11-12 定版：三套 amt30_min 已同步为 2e8，与 PARAMS 一致（代码内无 4 亿残留）。
#   2026-11-12 v2.2 定版：三套同步改为 VCP 口径（prox_now_min 0.88 / width30_max 0.50
#   兜底 / turn 门删除 None）；VCP 形态门 shape_* 由 PARAMS 顶层统一携带（三套相同）。
PARAMS_BY_REGIME = {
    # 上涨：距年高 5% 以内 + 窄箱 + 控换手，beta 调整超额 +4.22%，前后半样本与老股样本均通过
    # 上涨：保留营收增速门槛。区块 bootstrap 下点估计单调递增且幅度大
    #   （>=0 -> +0.80pp, >=10 -> +3.93pp, >=20 -> +6.63pp, >=30 -> +10.13pp,
    #    >=40 -> +14.63pp, >=50 -> +17.04pp），但受限于 UP 仅 12 个有效区块，
    #   CI 均含零、未达显著。保留的理由是「方向一致且幅度大」，不是「已证实」。
    "UP": {
        "prox_now_min": 0.88,
        "width30_max": 0.50,
        "amt30_min": 2e8,
        "turn30_min": None,
        "turn30_max": None,
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
        "prox_now_min": 0.88,
        "width30_max": 0.50,
        "amt30_min": 2e8,
        "turn30_min": None,
        "turn30_max": None,
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
        "prox_now_min": 0.88,
        "width30_max": 0.50,
        "amt30_min": 2e8,
        "turn30_min": None,
        "turn30_max": None,
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
    "shape_w10_max": None,  # v1 无 VCP 形态门（v2.2 新增），显式屏蔽防继承
    "shape_ct_max": None,
    "shape_w30_cap": None,
    # v1 无 v2.3 精准形态四门，全部置 None → _precise_gate 返回全 True（不施加）
    "shape_hi_age_max": None,
    "shape_ret10_max": None,
    "shape_c_w20": None,
    "shape_c_w30": None,
    "shape_c_fast": None,
    "shape_h_ratio": None,
    "shape_base_max": None,   # v1 无横盘天数上限
    "amt30_min": 4e8,   # 永久冻结 4亿：v1 旧口径对照，不随 v2.1 定版放宽
    "turn30_min": 1.5,
    "turn30_max": 10.0,
    "ret60_max": None,      # v1 无前期涨幅约束（该门为 v2 于 2026-10-28 新增）
    "rev_yoy_min": 30.0,    # v1 原文条件：营收同比 >= 30%
    "rev_yoy_prev_min": 0.0,
    # v1 是旧口径对照，仍然施加基本面（营收 TTM>=4亿 + TTM 盈利），
    #   否则 v1/v2 对比会混入「是否筛基本面」这个额外变量。2026-11-12 显式冻结。
    "use_fundamentals": True,
    "revenue_min": 4e8,
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
      - 成交额刻度起点随门槛走（a_ref 直接读 amt30_min）。
        2026-11-12 起门槛降到 2亿，刻度起点同步下移，全池资金分与排序会重算。
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

    # 1) 结构分（27）：离一年新高越近、末端越窄越好
    #    v2.2（VCP 形态门启用时）：prox 刻度起点随 prox_now_min 下移，
    #    窄度项改用 width10（10日箱宽，刻度 0~15% = 门 10% 的 1.5 倍）；
    #    未启用时（V1 对照）保持旧刻度：prox 起点 0.90 + width30 按 width30_max 归一。
    pn_min = pr.get("prox_now_min", None)
    if pr.get("shape_w10_max") is not None and "width10" in out.columns:
        prox_lo = (pn_min - 0.03) if pn_min is not None else 0.85
        s_struct = (0.5 * clip01((out["prox_now"] - prox_lo) / (1.0 - prox_lo)) +
                    0.5 * clip01((0.15 - out["width10"]) / 0.15))
    else:
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
    与「要不要设增速门槛」是两件事。

    2026-11-12：p['use_fundamentals'] 为 False 时【完全不看基本面】，直接全放行
    （含「必须有财报数据」这条完整性要求）。"""
    if not p.get("use_fundamentals", True):
        return pd.Series(True, index=df.index)
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
    若面板没有 TTM 字段则回退到累计口径，并在调用方给出提示。

    2026-11-12：p['use_fundamentals'] 为 False 时这道门不施加（全放行）。"""
    if not p.get("use_fundamentals", True):
        return pd.Series(True, index=df.index)
    col = "revenue_ttm" if "revenue_ttm" in df.columns else "revenue"
    return pd.to_numeric(df[col], errors="coerce") >= p["revenue_min"]


def _profit_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """TTM 盈利门槛（剔除亏损票）。2026-11-12：use_fundamentals=False 时不施加。"""
    if not p.get("use_fundamentals", True):
        return pd.Series(True, index=df.index)
    return df["is_loss_ttm"] == 0


def _ret60_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """前期涨幅上限。p['ret60_max'] 为 None 时不设这道门（v1 旧口径 / v2.2 起主池也删除）。
    ret60 缺失时一律放行——缺失是数据完整性问题，不应静默淘汰样本。"""
    thr = p.get("ret60_max", None)
    if thr is None or "ret60" not in df.columns:
        return pd.Series(True, index=df.index)
    return pd.to_numeric(df["ret60"], errors="coerce") <= thr


def _turn_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """换手率门。任一端为 None 即不设该端（2026-11-12 v2.2 起主池两端均为 None = 门删除）。
    换手仍留在 score() 资金分里作软衰减（只影响排序，不踢人）。"""
    m = pd.Series(True, index=df.index)
    lo, hi = p.get("turn30_min", None), p.get("turn30_max", None)
    if lo is not None:
        m &= df["turn30"] >= lo
    if hi is not None:
        m &= df["turn30"] <= hi
    return m


def _shape_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """VCP 多尺度形态门（2026-11-12 v2.2 定版，替代原「w30≤15%」绝对箱宽门）。

    三条件同时满足：
      ① width10 <= shape_w10_max（10日箱宽，末端绝对窄）
      ② contraction = width10/width30 <= shape_ct_max（收缩比，先宽后窄）
      ③ width30 <= shape_w30_cap（30日箱宽极端波动兜底）
    依据：德尔股份式平台的 30 日窗口被前一波拉升污染（w30 中位 44.2%），
    真实收敛在 10 日尺度（w10 中位 9.4%、ct 中位 0.19）。

    回落：shape_w10_max / shape_ct_max 为 None，或面板缺 width10 / contraction 列时，
    回落到旧口径「width30 <= width30_max」（V1 对照走此路径）。"""
    w10m, ctm = p.get("shape_w10_max", None), p.get("shape_ct_max", None)
    if (w10m is None or ctm is None
            or "width10" not in df.columns or "contraction" not in df.columns):
        return df["width30"] <= p["width30_max"]
    m = (pd.to_numeric(df["width10"], errors="coerce") <= w10m) & \
        (pd.to_numeric(df["contraction"], errors="coerce") <= ctm)
    cap = p.get("shape_w30_cap", None)
    if cap is not None:
        m &= df["width30"] <= cap
    return m


def _precise_gate(df: pd.DataFrame, p: Dict) -> pd.Series:
    """精准形态四门（2026-11-12 v2.3 定版，叠加在 v2.2 VCP 基线之上）。

    逐条语义与依据：
      N 新高新鲜  hi250_ago <= shape_hi_age_max（250日最高收盘距今 <= 60 交易日）
         → 杀「回到半年前同一价位的双顶」（南方传媒199日/双环194日/九州通191日）
      R 不追高    ret10 <= shape_ret10_max（8%）
         → 杀「正在突破的追高票」（安图生物 ret20 +20.7%），且不误伤德尔平台（近10日横盘）
      C 逐级收缩  (w10 <= w20×0.70 且 w20 <= w30×0.90) 或 w10 <= shape_c_fast(4%)
         → 区分「逐级收敛」与「断层骤降」；4% 豁免保护长江电力/川投这类一直很窄的票
      H 箱顶不塌  hi10 >= hi30 × shape_h_ratio(0.95)
         → 末端没有脱离箱体上沿，防「箱顶塌陷后的低位窄幅」
      F 横盘天数  base_days <= shape_base_max(35)   【v2.6 新增，只设上限】
         → 剔「横 60 天以上还窄幅趴着」的低波动红利股；下限不设，见 PARAMS 注释

    ⚠️ 刻意不含均线多头（ma60>ma120）：德尔股份平台期 ma60 < ma120，会被永久误杀。

    回落：四个阈值全为 None（V1 对照）或面板缺列时，返回全 True（不施加）。
    任一阈值非 None 但对应列缺失 → 抛 KeyError 由调用方暴露，不静默放行。
    """
    keys = ("shape_hi_age_max", "shape_ret10_max", "shape_c_w20",
            "shape_c_w30", "shape_c_fast", "shape_h_ratio", "shape_base_max")
    if all(p.get(k) is None for k in keys):
        return pd.Series(True, index=df.index)
    need = ["hi250_ago", "ret10", "width10", "width20", "width30", "hi10", "hi30", "base_days"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"精准形态门缺少面板列 {miss}：请重算面板（删除 cache/panel_*.pkl）")

    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    m = pd.Series(True, index=df.index)

    age = p.get("shape_hi_age_max")
    if age is not None:
        m &= num("hi250_ago") <= age
    r10 = p.get("shape_ret10_max")
    if r10 is not None:
        m &= num("ret10") <= r10
    w20k, w30k = p.get("shape_c_w20"), p.get("shape_c_w30")
    if w20k is not None and w30k is not None:
        w10, w20, w30 = num("width10"), num("width20"), num("width30")
        fast = p.get("shape_c_fast")
        cond = (w10 <= w20 * w20k) & (w20 <= w30 * w30k)
        if fast is not None:
            cond = cond | (w10 <= fast)
        m &= cond
    hr = p.get("shape_h_ratio")
    if hr is not None:
        m &= num("hi10") >= num("hi30") * hr
    bm = p.get("shape_base_max")
    if bm is not None:
        # F 横盘天数上限：剔「横 60 天以上还窄幅趴着」的低波动红利股（只设上限，见 PARAMS 注释）
        m &= num("base_days") <= bm
    return m


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
        _shape_gate(df, p) &
        _precise_gate(df, p) &
        (df["amt30"] >= p["amt30_min"]) &
        _turn_gate(df, p) &
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
        pg & _shape_gate(df, p) & _precise_gate(df, p) &
        (df["amt30"] >= p["amt30_min"]) &
        _turn_gate(df, p) &
        _ret60_gate(df, p) &
        _rev_size_gate(df, p) &
        df["board"].notna() & (~df["board"].isin(p["exclude_industries"]))
    )
    # 形态三分：仅标注，不参与筛选（见 form_tag 说明）
    df["form"] = form_tag(df)
    # RS 相对强度（O'Neil RS rating 同义）：当日全市场 ret250 百分位，仅标注。
    # 2026-10-28 诊断：RS>=85 组点估计最高（+4.28%）、胜率 63.3%，但 CI 含零 → 不做门槛。
    if "ret250" in df.columns:
        r250 = pd.to_numeric(df["ret250"], errors="coerce")
        # 口径对齐回测脚本 yh_d3_d6_rs.py：必须在「当日」横截面内排名。
        # 面板里停牌股会沿用旧 bar（date 不是最新日），直接全体 rank 会把旧价混进今日截面。
        # 做法：以最新交易日的 ret250 分布为参照，用经验分位给所有票打分。
        if "date" in df.columns and df["date"].nunique() > 1:
            latest = df["date"].max()
            ref = r250[df["date"] == latest].dropna().sort_values().values
            if len(ref) > 0:
                df["rs_pct"] = (np.searchsorted(ref, r250.values, side="right")
                                / float(len(ref)) * 100.0)
            else:
                df["rs_pct"] = r250.rank(pct=True) * 100.0
        else:
            df["rs_pct"] = r250.rank(pct=True) * 100.0
    else:
        df["rs_pct"] = np.nan
    df["pass"] = struct & _rev_gate(df, p) & _rev_prev_gate(df, p) & _profit_gate(df, p)
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
        if p.get("shape_w10_max") is not None and "width10" in df.columns:
            _ok_shape = (pd.notna(r.get("width10")) and pd.notna(r.get("contraction"))
                         and r["width10"] <= p["shape_w10_max"]
                         and r["contraction"] <= p["shape_ct_max"]
                         and r["width30"] <= p.get("shape_w30_cap", 9.9))
            if not _ok_shape:
                reasons.append(
                    f"VCP 形态破坏：10日箱宽 {r['width10']:.1%}（门≤{p['shape_w10_max']:.0%}）"
                    f" / 收缩比 {r['contraction']:.2f}（门≤{p['shape_ct_max']:.2f}）"
                    f" / 30日箱宽 {r['width30']:.1%}（兜底≤{p.get('shape_w30_cap', 0.5):.0%}）")
        elif r["width30"] > p["width30_max"]:
            reasons.append(f"30日箱宽 {r['width30']:.1%} 超过 {p['width30_max']:.0%}")
        # v2.3 精准形态四门（逐条给出失败原因，便于人工复核持仓）
        if any(p.get(k) is not None for k in
               ("shape_hi_age_max", "shape_ret10_max", "shape_c_w20", "shape_h_ratio",
                "shape_base_max")):
            _age, _r10 = p.get("shape_hi_age_max"), p.get("shape_ret10_max")
            if _age is not None and pd.notna(r.get("hi250_ago")) and r["hi250_ago"] > _age:
                reasons.append(f"新高不新鲜：250日最高收盘在 {int(r['hi250_ago'])} 日前（门≤{_age}日）")
            if _r10 is not None and pd.notna(r.get("ret10")) and r["ret10"] > _r10:
                reasons.append(f"近10日涨幅 {r['ret10']:.1%} 超过 {_r10:.0%}（追高门）")
            _w20k, _w30k = p.get("shape_c_w20"), p.get("shape_c_w30")
            if _w20k is not None and pd.notna(r.get("width20")):
                _fast = p.get("shape_c_fast")
                _cok = (r["width10"] <= r["width20"] * _w20k) and (r["width20"] <= r["width30"] * _w30k)
                if _fast is not None and r["width10"] <= _fast:
                    _cok = True
                if not _cok:
                    reasons.append(
                        f"非逐级收缩：w10 {r['width10']:.1%} / w20 {r['width20']:.1%} / w30 {r['width30']:.1%}")
            _hr = p.get("shape_h_ratio")
            if _hr is not None and pd.notna(r.get("hi10")) and r["hi10"] < r["hi30"] * _hr:
                reasons.append(f"箱顶塌陷：10日高 {r['hi10']:.2f} 低于 30日高 {r['hi30']:.2f} × {_hr:.2f}")
            _bm = p.get("shape_base_max")
            if _bm is not None and pd.notna(r.get("base_days")) and r["base_days"] > _bm:
                reasons.append(
                    f"横盘过久：已窄幅横盘 {int(r['base_days'])} 日（上限 {_bm} 日），属长期低波动而非启动前平台")
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
        _r60 = p.get("ret60_max", None)
        if _r60 is not None and pd.notna(r.get("ret60")) and r["ret60"] > _r60:
            reasons.append(f"近3月涨幅 {r['ret60']:.1%} 超过 {_r60:.0%}（前期涨幅门）")
        if reasons:
            rem_rows.append({"code": c, "name": r["name"], "industry": r["industry"],
                             "close": r["close"], "reasons": "；".join(reasons)})
    removed = pd.DataFrame(rem_rows)

    # --- 打分与排序 ---
    scored = score(df, p)
    pool = scored[scored["pass"]].sort_values("score", ascending=False)
    watch = scored[watch_mask & _profit_gate(scored, p)].sort_values("score", ascending=False)
    story = scored[story_mask].sort_values("score", ascending=False)

    # --- 坑后平台卫星池（2026-11-12 v2.5）---
    # 与主池并列输出、不进主池：PCB 的 ab +0.49% / 大亏 15.9% / β 1.02 明显弱于主池
    # （+1.09% / 10.5% / 0.78），并入主池的配对 bootstrap 为 −0.17pp、CI 含零。
    # 定位是「VCP 抓不到但形态说得通」的第二条腿（德尔股份式：挖坑→修复→横盘 20~30 日）。
    # 口径：prox≥0.88 + 额≥2亿 + PCB + R(不追高) + 行业 + 非ST，不含 VCP/C/H。
    pcb_pool = pd.DataFrame()
    if "pcb_hit" in scored.columns:
        _m = ((scored["prox_now"] >= p.get("prox_now_min", 0.88)) &
              (scored["amt30"] >= p["amt30_min"]) &
              (pd.to_numeric(scored["pcb_hit"], errors="coerce").fillna(0) == 1) &
              scored["board"].notna() &
              (~scored["board"].isin(p["exclude_industries"])))
        _r10 = p.get("shape_ret10_max")
        if _r10 is not None:
            _m &= pd.to_numeric(scored["ret10"], errors="coerce") <= _r10
        pcb_pool = scored[_m].sort_values("score", ascending=False)

    return {"pool": pool, "watch": watch, "story": story, "pcb": pcb_pool,
            "removed": removed, "all_scored": scored}


# ============================== 展示列 ==============================
POOL_COLS = ["name", "industry", "close", "score", "prox_now", "prox", "base_days",
             "width30", "width20", "width10",
             "contraction", "hi250_ago", "ret10", "dd_from_high",
             "amt30", "turn30", "vr20", "rs_pct", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w",
             "pe_ttm", "pb", "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]

# 比例型字段（0~1）→ 显示为百分数
_FRACTION_COLS = {"prox", "prox_now", "width30", "width20", "width10", "contraction", "ret10",
                  "dd_from_high", "pos_in_box", "ret60", "today_pct", "dist_to_trig"}
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
