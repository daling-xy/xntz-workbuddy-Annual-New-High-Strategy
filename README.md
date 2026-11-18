# 年度新高策略 v3 · Annual New High Strategy

> 在**低波动、筹码稳定、中期向上**的股票里，买**贴近年高**的那一批。
> 本仓库当前生效版本 = **v3**（2026-11-13 定版，纸上跟踪中，不构成投资建议）。

## 快速开始

```bash
pip install -r requirements.txt

# 1) 收盘后刷新数据（会拉全A日线320日 + 业绩报表/board + 快照）
python run_daily.py --check          # 先探测数据源是否已出当日日线
python run_daily.py --fresh --asof 0 # 收盘后跑（asof=0 用最新bar；盘中用默认 asof=1）

# 2) 也可以不刷新，直接用本地缓存跑选股
python run_daily.py --asof 0
```

输出：`output/v3/daily_rank_v3_YYYYMMDD.csv`

## 目录结构

| 路径 | 内容 |
|---|---|
| `run_daily.py` | 每日入口：刷新行情/board → 跑 v3 选股 |
| `yh_datasource.py` | 数据源（腾讯日线、东财快照/业绩报表/F10） |
| `v3/` | **v3 选股主体**：`daily_rank_v3.py`、`README_v3.md`、证据脚本、行业身份证与 board 快照 |
| `reminder/` | 独立提醒工具（与本策略无关） |
| `CORE_RULES.md` | **买卖点权威口径**（2026-11-18 定版，见下） |
| `v3/core_backtest.py` | 按权威买卖点做真实交易口径回测（日线近似） |

## v3 规则（一句话版）

1. **宇宙**：当日成交额 ≥2 亿；vol20（20 日年化波动率）≤1.0×自身过去一年均值 且 ≤35%；非涨停、非 ST/退、非 9 类行业黑名单
2. **条件**：prox（距 250 日最高收盘）≥0.90；ret10 ≤8%；turn30 ≤5%；ret60 ≥0
3. **排序**：prox；分档 A ≥0.98 / B ≥0.95 / C ≥0.90（不设名额）
4. **每日输出（交易卡）**：`tier / prox / code / name / close / box_top / buy_lo(×1.07) / buy_hi(×1.15) / dist_trigger / stop_px / box_w30 / status / volratio / ret10 / ret60 / turn30 / amt_yi / industry`

完整定义、阈值来源、双向样本外结果见 **[v3/README_v3.md](v3/README_v3.md)**。

## 核心规则（交易口径，权威）

买卖点以 **[CORE_RULES.md](CORE_RULES.md)** 为唯一权威口径（2026-11-18 定版）：

1. **剔除**：收盘 > 第一次突破触发价 × 1.20 → 移出；收盘 < 30 日最高收盘 × 0.80 → 移出；
2. **买入**：连续两个交易日内，分时价进入「30 日最高收盘 × **1.07–1.15**」区间 → 买入，不补仓；
3. **卖出**：① 14:30 低于买入价 87% 止损；② 盈利 >30% 后，当日涨幅 ≥9% 卖一半 / 连续两日涨幅 ≤3% 在 14:45 后卖一半；③ 跌幅 >7% 且 10:30 前量 ≥ 前日 70%（或 14:30 量 ≥ 前日）→ 全部卖出。

真实交易口径回测脚本：`python v3/core_backtest.py`。


## 版本说明

- **v3（当前）**：推倒重来版，独立于 v2.6；条件更少、只保留验证过的 prox 排序。
- **v2.6（已归档）**：旧版（VCP 主池 + 坑后平台卫星池 + 打分体系）。
  历史在 git 里，可用 `git checkout v2.6-final` 取回，tag 名 `v2.6-final`。

> 本项目所有结论均为统计回测，**不构成投资建议**。