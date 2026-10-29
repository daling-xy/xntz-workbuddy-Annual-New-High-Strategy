# 年度新高策略 / Annual New High Strategy

> 由信能投资开发迭代优化的「年度新高策略」股票策略，旨在挖掘 **基本面优质 + 公司营收处于增长逻辑中 + 资金流活跃** 的股票进行投资。

本仓库当前生效版本为 **v2**。策略全貌（参数、口径、打分、局限）见 **[`README_策略说明.md`](README_策略说明.md)**。

---

## 一、策略在做什么

在同时满足以下条件的股票里，按五维打分排序取头部：

- **站在一年高点 5% 以内**（当前价 ≥ 250 日最高收盘 × 0.95）
- **近 30 日窄幅整理**（30 日振幅 ≤ 15%）
- **有承接**（日均成交额 ≥ 4 亿元、换手率 0~6%）
- **营收有规模**（TTM 营收 ≥ 4 亿元/年）
- 剔除 ST / 退市 / 停牌 / 上市不足 250 个交易日，以及 9 类特定行业

本质是**右偏动量策略**：胜率约 38%，收益由右尾少数大赢家驱动，因此**执行纪律比选股本身更重要**。

---

## 二、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖只有三个：`pandas` / `numpy` / `requests`。

### 2. 首次准备数据（只需做一次）

缓存体积约 3.6 GB，**不放进仓库**（可再生），需要在本机重建：

```bash
# ① 拉取 2018 年至今全历史日线（最慢的一步，视网络约 20~60 分钟）
python -m research.yh_fetch_long2

# ② 刷新行情/快照/财报/宽度/周期标签/截面面板（约 3~8 分钟）
python run_daily.py --fresh
```

> 若第 ① 步不是从零开始（已有 `cache/kline_long2.pkl`），可跳过。

### 3. 日常使用

```bash
# 用本地最新缓存跑，秒级出结果，不联网
python run_daily.py

# 先刷新数据再跑，联网，约 3~8 分钟
python run_daily.py --fresh
```

结果输出到 `output/v2/`。

---

## 三、目录结构

```
.
├── run_daily.py          # 【日常入口】检查缓存新鲜度 → 按需刷新 → 调用选股
├── run_screen_v2.py      # 【策略主入口】v2 选股逻辑与输出
├── yh_datasource.py      # 数据源：腾讯行情 / 东财财报 / 通达信指数
├── yh_panel.py           # 截面面板构建
├── yh_factors.py         # 因子计算
├── yh_screen.py          # 筛选、打分、参数表
├── yh_regime.py          # 市场周期判定
├── yh_regime_state.py    # 周期状态
├── yh_rev_fine.py        # 营收数据精算
├── yh_blockboot_all.py   # 分块 bootstrap 检验
├── yh_pos_in_box.py      # 箱体内相对位置
├── yh_tdx.py             # 通达信数据接口
│
├── research/             # 研究脚本（50 个）：参数校准、诊断、事件研究，不参与日常运行
├── archive/              # 历史版本入口（v1）
├── versions/             # v1 / v2 代码快照与推导归档
├── cache/                # 数据缓存（不入库，可再生）
└── output/               # 运行产出（不入库）
```

---

## 四、研究脚本怎么跑

`research/` 下的脚本需要 import 根目录的生产模块，**必须用 `-m` 方式运行**：

```bash
# 正确
python -m research.yh_regime_calib6

# 错误：会报 ModuleNotFoundError: No module named 'yh_datasource'
python research/yh_regime_calib6.py
```

原因：直接用路径运行时，Python 只把 `research/` 加入模块搜索路径，找不到根目录的模块；用 `-m` 则会先执行 `research/__init__.py`，把两个目录都加进去。

---

## 五、文档导航

| 文档 | 内容 |
|---|---|
| [`README_策略说明.md`](README_策略说明.md) | **当前生效的策略全貌**：参数总表、口径、打分、用法、局限 |
| [`versions/思路演进史_v1_to_v2.md`](versions/思路演进史_v1_to_v2.md) | 参数如何一步步变成今天这样，哪些中间结论被推翻 |
| [`versions/推导归档_参数实证_20261027.md`](versions/推导归档_参数实证_20261027.md) | 逐参数的定参依据与证据链 |
| [`versions/推导归档_补充诊断_20261029.md`](versions/推导归档_补充诊断_20261029.md) | 补充诊断记录 |

---

## 六、数据来源

- **行情**：腾讯前复权日线、换手率
- **指数**：通达信五大指数日线
- **财报与快照**：东方财富

---

## 七、免责声明

本项目全部内容均为历史统计与回测结论，**不构成任何投资建议**，据此操作风险自负。
