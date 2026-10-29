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

数据缓存在 `cache/`，约 1 GB，**不放进 git**——GitHub 单文件上限 100 MB，且 git 对二进制是"完整快照"存储而非增量，大缓存会把仓库撑爆。两条路任选：

**方式 A：下载现成数据包（推荐，约 1 分钟）**

最简单——打开 Release 页面，点 `cache_core_20261029.zip` 下载：

https://github.com/daling-xy/xntz-workbuddy-Annual-New-High-Strategy/releases/tag/data-20261029

命令行（`gh` 已登录时）：

```bash
gh release download data-20261029 \
  -R daling-xy/xntz-workbuddy-Annual-New-High-Strategy \
  -p "cache_core_*.zip" -D .
unzip cache_core_*.zip -d cache/
python run_daily.py
```

> ⚠️ 私有仓库的 Release 附件**必须登录才能下载**，直接 `curl` 网页链接会返回 404。
> Windows 上用 curl 走 API 时若报 `CRYPT_E_REVOCATION_OFFLINE`，加 `--ssl-no-revoke`。

> 数据包解压后约 1040 MB，**不含** `events_*.pkl`（1.6 GB，仅参数研究用）。
> 缺失时「严格度阶梯」的历史实证栏显示 `—`，选股主流程不受影响。

**方式 B：从公开数据源自行重建（约 30~70 分钟）**

```bash
# ① 拉取 2018 年至今全历史日线（最慢的一步，视网络 20~60 分钟）
python -m research.yh_fetch_long2

# ② 刷新行情/快照/财报/宽度/周期标签/截面面板（3~8 分钟）
python run_daily.py --fresh
```

> 若 `cache/kline_long2.pkl` 已存在，第 ① 步可跳过。
> 若本机未配置通达信 `tdx-connector`，第 ② 步会在重算周期标签时失败——改用方式 A。

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

12 个生产模块共 3159 行，按「一件事一个文件」划分。**文件名即职责**，便于人和 AI 快速定位：

| 文件 | 行数 | 职责 | 上游调用者 |
|---|---|---|---|
| `run_daily.py` | 311 | 日常入口：查缓存新鲜度 → 按需刷新 → 调选股 | 人 |
| `run_screen_v2.py` | 256 | 策略主入口：周期判定 → 筛选 → 打分 → 输出 | `run_daily` |
| `yh_screen.py` | 511 | 筛选条件、五维打分、参数总表 | `run_screen_v2` |
| `yh_panel.py` | 198 | 截面面板构建（把行情拼成选股用的表） | `run_screen_v2` |
| `yh_factors.py` | 180 | 因子计算（prox / width30 / amt30 / turn30 等） | `yh_panel` |
| `yh_datasource.py` | 400 | **HTTP 数据源**：腾讯行情、东财财报/快照/股东 | 多个 |
| `yh_tdx.py` | 106 | **通达信 MCP 客户端**：五大指数日线 | `yh_regime` |
| `yh_regime.py` | 204 | 市场周期判定（UP / CHOP / DOWN） | `yh_regime_state` |
| `yh_regime_state.py` | 213 | 周期状态读写缓存 | `run_screen_v2` |
| `yh_rev_fine.py` | 329 | 营收数据精算（TTM 口径） | `yh_pos_in_box`、`yh_blockboot_all` |
| `yh_blockboot_all.py` | 270 | 分块 bootstrap 显著性检验 | `yh_pos_in_box` |
| `yh_pos_in_box.py` | 181 | 箱体内相对位置 + 严格度阶梯的历史实证 | `run_screen_v2` |

**依赖方向**（箭头表示调用，从入口往下）：

```
run_daily
  └─ run_screen_v2
       ├─ yh_screen
       ├─ yh_panel ── yh_factors
       ├─ yh_regime_state ── yh_regime ── yh_tdx
       ├─ yh_datasource          (腾讯 / 东财)
       └─ yh_pos_in_box ── yh_blockboot_all ── yh_rev_fine
```

其余目录：

```
research/   研究脚本（50 个）：参数校准、诊断、事件研究，不参与日常运行
archive/    历史版本入口（v1）
versions/   v1 / v2 代码快照与推导归档
tools/      运维脚本（pack_data.py 打包数据）
cache/      数据缓存（不入库，可再生）
output/     运行产出（不入库）
```

### 为什么 `yh_tdx.py` 和 `yh_datasource.py` 不合并

两者协议层完全不同，合并会互相拖累：

| | `yh_datasource.py` | `yh_tdx.py` |
|---|---|---|
| 协议 | 公开 HTTP 接口 | MCP-over-HTTP（SSE + session 握手） |
| 依赖 | 只需 `requests` | 本机 `tdx-connector` 服务 + `CODEBUDDY_MCP_CONFIG` 环境变量 |
| 失效影响 | 拉不到某只票的数据 | 整条指数/周期链路不可用 |
| 重试策略 | 简单 3 次重试 | 需维护 session id、重新 initialize |

一个数据源一个文件是适配器模式的标准做法：新增源只加文件、不动老代码；一个源出问题不牵连另一个。合并成 500 多行的混合文件，反而更难定位问题——**对人和 AI 都是**。

> 通达信 MCP 的依赖情况见 [第六节](#六通达信-mcp-依赖说明组员必读)。结论：**日常选股不需要它。**

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

## 六、通达信 MCP 依赖说明（组员必读）

### 结论先说

**日常选股不需要通达信。** 只有显式加 `--idx` 参数时才需要。

### 它是怎么连的

`yh_tdx.py` 通过 MCP-over-HTTP(SSE) 连接本机的 `tdx-connector` 服务，地址从环境变量读取：

```python
# yh_tdx.py:20
cfg = json.loads(os.environ["CODEBUDDY_MCP_CONFIG"])
```

这个变量由 WorkBuddy / CodeBuddy 注入。**在普通终端、组员机器上都不存在**，所以在那些环境里调用会失败。

### 影响范围（比听起来小得多）

| 场景 | 需要通达信？ | 说明 |
|---|---|---|
| `python run_daily.py` | ❌ | 只读本地缓存 |
| `python run_daily.py --fresh` | ❌ **默认不需要** | 指数面板沿用 `cache/tdx_idx_*.pkl` |
| `python run_daily.py --fresh --idx` | ✅ 需要 | 只有这个组合会强制重拉五大指数 |
| **周期判定 UP/CHOP/DOWN** | ❌ | **只依赖市场宽度 `breadth`**，指数仅作观察字段 |

`run_daily.py` 的刷新流程（第 164~175 行）对指数拉取失败**已内置降级**：捕获异常 → 沿用旧 `index_panel.pkl` → 两者都没有就置空，并打印提示，**不中断刷新**。

### 你会看到的输出长这样

```
[2/4] 重拉通达信指数 ...
  [降级] 通达信指数拉取失败：'CODEBUDDY_MCP_CONFIG'
  已沿用旧指数面板，最新 2026-10-28（指数仅作观察字段，不影响周期判定）
```

**看到这行不用处理**，后面的选股会照常跑完。

### 如果要完整启用通达信

1. 本机运行 `tdx-connector`（MCP-over-HTTP 服务）
2. 设环境变量指向它：

```bash
export CODEBUDDY_MCP_CONFIG='{"mcpServers":{"tdx-connector":{"url":"http://127.0.0.1:PORT/sse"}}}'
```

3. 自检：

```bash
python -c "import yh_tdx; print('工具数:', len(yh_tdx.TdxMcp().list_tools()))"
```

能打印工具数 = 连通；报 `KeyError: 'CODEBUDDY_MCP_CONFIG'` = 环境变量没设。

### 为什么不改代码

- 降级逻辑**已经存在**，再加一层是重复
- 周期判定不依赖指数，缺指数不改变策略结论
- 动数据接入层收益低、风险高

---

## 七、数据来源

- **行情**：腾讯前复权日线、换手率
- **指数**：通达信五大指数日线
- **财报与快照**：东方财富

---

## 八、免责声明

本项目全部内容均为历史统计与回测结论，**不构成任何投资建议**，据此操作风险自负。
