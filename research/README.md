# research — 研究脚本

这里放策略开发过程中的**研究、校准、诊断脚本**，共 28 个。
它们**不参与日常选股流程**（`run_daily.py` 完全不会用到），只在需要复核历史结论、重新做参数研究时才会跑。

## 怎么运行

**必须用 `-m` 方式**（原因见下）：

```bash
python -m research.yh_regime_calib6
```

## 为什么不能直接 `python research/xxx.py`

这些脚本需要 import 项目根目录的生产模块（如 `yh_datasource`、`yh_panel`）。

- 直接按路径运行时，Python 只把 `research/` 加入模块搜索路径 → `ModuleNotFoundError`
- 用 `-m` 运行时会先执行 `research/__init__.py`，它把**项目根目录和 research 目录**都加进 `sys.path` → 两类 import 都能找到

## 脚本索引

| 类别 | 脚本 | 用途 |
|---|---|---|
| **数据准备** | `yh_fetch_long2.py` | 拉取 2018 年至今全历史日线（重建 `cache/kline_long2.pkl`，首次运行必需） |
| **参数校准** | `yh_calibrate.py` | v1 参数校准：每个条件的超额收益检验 |
| | `yh_regime_calib6.py` | 分周期参数校准（**定稿版**，`--fresh` 之后建议每季度跑一次「三段全胜」复检） |
| | `yh_final_calib2.py` | 终值校准（被 `yh_sample_window`、`yh_window_deep` 依赖） |
| | `yh_sample_window.py` | 样本窗口敏感性 |
| | `yh_window_deep.py` `yh_window_pair.py` | 窗口分段与配对稳定性 |
| **市场周期** | `yh_regime_or_not.py` | 「是否值得分周期」的论证 |
| **箱体 / 位置** | `yh_width_evolution.py` `yh_width_fine.py` `yh_width_vs_vol.py` | 箱宽口径、箱宽-波动率关系 |
| | `yh_prox_fine.py` | 距年高 `prox` 口径的精细化检验 |
| **事件研究** | `yh_eventstudy.py` `yh_eventstudy_long.py` | 事件研究基础框架 |
| | `yh_classic_gap.py` `yh_d3_d6_rs.py` | 缺口与 D3-D6 相对强度 |
| **Beta / 超额** | `yh_beta.py` `yh_beta_deep.py` `yh_beta_ladder.py` | Beta 调整、超额收益分解、分层测试 |
| **营收 / 成长** | `yh_rev_gate_final.py` `yh_rev_regime.py` `yh_ttm_revenue.py` | 营收门槛、TTM 口径、分周期营收 |
| **诊断** | `yh_vol_check.py` `yh_def_check.py` `yh_fix_verify.py` | 波动率、定义、数据修复校验 |
| | `yh_why_not_688018.py` | 个案例外诊断（乐鑫科技为何被箱宽门槛挡住） |
| | `yh_year_stability.py` `yh_asof_1019.py` | 分年度稳定性、指定日期复现 |

## 2026-10-29 清理说明

本次清理移除了 **22 个既无文档引用、也不被任何脚本 import** 的脚本，包括：

- `yh_regime_calib` 的中间版本 2/3/4/5/7/8（仅保留有持续复检用途的 **calib6**）
- `yh_fetch_long`（已被 `yh_fetch_long2` 取代）
- 一次性诊断：`yh_alpha_check` `yh_beta_demo` `yh_blockboot` `yh_effn` `yh_gate_value`
  `yh_growth_attrib` `yh_growth_version` `yh_overlap_diag` `yh_pool_anatomy`
  `yh_why_2broker` `yh_final_calib` `yh_fix_data` `yh_backfill_688018`
  `yh_regime_label`

它们的结论已经固化在 `versions/推导归档_*.md` 与 `versions/思路演进史_v1_to_v2.md` 里。
如需找回某个脚本：

```bash
git show b50e2ef:research/yh_regime_calib8.py > research/yh_regime_calib8.py
```

（`b50e2ef` 是清理前的最后一个提交，完整保留 50 个脚本。）

## 约定

- 新增研究脚本请放本目录，并用 `python -m research.<name>` 运行
- 若产出证据文件，请写入 `output/`（不入库）
- 结论若被采纳进生产逻辑，请把证据摘要写入 `versions/推导归档_*.md`
