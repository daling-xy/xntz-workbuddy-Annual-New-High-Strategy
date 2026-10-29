# research — 研究脚本

这里放的是策略开发过程中的**研究、校准、诊断脚本**，共 50 个。
它们**不参与日常选股流程**（`run_daily.py` 完全不会用到），只在需要重新做参数研究、复查历史结论时才会跑。

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
| **数据准备** | `yh_fetch_long.py` `yh_fetch_long2.py` | 拉取 2018 年至今全历史日线（重建 `cache/kline_long2.pkl`） |
| | `yh_fix_data.py` `yh_fix_verify.py` `yh_def_check.py` | 数据修复与口径校验 |
| | `yh_backfill_688018.py` | 单只股票历史回填 |
| **参数校准** | `yh_calibrate.py` `yh_final_calib.py` `yh_final_calib2.py` | 早期参数校准迭代 |
| | `yh_regime_calib.py` ~ `yh_regime_calib8.py` | 分周期参数校准（编号越大越新，**calib8 为最终版**） |
| | `yh_sample_window.py` `yh_window_deep.py` `yh_window_pair.py` | 样本窗口与分段稳定性检验 |
| **市场周期** | `yh_regime_label.py` `yh_regime_or_not.py` | 周期标签生成与「是否分周期」的论证 |
| **箱体 / 位置** | `yh_width_evolution.py` `yh_width_fine.py` `yh_width_vs_vol.py` | 箱宽口径与箱宽-波动率关系 |
| | `yh_prox_fine.py` | 距年高 prox 口径的精细化检验 |
| **事件研究** | `yh_eventstudy.py` `yh_eventstudy_long.py` | 事件研究基础框架 |
| | `yh_blockboot.py` | 分块 bootstrap 显著性 |
| | `yh_classic_gap.py` `yh_d3_d6_rs.py` | 缺口与 D3-D6 相对强度分析 |
| **Beta / 超额** | `yh_beta.py` `yh_beta_deep.py` `yh_beta_demo.py` `yh_beta_ladder.py` | Beta 调整、超额收益分解、分层测试 |
| | `yh_alpha_check.py` | Alpha 显著性检验 |
| **营收 / 成长** | `yh_rev_gate_final.py` `yh_rev_regime.py` `yh_ttm_revenue.py` | 营收门槛与 TTM 口径 |
| | `yh_growth_attrib.py` `yh_growth_version.py` | 成长因子归因与版本对比 |
| **诊断** | `yh_effn.py` `yh_gate_value.py` `yh_vol_check.py` | 趋势效率、门槛价值、波动率检查 |
| | `yh_overlap_diag.py` `yh_pool_anatomy.py` | 候选池重叠与结构解剖 |
| | `yh_why_2broker.py` `yh_why_not_688018.py` | 个案例外诊断 |
| | `yh_year_stability.py` `yh_asof_1019.py` | 分年度稳定性、指定日期复现 |

## 约定

- 新增研究脚本请放本目录，并用 `python -m research.<name>` 运行
- 脚本若产出证据文件，请写入 `output/`（不入库）
- 结论若被采纳进生产逻辑，请把证据摘要写入 `versions/推导归档_*.md`
