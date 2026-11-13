# -*- coding: utf-8 -*-
"""
v2 主入口（已含前置数据修复与口径修正）：单一参数集选股（2026-11-10 起不再做市场周期判断），
并与 v1（全局单一参数、旧口径）对比。

相对上一版 v2 的三处修正（证据见 output/final_calib2.txt、def_check.txt、vol_check.txt、fix_verify.txt）：
  1) prox 口径：改用 prox_now = 当前收盘 / 250日最高收盘。
     旧口径用「近30日最高/年高」，会放行「从年高回落 15% 后横盘」的票
     —— 中信证券、财通证券就是这么混进来的。
  2) 成交额门槛：上一版曾从 4 亿降到 2 亿，修复时变性后证据不稳定，撤回为 4 亿。
     ⚠️ 该撤回已于 2026-11-12 被用户指令推翻：门③【正式定版】为 2 亿（当前唯一生效值）。
        历史实证数字仍按 4 亿口径，分歧留档 output/amt_relax_20261112.txt。
  3) 箱宽口径澄清：箱宽是「30日区间振幅」，不是波动率；
     箱宽 15% ≈ 日波动中位 1.73% / 年化 27.3%。

  python run_screen_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_panel import build_panel                                    # noqa: E402
from yh_screen import (screen, fmt, PARAMS,                          # noqa: E402
                       params_for, V1_PARAMS)
from yh_datasource import fetch_holders_many                        # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
# 2026-11-12：支持 YH_OUT_DIR 环境变量改输出目录（目标 CSV 被 Excel/预览占用时
# 先落临时目录验证，文件关闭后再补拷，避免 PermissionError 中断整轮跑批）。
OUT_DIR = os.environ.get("YH_OUT_DIR") or os.path.join(BASE, "output", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

# 2026-10-29 更新为用户当前实际持仓（雷赛智能/瑞芯微/美的集团/中远海控/春风动力/沪电股份）
HOLDINGS = ["002979", "603893", "000333", "601919", "603129", "002463"]

PARAMS_ORDER = ["prox_now", "prox", "width30", "width20", "width10", "contraction",
                "base_days", "hi250_ago", "ret10", "dd_from_high", "amt30", "turn30",
                "vr20", "rs_pct", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w", "pe_ttm",
                "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]


def main():
    # ---------- 1. 参数 ----------
    # 2026-11-10 起：删除市场周期判断（用户指示）。
    #   删除后选股结果【完全不变】：UP/CHOP/DOWN 三套参数经 2026-10-27 实证后
    #   结构本就完全相同（距年高 0.95 / 箱宽 15% / 换手 ≤6% / 营收增速关闭）。
    #   成交额门槛 2026-11-12 由 4亿 正式定版放宽至 2亿（v2.1），三套已同步；
    #   周期标签只影响打印的「建议参与强度」仓位提示，不影响任何一道门。
    #   历史回测中的 regime 分层仅作研究归档用途（research/ 与 versions/），
    #   日常跑批不再读取 regime_state.pkl、不再判定市场状态。
    p2 = params_for("CHOP")   # 三周期参数统一，取任一套；regime 键无下游消费
    p2.pop("regime", None)

    # asof_offset: 1=倒数第2根K线（盘中安全，默认）；0=最新一根（收盘后，bar已定型）。
    # 由 run_daily.py --asof 通过环境变量传入，保证刷新与筛选用同一个基准日。
    _asof = int(os.environ.get("YH_ASOF", "1"))
    panel = build_panel(asof_offset=_asof, days=320)
    print(f"全市场有效样本: {len(panel)} 只，结构评估基准日: {panel['date'].max()}")
    print()

    # ---------- 2. v2（修正版）与 v1（旧口径）并行 ----------
    res_v2 = screen(panel, params=p2, holdings=HOLDINGS)
    res_v1 = screen(panel, params=dict(V1_PARAMS), holdings=HOLDINGS)
    pool2, watch2, story2 = res_v2["pool"], res_v2["watch"], res_v2["story"]
    pool1, watch1 = res_v1["pool"], res_v1["watch"]

    p1 = dict(V1_PARAMS)

    def gate_line(tag, p):
        pn = p.get("prox_now_min")
        ps = (f"当前价/年高>={pn:.2f}" if pn is not None
              else f"近30日高/年高>={p['prox_min']:.2f}")
        # v2.2：VCP 形态门启用时展示多尺度口径；否则旧箱宽口径
        if p.get("shape_w10_max") is not None:
            shape_s = (f"10日箱宽<={p['shape_w10_max']:.0%} 收缩比<={p['shape_ct_max']:.2f} "
                       f"30日箱宽兜底<={p['shape_w30_cap']:.0%}")
        else:
            shape_s = f"箱宽<={p['width30_max']:.0%}"
        # v2.3：精准形态门（N/R/C/H），逐条条件化展示；v2.5 起 N 已删；v2.6 加 F 横盘上限
        if p.get("shape_hi_age_max") is not None:
            shape_s += f" | 新高<={p['shape_hi_age_max']:.0f}日"
        if p.get("shape_ret10_max") is not None:
            shape_s += (f" | 近10日涨幅<={p['shape_ret10_max']:.0%}"
                        f" 逐级收缩 箱顶不塌×{p['shape_h_ratio']:.2f}")
        if p.get("shape_base_max") is not None:
            shape_s += f" 横盘<={p['shape_base_max']:.0f}日"
        t_lo, t_hi = p.get("turn30_min"), p.get("turn30_max")
        turn_s = ("换手门已删" if t_lo is None and t_hi is None
                  else f"换手 {t_lo:.1f}~{t_hi:.0f}%")
        return (f"{tag}: {ps}  {shape_s}  "
                f"额>={p['amt30_min']/1e8:.0f}亿  {turn_s}")

    print("=" * 120)
    print("年度新高策略 v2.6 | 2026-11-12 定版 | VCP 形态门（w10≤10% & ct≤0.45 & w30≤50%）"
          "+ 精准形态 R/C/H/F（近10日涨幅≤8% / 逐级收缩 / 箱顶不塌 / 横盘≤35日）| prox≥0.88"
          " | N(新高新鲜)已删、换手率门与 ret60 门已删、基本面筛选全部删除"
          " | 另出「坑后平台」卫星池")
    print("=" * 120)
    print("参数对比")
    print("=" * 120)
    print("  " + gate_line("v1（全局单一，旧口径）", p1))
    print("  " + gate_line("v2（修正后）          ", p2))
    print()
    print("  ※ 基本面：【全部删除】（2026-11-12 用户指令）。现行口径不再施加")
    print("    营收 TTM≥4亿、TTM 盈利、营收增速门槛，也不再要求必须有财报数据；")
    print("    保留的只有技术面（形态/距年高/成交额）+ 行业黑名单 + ST 剔除。")
    print()

    # ---------- 3. 基金持股增强 ----------
    cand = list(pool2.index[:30]) + list(watch2.index[:15]) + list(story2.index[:15])
    if cand:
        try:
            holders = fetch_holders_many(cand, workers=10)
            for d in (pool2, watch2, story2, res_v2["all_scored"]):
                d["fund_holders"] = [holders.get(c, {}).get("fund_count", np.nan) for c in d.index]
        except Exception as e:
            print("基金股东查询失败（不影响主流程）:", e)

    for d in (pool2, watch2, story2, pool1, watch1):
        if "last_price" in d.columns and "trig_box" in d.columns:
            d["dist_to_trig"] = d["last_price"] / d["trig_box"] - 1.0

    # ---------- 4. 输出 ----------
    cols = ["form", "name", "industry", "close", "score", "fund_holders", "dist_to_trig"] + PARAMS_ORDER
    print("=" * 120)
    print(f"==== v2 核心候选池（共 {len(pool2)} 只）====")
    print("=" * 120)
    print(fmt(pool2, cols=cols, n=40).to_string() if len(pool2) else "  （空）")
    print()

    # ---------- 4.5 坑后平台卫星池（2026-11-12 v2.5）----------
    pcb2 = res_v2.get("pcb", pd.DataFrame())
    print("=" * 120)
    print(f"==== 坑后平台卫星池（共 {len(pcb2)} 只，独立于主池，不参与打分排序之外的任何门槛）====")
    print("=" * 120)
    print("  形态：长期横盘 → 挖坑(≥25%) → 快速修复(回到坑前高 90%) → 横盘 10~45 日(宽≤30%)")
    print("  实证：ab +0.49% / 大亏 15.9% / β 1.02，弱于主池，故只做并列展示（见 output/pcb_20261112.txt）")
    if len(pcb2):
        show = pd.DataFrame({
            "名称": pcb2["name"].values,
            "行业": pcb2["industry"].values,
            "收盘": pcb2["close"].round(2).values,
            "评分": pcb2["score"].values,
            "距年高": (pcb2["prox_now"] * 100).round(1).values,
            "坑深%": (pcb2["pcb_dd"] * 100).round(1).values,
            "横盘日": pcb2["pcb_days"].astype(int).values,
            "平台宽%": (pcb2["pcb_plat_w"] * 100).round(1).values,
            "长期箱宽%": (pcb2["pcb_box_w"] * 100).round(1).values,
            "额(亿)": (pcb2["amt30"] / 1e8).round(1).values,
            "近10日%": (pcb2["ret10"] * 100).round(1).values,
        })
        print(show.to_string(index=False))
    else:
        print("  （空）")
    print()

    print(f"==== v1 核心候选池（旧口径，共 {len(pool1)} 只，用于对照）====")
    print(fmt(pool1, cols=["name", "industry", "close", "score", "prox_now", "prox",
                           "width30", "amt30", "turn30", "rev_yoy"], n=40).to_string()
          if len(pool1) else "  （空）")
    print()

    s1, s2 = set(pool1.index), set(pool2.index)
    print(f"  池变动：v2 新增 {len(s2 - s1)} 只 / 保留 {len(s1 & s2)} 只 / v1 独有被剔除 {len(s1 - s2)} 只")
    if s1 - s2:
        print("  被 v2 剔除的 v1 成员：", "、".join(f"{pool1.at[c,'name']}({c})" for c in sorted(s1 - s2)))
    if s2 - s1:
        print("  v2 新增成员：", "、".join(f"{pool2.at[c,'name']}({c})" for c in sorted(s2 - s1)))
    print()

    if len(watch2):
        print(f"==== v2 观察池（营收同比 15%~30%，共 {len(watch2)} 只）====")
        print(fmt(watch2, cols=cols, n=25).to_string())
        print()
    if len(story2):
        print(f"==== v2 故事票（营收>=30% 增长但亏损，共 {len(story2)} 只）====")
        print(fmt(story2, cols=cols, n=20).to_string())
        print()

    # ---------- 5. 持仓检查 ----------
    print("==== 持仓 %d 只：v2 剔除规则检查 ====" % len(HOLDINGS))
    if len(res_v2["removed"]):
        for _, r in res_v2["removed"].iterrows():
            print(f"  [移除] {r['code']} {r['name']}（{r['industry']}）: {r['reasons']}")
    else:
        print("  持仓均未触发剔除规则（详见下方逐票诊断）。")
    print()

    print("==== 持仓 %d 只逐项诊断（对照 v2 门槛）====" % len(HOLDINGS))
    for c in HOLDINGS:
        if c not in panel.index:
            continue
        r = panel.loc[c]
        prox_now = float(r["prox_now"]) if "prox_now" in panel.columns else 1.0 + float(r["dd_from_high"])
        # v2.2：箱宽门已换成 VCP 形态门；换手门已删除（turn30_max 为 None 时不展示）
        _shape_ok = (r["width10"] <= p2["shape_w10_max"]
                     and r["contraction"] <= p2["shape_ct_max"]
                     and r["width30"] <= p2.get("shape_w30_cap", 9.9))
        checks = [
            ("VCP形态(10日宽<=%.0f%%/收缩<=%.2f)" % (p2["shape_w10_max"] * 100, p2["shape_ct_max"]),
             _shape_ok,
             "10日宽%.1f%% 收缩%.2f 30日宽%.1f%%" % (r["width10"] * 100, r["contraction"], r["width30"] * 100)),
            ("当前价/年高>=%.2f" % p2["prox_now_min"], prox_now >= p2["prox_now_min"], f"{prox_now:.2f}"),
            ("30日均额>=%.0f亿" % (p2["amt30_min"] / 1e8), r["amt30"] >= p2["amt30_min"], f"{r['amt30']/1e8:.1f}亿"),
            ("行业未剔除", r["board"] not in p2["exclude_industries"], str(r["board"])),
        ]
        # v2.3 精准形态四门（N/R/C/H）逐项诊断
        _age = p2.get("shape_hi_age_max")
        if _age is not None:
            checks.insert(1, ("新高<=%d日" % _age, r["hi250_ago"] <= _age, "%d日" % int(r["hi250_ago"])))
        _r10m = p2.get("shape_ret10_max")
        if _r10m is not None:
            checks.insert(2, ("近10日涨幅<=%.0f%%" % (_r10m * 100), r["ret10"] <= _r10m,
                              "%+.1f%%" % (r["ret10"] * 100)))
        _w20k, _w30k = p2.get("shape_c_w20"), p2.get("shape_c_w30")
        if _w20k is not None:
            _cok = ((r["width10"] <= r["width20"] * _w20k and r["width20"] <= r["width30"] * _w30k)
                    or (p2.get("shape_c_fast") is not None and r["width10"] <= p2["shape_c_fast"]))
            checks.insert(3, ("逐级收缩", _cok,
                              "w10 %.1f%% / w20 %.1f%% / w30 %.1f%%" % (
                                  r["width10"] * 100, r["width20"] * 100, r["width30"] * 100)))
        _hr = p2.get("shape_h_ratio")
        if _hr is not None:
            checks.insert(4, ("箱顶不塌(hi10>=hi30×%.2f)" % _hr, r["hi10"] >= r["hi30"] * _hr,
                              "10日高 %.2f / 30日高 %.2f" % (r["hi10"], r["hi30"])))
        _bm = p2.get("shape_base_max")
        if _bm is not None and pd.notna(r.get("base_days")):
            checks.insert(5, ("横盘<=%d日" % _bm, r["base_days"] <= _bm,
                              "已横盘 %d 日" % int(r["base_days"])))
        _t = p2.get("turn30_max", None)
        if _t is not None:
            checks.append(("换手<=%.0f%%" % _t, r["turn30"] <= _t, f"{r['turn30']:.2f}%"))
        # 前期涨幅门（2026-10-28 起生效）：ret60_max 为 None 时不展示该行
        _r60 = p2.get("ret60_max", None)
        if _r60 is not None:
            checks.append(("近3月涨幅<=%.0f%%" % (_r60 * 100),
                           r["ret60"] <= _r60, "%.1f%%" % (r["ret60"] * 100)))
        # 营收增速门槛：关闭（2026-10-27 起，见 yh_screen.PARAMS 与推导归档）
        _thr = p2.get("rev_yoy_min", None)
        if _thr is None:
            checks.append(("营收同比(不设门槛)", True, f"{r['rev_yoy']:.1f}%"))
        else:
            checks.append(("营收同比>=%.0f%%" % _thr, r["rev_yoy"] >= _thr, f"{r['rev_yoy']:.1f}%"))
            checks.append(("上期营收同比>0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"))
        flag = " ".join(("✓" if ok else "✗") + n for n, ok, _v in checks)
        print(f"  {c} {r['name']}")
        print(f"     {flag}")
        print(f"     距一年新高 {r['dd_from_high']:.1%} | 近30日高/年高 {r['prox']:.2f} | "
              f"箱内位置 {r['pos_in_box']:.0%} | 近60日 {r['ret60']:.1%} | PE {r['pe_ttm']:.0f} | "
              f"亏损={bool(r['is_loss'])}")
        print(f"     250日高点距今 {int(r['hi250_ago'])} 日 | 近10日 {r['ret10']:+.1%} | "
              f"箱宽 w10 {r['width10']:.1%} / w20 {r['width20']:.1%} / w30 {r['width30']:.1%}")
    print()

    # ---------- 5.5 严格度阶梯（v2.2 口径）----------
    scored = res_v2["all_scored"]
    if "score" in scored.columns:
        ex = set(p2["exclude_industries"])
        # 2026-11-12：基本面全删 → 阶梯里也不再施加 营收规模 / 盈利 / 增速
        if p2.get("use_fundamentals", True):
            _rcol = "revenue_ttm" if "revenue_ttm" in scored.columns else "revenue"
            _size = pd.to_numeric(scored[_rcol], errors="coerce") >= p2["revenue_min"]
            _thr = p2.get("rev_yoy_min", None)
            _rev = (scored["rev_yoy"].notna() if _thr is None
                    else (scored["rev_yoy"].notna() & (scored["rev_yoy"] >= _thr) &
                          (scored["rev_yoy_prev"] > 0)))
            _profit = (scored["is_loss_ttm"] == 0) if "is_loss_ttm" in scored.columns else True
        else:
            _size = _rev = _profit = pd.Series(True, index=scored.index)
        # v2.2：换手门与 ret60 门已删除（用户指令），不再进入漏斗
        gate = (_rev & _size & _profit & scored["board"].notna() &
                (~scored["board"].isin(ex)) &
                (scored["amt30"] >= p2["amt30_min"]))
        print("==== 严格度阶梯（v2.6 口径：当前价/年高 × VCP形态门 × R/C/H/F；"
              "已含 额>=%.0f亿，换手/ret60/基本面 全部不设）====" % (p2["amt30_min"] / 1e8))
        # 档位 = prox × (10日箱宽, 收缩比)，均含 30日箱宽≤50% 兜底 + 精准形态四门。
        # 历史实证（2026-11-12，research/yh_shape_precise.py + yh_vcp_capture.py）：
        #   v2.2 基线 ab +0.75%/大亏 11.2% → 叠加 N+R+C+H 后 ab +1.07%/大亏 11.5%；
        #   配对 bootstrap 点估计 −0.34pp、90%CI[−0.92,+0.18] 含零 → 形态更干净但优势不显著。
        ladder = [
            ("v2.6 定版  prox>=0.88  10日宽<=10% 收缩<=0.45", 0.88, 0.10, 0.45),
            ("  收紧一档 prox>=0.90  10日宽<=8%  收缩<=0.35（VCP-A）", 0.90, 0.08, 0.35),
            ("  收紧两档 prox>=0.95  10日宽<=8%  收缩<=0.35", 0.95, 0.08, 0.35),
            ("  放宽一档 prox>=0.85  10日宽<=12% 收缩<=0.45（VCP-C）", 0.85, 0.12, 0.45),
        ]
        # v2.3 四门掩码（与 yh_screen._precise_gate 同口径），用于对照「加/不加」差异
        _p4 = pd.Series(True, index=scored.index)
        if p2.get("shape_hi_age_max") is not None:
            _p4 &= pd.to_numeric(scored["hi250_ago"], errors="coerce") <= p2["shape_hi_age_max"]
        if p2.get("shape_ret10_max") is not None:
            _p4 &= pd.to_numeric(scored["ret10"], errors="coerce") <= p2["shape_ret10_max"]
        if p2.get("shape_c_w20") is not None:
            _w10 = pd.to_numeric(scored["width10"], errors="coerce")
            _w20 = pd.to_numeric(scored["width20"], errors="coerce")
            _w30s = pd.to_numeric(scored["width30"], errors="coerce")
            _c = ((_w10 <= _w20 * p2["shape_c_w20"]) & (_w20 <= _w30s * p2["shape_c_w30"]))
            if p2.get("shape_c_fast") is not None:
                _c |= (_w10 <= p2["shape_c_fast"])
            _p4 &= _c
        if p2.get("shape_h_ratio") is not None:
            _p4 &= (pd.to_numeric(scored["hi10"], errors="coerce")
                    >= pd.to_numeric(scored["hi30"], errors="coerce") * p2["shape_h_ratio"])
        _p3 = _p4.copy()          # R/C/H（不含 F 横盘天数上限），用于拆分 F 的独立贡献
        if p2.get("shape_base_max") is not None and "base_days" in scored.columns:
            _p4 &= pd.to_numeric(scored["base_days"], errors="coerce") <= p2["shape_base_max"]
        rows = []
        for tag, px, w10, ct in ladder:
            base_m = (gate & (scored["prox_now"] >= px)
                      & (scored["width10"] <= w10) & (scored["contraction"] <= ct)
                      & (scored["width30"] <= 0.50))
            d_all = scored[base_m].sort_values("score", ascending=False)
            d3 = scored[base_m & _p3]
            d = scored[base_m & _p4].sort_values("score", ascending=False)
            rows.append({"档位": tag, "入选": len(d),
                         "不加 RCH(v2.2)": len(d_all),
                         "RCH 剔掉": len(d_all) - len(d3),
                         "F 剔掉": len(d3) - len(d),
                         "前 6 名": "、".join(d["name"].head(6))})
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print("  ※ v2.6 = v2.2 VCP 中继平台口径 + R/C/H + F(横盘≤35日)。历史定位仍是高β弹性池")
        print("    （β 0.80→0.91、大亏率 10.3%→13.3%，v2.1 低波池为 0.40 / 4.9%），命中数本身也是仓位信号。")
        print("    F 单独贡献：剔除「横盘≥36日」的长期低波动红利股（长电/核电/唐山港/美的 一类）。")
        print()

    # ---------- 6. 写文件 ----------
    pool2.reset_index().to_csv(os.path.join(OUT_DIR, "候选池_v2.csv"), index=False, encoding="utf-8-sig")
    pool1.reset_index().to_csv(os.path.join(OUT_DIR, "候选池_v1对照.csv"), index=False, encoding="utf-8-sig")
    if len(watch2):
        watch2.reset_index().to_csv(os.path.join(OUT_DIR, "观察池_v2.csv"), index=False, encoding="utf-8-sig")
    if len(story2):
        story2.reset_index().to_csv(os.path.join(OUT_DIR, "故事票_v2.csv"), index=False, encoding="utf-8-sig")
    if len(pcb2):
        pcb2.reset_index().to_csv(os.path.join(OUT_DIR, "坑后平台池_v2.csv"), index=False, encoding="utf-8-sig")
    res_v2["all_scored"].reset_index().to_csv(
        os.path.join(OUT_DIR, "全市场打分明细_v2.csv"), index=False, encoding="utf-8-sig")
    if len(res_v2["removed"]):
        res_v2["removed"].to_csv(os.path.join(OUT_DIR, "持仓剔除_v2.csv"), index=False, encoding="utf-8-sig")

    # 2026-11-10 起不再输出「分周期参数表.csv」（已删除市场周期判断，
    # 参数全周期统一；历史文件保留在 output/v2 中供回溯，不再覆写）。

    print("已输出到:", OUT_DIR)
    print()
    print("※ 数据来源：腾讯前复权日线（全历史长序列）／通达信五大指数日线／东方财富财报与快照。")
    print("※ 以上均为统计回测结论，不构成投资建议。")


if __name__ == "__main__":
    main()
