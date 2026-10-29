# -*- coding: utf-8 -*-
"""
v2 主入口（已含前置数据修复与口径修正）：先判定中级市场周期，再按周期取参数做选股，
并与 v1（全局单一参数、旧口径）对比。

相对上一版 v2 的三处修正（证据见 output/final_calib2.txt、def_check.txt、vol_check.txt、fix_verify.txt）：
  1) prox 口径：改用 prox_now = 当前收盘 / 250日最高收盘。
     旧口径用「近30日最高/年高」，会放行「从年高回落 15% 后横盘」的票
     —— 中信证券、财通证券就是这么混进来的。
  2) 成交额门槛：上一版曾从 4 亿降到 2 亿，修复时变性后证据不稳定，撤回为 4 亿。
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
from yh_screen import (screen, fmt, PARAMS, PARAMS_BY_REGIME,       # noqa: E402
                       REGIME_EVIDENCE, REGIME_SIZE, resolve_regime,
                       params_for, V1_PARAMS)
from yh_datasource import fetch_holders_many                        # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

# 2026-10-29 更新为用户当前实际持仓（雷赛智能/瑞芯微/美的集团/中远海控/春风动力/沪电股份）
HOLDINGS = ["002979", "603893", "000333", "601919", "603129", "002463"]

PARAMS_ORDER = ["prox_now", "prox", "width30", "dd_from_high", "amt30", "turn30",
                "vr20", "rs_pct", "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w", "pe_ttm",
                "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]


def main():
    # ---------- 1. 周期判定 ----------
    st = resolve_regime()
    reg = st["regime"]
    print("=" * 120)
    print(f"中级市场周期判定（评估日 {st['date']}）　数据来源：通达信指数日线 + 全A宽度")
    print("=" * 120)
    print(f"  当前状态： {reg}")
    print(f"  市场宽度： {st['breadth']:.2%}（全A中收盘站上自身 MA60 的股票占比）；"
          f"20 日变化 {st['breadth_slope']:+.2%}")
    print(f"  沪深300 距 250 日高： {st['hs300_dd250']:.2%}")
    print(f"  规则： UP 需宽度≥0.55 且上行；DOWN 需宽度≤0.35 且下行；其余 CHOP（含 5 日因果去抖）")
    print(f"  历史证据： {REGIME_EVIDENCE.get(reg, '')}")
    print(f"  建议参与强度： {REGIME_SIZE.get(reg, '')}")
    print()

    # asof_offset: 1=倒数第2根K线（盘中安全，默认）；0=最新一根（收盘后，bar已定型）。
    # 由 run_daily.py --asof 通过环境变量传入，保证刷新与筛选用同一个基准日。
    _asof = int(os.environ.get("YH_ASOF", "1"))
    panel = build_panel(asof_offset=_asof, days=320)
    print(f"全市场有效样本: {len(panel)} 只，结构评估基准日: {panel['date'].max()}")
    print()

    # ---------- 2. v2（修正版）与 v1（旧口径）并行 ----------
    p2 = params_for(reg)
    res_v2 = screen(panel, params=p2, holdings=HOLDINGS)
    res_v1 = screen(panel, params=dict(V1_PARAMS), holdings=HOLDINGS)
    pool2, watch2, story2 = res_v2["pool"], res_v2["watch"], res_v2["story"]
    pool1, watch1 = res_v1["pool"], res_v1["watch"]

    p1 = dict(V1_PARAMS)

    def gate_line(tag, p):
        pn = p.get("prox_now_min")
        ps = (f"当前价/年高>={pn:.2f}" if pn is not None
              else f"近30日高/年高>={p['prox_min']:.2f}")
        return (f"{tag}: {ps}  箱宽<={p['width30_max']:.0%}  "
                f"额>={p['amt30_min']/1e8:.0f}亿  换手 {p['turn30_min']:.1f}~{p['turn30_max']:.0f}%")

    print("=" * 120)
    print("参数对比")
    print("=" * 120)
    print("  " + gate_line("v1（全局单一，旧口径）", p1))
    print("  " + gate_line(f"v2（{reg}，修正后）  ", p2))
    print()
    print("  ※ 结构性门槛（距年高/箱宽/成交额/换手）三周期统一；")
    print("    营收增速门槛：全周期【关闭】（2026-10-27 起）。区块 bootstrap 复核显示")
    print("    设门槛在 CHOP / DOWN 显著为负，UP 期 CI 含零且样本不足 1 个独立持有期。")
    print("    周期标签同时决定【参与强度】。")
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
    print(f"==== v2 核心候选池（{reg} 周期参数，共 {len(pool2)} 只）====")
    print("=" * 120)
    print(fmt(pool2, cols=cols, n=40).to_string() if len(pool2) else "  （空）")
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
        checks = [
            ("30日箱宽<=%.0f%%" % (p2["width30_max"] * 100), r["width30"] <= p2["width30_max"], f"{r['width30']:.1%}"),
            ("当前价/年高>=%.2f" % p2["prox_now_min"], prox_now >= p2["prox_now_min"], f"{prox_now:.2f}"),
            ("30日均额>=%.0f亿" % (p2["amt30_min"] / 1e8), r["amt30"] >= p2["amt30_min"], f"{r['amt30']/1e8:.1f}亿"),
            ("换手<=%.0f%%" % p2["turn30_max"], r["turn30"] <= p2["turn30_max"], f"{r['turn30']:.2f}%"),
            ("行业未剔除", r["board"] not in p2["exclude_industries"], str(r["board"])),
        ]
        # 营收增速门槛：全周期关闭（见 yh_screen.PARAMS_BY_REGIME 与推导归档）
        _thr = p2.get("rev_yoy_min", None)
        if _thr is None:
            checks.append(("营收同比(本周期不设门槛)", True, f"{r['rev_yoy']:.1f}%"))
        else:
            checks.append(("营收同比>=%.0f%%" % _thr, r["rev_yoy"] >= _thr, f"{r['rev_yoy']:.1f}%"))
            checks.append(("上期营收同比>0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"))
        flag = " ".join(("✓" if ok else "✗") + n for n, ok, _v in checks)
        print(f"  {c} {r['name']}")
        print(f"     {flag}")
        print(f"     距一年新高 {r['dd_from_high']:.1%} | 近30日高/年高 {r['prox']:.2f} | "
              f"箱内位置 {r['pos_in_box']:.0%} | 近60日 {r['ret60']:.1%} | PE {r['pe_ttm']:.0f} | "
              f"亏损={bool(r['is_loss'])}")
    print()

    # ---------- 5.5 严格度阶梯（新口径）----------
    scored = res_v2["all_scored"]
    if "score" in scored.columns:
        ex = set(p2["exclude_industries"])
        # 营收规模用 TTM（最近四个季度）口径；增速门槛按周期分化
        _rcol = "revenue_ttm" if "revenue_ttm" in scored.columns else "revenue"
        _size = pd.to_numeric(scored[_rcol], errors="coerce") >= p2["revenue_min"]
        _thr = p2.get("rev_yoy_min", None)
        _rev = (scored["rev_yoy"].notna() if _thr is None
                else (scored["rev_yoy"].notna() & (scored["rev_yoy"] >= _thr) &
                      (scored["rev_yoy_prev"] > 0)))
        gate = (_rev & _size & scored["board"].notna() &
                (~scored["board"].isin(ex)) & (scored["turn30"] <= p2["turn30_max"]) &
                (scored["amt30"] >= p2["amt30_min"]) &
                (scored["is_loss_ttm"] == 0) if "is_loss_ttm" in scored.columns else True)
        print("==== 严格度阶梯（新口径：当前价/年高；已含 额>=%.0f亿、换手<=%.0f%%）===="
              % (p2["amt30_min"] / 1e8, p2["turn30_max"]))
        # 每档的历史实证（下跌市 + 完整纪律下该档的 beta 调整超额与 90% 置信区间）
        # 目的：明确区分「有证据的推荐档」和「只是放宽出来的无效档」，避免误读。
        try:
            from yh_pos_in_box import ladder_evidence
            EVI = ladder_evidence("DOWN")
        except Exception as _e:
            print("  （历史实证加载失败，不影响主流程：", _e, ")")
            EVI = {}
        ladder = [
            ("v2 推荐   当前价/年高>=0.95 箱宽15%", "now", 0.95, 0.15, "0.95"),
            ("  放宽一档 当前价/年高>=0.90 箱宽15%", "now", 0.90, 0.15, "0.90"),
            ("  放宽两档 当前价/年高>=0.85 箱宽15%", "now", 0.85, 0.15, "0.85"),
            ("  放宽两档 当前价/年高>=0.85 箱宽20%", "now", 0.85, 0.20, "0.85"),
            ("  放宽三档 当前价/年高>=0.80 箱宽25%", "now", 0.80, 0.25, "0.80"),
            ("【旧口径对照】近30日高/年高>=0.95 箱宽15%", "old", 0.95, 0.15, None),
            ("【旧口径对照】近30日高/年高>=0.90 箱宽15%", "old", 0.90, 0.15, None),
        ]
        rows = []
        for tag, kind, thr, w, etag in ladder:
            col = "prox_now" if kind == "now" else "prox"
            m = gate & (scored[col] >= thr) & (scored["width30"] <= w)
            d = scored[m].sort_values("score", ascending=False)
            e = EVI.get(etag) if etag else None
            evs = "—" if not e else f"n={e['n']} β调整{e['est']} CI{e['ci']} {e['verdict']}"
            rows.append({"档位": tag, "入选": len(d), "历史实证(下跌市+纪律)": evs,
                         "前 6 名": "、".join(d["name"].head(6))})
        print(pd.DataFrame(rows).to_string(index=False))
        print()
        print("  ※ 只有「当前价/年高>=0.95」这一档在历史上有统计证据（β调整超额显著为正）。")
        print("    下面那些放宽档是为了让你看清「再松一点会放进谁」，它们的历史超额置信区间都含零，")
        print("    即【没有证据支持】，不是推荐。命中数本身也是仓位信号。")
        print()

    # ---------- 6. 写文件 ----------
    pool2.reset_index().to_csv(os.path.join(OUT_DIR, "候选池_v2.csv"), index=False, encoding="utf-8-sig")
    pool1.reset_index().to_csv(os.path.join(OUT_DIR, "候选池_v1对照.csv"), index=False, encoding="utf-8-sig")
    if len(watch2):
        watch2.reset_index().to_csv(os.path.join(OUT_DIR, "观察池_v2.csv"), index=False, encoding="utf-8-sig")
    if len(story2):
        story2.reset_index().to_csv(os.path.join(OUT_DIR, "故事票_v2.csv"), index=False, encoding="utf-8-sig")
    res_v2["all_scored"].reset_index().to_csv(
        os.path.join(OUT_DIR, "全市场打分明细_v2.csv"), index=False, encoding="utf-8-sig")
    if len(res_v2["removed"]):
        res_v2["removed"].to_csv(os.path.join(OUT_DIR, "持仓剔除_v2.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"周期": r,
         "当前价/年高>=": v.get("prox_now_min"), "箱宽<=": v["width30_max"],
         "额>=亿": v["amt30_min"] / 1e8,
         "换手区间": f"{v['turn30_min']:.1f}~{v['turn30_max']:.0f}%",
         "建议参与强度": REGIME_SIZE.get(r, ""),
         "历史事件证据": REGIME_EVIDENCE.get(r, "")}
        for r, v in PARAMS_BY_REGIME.items()
    ]).to_csv(os.path.join(OUT_DIR, "分周期参数表.csv"), index=False, encoding="utf-8-sig")

    print("已输出到:", OUT_DIR)
    print()
    print("※ 数据来源：腾讯前复权日线（全历史长序列）／通达信五大指数日线／东方财富财报与快照。")
    print("※ 以上均为统计回测结论，不构成投资建议。")


if __name__ == "__main__":
    main()
