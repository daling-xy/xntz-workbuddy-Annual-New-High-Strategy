# -*- coding: utf-8 -*-
"""
v2 主入口：先判定中级市场周期，再按周期取参数做选股，并与 v1（全局单一参数）对比。

  python run_screen_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_panel import build_panel                                   # noqa: E402
from yh_screen import (screen, fmt, PARAMS, PARAMS_BY_REGIME,      # noqa: E402
                       REGIME_EVIDENCE, resolve_regime, params_for)
from yh_datasource import fetch_holders_many                       # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

HOLDINGS = ["688120", "300209", "002463", "301205"]

PARAMS_ORDER = ["prox", "width30", "dd_from_high", "amt30", "turn30",
                "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w", "pe_ttm",
                "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]


def main():
    # ---------- 1. 周期判定 ----------
    st = resolve_regime()
    reg = st["regime"]
    print("=" * 118)
    print(f"中级市场周期判定（评估日 {st['date']}）")
    print("=" * 118)
    print(f"  当前状态： {reg}")
    print(f"  市场宽度： {st['breadth']:.2%}（全A中收盘站上自身 MA60 的股票占比）；"
          f"20 日变化 {st['breadth_slope']:+.2%}")
    print(f"  沪深300 距 250 日高： {st['hs300_dd250']:.2%}（数据来源：通达信指数日线）")
    print(f"  阈值规则： UP 需宽度≥0.55 且上行；DOWN 需宽度≤0.35 且下行；其余为 CHOP（含 5 日去抖）")
    print(f"  该参数集的历史证据： {REGIME_EVIDENCE.get(reg, '')}")
    print()

    panel = build_panel(asof_offset=1, days=320)
    print(f"全市场有效样本: {len(panel)} 只，结构评估基准日: {panel['date'].max()}")
    print()

    # ---------- 2. v2 与 v1 并行跑 ----------
    res_v2 = screen(panel, params=params_for(reg), holdings=HOLDINGS)
    res_v1 = screen(panel, params=dict(PARAMS), holdings=HOLDINGS)
    pool2, watch2, story2 = res_v2["pool"], res_v2["watch"], res_v2["story"]
    pool1, watch1 = res_v1["pool"], res_v1["watch"]

    p1, p2 = dict(PARAMS), params_for(reg)

    def gate_line(tag, p):
        return (f"{tag}: prox>={p['prox_min']:.2f}  箱宽<={p['width30_max']:.0%}  "
                f"额>={p['amt30_min']/1e8:.0f}亿  换手 {p['turn30_min']:.1f}~{p['turn30_max']:.0f}%")

    print("=" * 118)
    print("参数对比")
    print("=" * 118)
    print("  " + gate_line("v1（全局单一）", p1))
    print("  " + gate_line(f"v2（{reg}）   ", p2))
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

    for d in (pool2, watch2, story2):
        d["dist_to_trig"] = d["last_price"] / d["trig_box"] - 1.0
    for d in (pool1, watch1):
        d["dist_to_trig"] = d["last_price"] / d["trig_box"] - 1.0

    # ---------- 4. 输出 ----------
    cols = ["name", "industry", "close", "score", "fund_holders", "dist_to_trig"] + PARAMS_ORDER
    print("=" * 118)
    print(f"==== v2 核心候选池（{reg} 周期参数，共 {len(pool2)} 只）====")
    print("=" * 118)
    print(fmt(pool2, cols=cols, n=40).to_string() if len(pool2) else "  （空）")
    print()

    print(f"==== v1 核心候选池（共 {len(pool1)} 只，用于对照）====")
    print(fmt(pool1, cols=["name", "industry", "close", "score", "prox", "width30",
                           "amt30", "turn30", "rev_yoy"], n=40).to_string() if len(pool1) else "  （空）")
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
    print("==== 持仓四票：v2 剔除规则检查 ====")
    if len(res_v2["removed"]):
        for _, r in res_v2["removed"].iterrows():
            print(f"  [移除] {r['code']} {r['name']}（{r['industry']}）: {r['reasons']}")
    else:
        print("  持仓均未触发剔除规则（详见下方逐票诊断）。")
    print()

    print("==== 持仓四票逐项诊断（对照 v2 门槛）====")
    for c in HOLDINGS:
        if c not in panel.index:
            continue
        r = panel.loc[c]
        checks = [
            ("30日箱宽<=%.0f%%" % (p2["width30_max"] * 100), r["width30"] <= p2["width30_max"], f"{r['width30']:.1%}"),
            ("近30日高/一年高>=%.2f" % p2["prox_min"], r["prox"] >= p2["prox_min"], f"{r['prox']:.2f}"),
            ("30日均额>=%.0f亿" % (p2["amt30_min"] / 1e8), r["amt30"] >= p2["amt30_min"], f"{r['amt30']/1e8:.1f}亿"),
            ("换手<=%.0f%%" % p2["turn30_max"], r["turn30"] <= p2["turn30_max"], f"{r['turn30']:.2f}%"),
            ("营收同比>=%.0f%%" % p2["rev_yoy_min"], r["rev_yoy"] >= p2["rev_yoy_min"], f"{r['rev_yoy']:.1f}%"),
            ("上期营收同比>0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"),
            ("行业未剔除", r["board"] not in p2["exclude_industries"], str(r["board"])),
        ]
        flag = " ".join(("✓" if ok else "✗") + n for n, ok, _v in checks)
        print(f"  {c} {r['name']}")
        print(f"     {flag}")
        print(f"     距一年新高 {r['dd_from_high']:.1%} | 箱内位置 {r['pos_in_box']:.0%} | "
              f"近60日 {r['ret60']:.1%} | PE {r['pe_ttm']:.0f} | 亏损={bool(r['is_loss'])}")
    print()

    # ---------- 5.5 严格度阶梯：让用户按自己的「资金流活跃」底线挑档位 ----------
    scored = res_v2["all_scored"]
    if "score" in scored.columns:
        ex = set(p2["exclude_industries"])
        gate = ((scored["rev_yoy"] >= p2["rev_yoy_min"]) & (scored["rev_yoy_prev"] > 0) &
                (scored["revenue"] >= p2["revenue_min"]) & scored["board"].notna() &
                (~scored["board"].isin(ex)) & (scored["turn30"] <= p2["turn30_max"]) &
                (scored["is_loss_ttm"] == 0) if "is_loss_ttm" in scored.columns else True)
        print("==== 严格度阶梯（同上表统一已加 turn30<=%.0f%% 天花板；按目前的 %s 市场挑档）===="
              % (p2["turn30_max"], reg))
        ladder = [
            ("v1 原版（箱宽25%/额4亿）", 0.25, 4e8, 1.5, 0.90),
            (f"v2-{reg} 箱宽15%", 0.15, 2e8, 0.0, p2["prox_min"]),
            ("   同上 + 换手>=1.0%", 0.15, 2e8, 1.0, p2["prox_min"]),
            ("   同上 + 换手>=1.5%", 0.15, 2e8, 1.5, p2["prox_min"]),
            ("折中 箱宽18%", 0.18, 2e8, 0.0, p2["prox_min"]),
            ("折中 箱宽18% + 换手>=1.0%", 0.18, 2e8, 1.0, p2["prox_min"]),
            ("折中 箱宽20%", 0.20, 2e8, 0.0, p2["prox_min"]),
            ("折中 箱宽20% + 换手>=1.5%", 0.20, 2e8, 1.5, p2["prox_min"]),
        ]
        rows = []
        for tag, w, a, t0, pr in ladder:
            m = (gate & (scored["prox"] >= pr) & (scored["width30"] <= w) &
                 (scored["amt30"] >= a) & (scored["turn30"] >= t0))
            d = scored[m].sort_values("score", ascending=False)
            rows.append({"档位": tag, "入选": len(d),
                         "前 6 名（按综合分）": "、".join(d["name"].head(6))})
        print(pd.DataFrame(rows).to_string(index=False))
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

    pd.DataFrame([
        {"周期": r, "prox_min": v["prox_min"], "width30_max": v["width30_max"],
         "amt30_min(亿)": v["amt30_min"] / 1e8, "turn30_min": v["turn30_min"],
         "turn30_max": v["turn30_max"], "历史事件证据": REGIME_EVIDENCE.get(r, "")}
        for r, v in PARAMS_BY_REGIME.items()
    ]).to_csv(os.path.join(OUT_DIR, "分周期参数表.csv"), index=False, encoding="utf-8-sig")

    print("已输出到:", OUT_DIR)
    print()
    print("※ 数据来源：腾讯前复权日线（全历史长序列）／通达信五大指数日线／东方财富财报与快照。")
    print("※ 以上均为统计回测结论，不构成投资建议。")


if __name__ == "__main__":
    main()
