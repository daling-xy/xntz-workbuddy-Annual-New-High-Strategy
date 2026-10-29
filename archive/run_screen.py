# -*- coding: utf-8 -*-
"""
主入口：跑一次完整选股。
  python run_screen.py            # 输出 CSV + 控制台表格
  python run_screen.py --today    # 额外用当日盘中价做触发检查
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_panel import build_panel          # noqa: E402
from yh_screen import screen, fmt, PARAMS # noqa: E402
from yh_datasource import fetch_holders_many  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

HOLDINGS = ["688120", "300209", "002463", "301205"]

PARAMS_ORDER = ["prox", "width30", "dd_from_high", "amt30", "turn30",
                "rev_yoy", "rev_yoy_prev", "profit_yoy", "roe_w", "pe_ttm",
                "float_mv", "trig_box", "hi250", "pos_in_box", "ret60"]


def main():
    panel = build_panel(asof_offset=1, days=320)
    print(f"全市场有效样本: {len(panel)} 只，结构评估基准日: {panel['date'].max()}")
    print()

    res = screen(panel, holdings=HOLDINGS)
    pool, watch, story, removed = res["pool"], res["watch"], res["story"], res["removed"]

    # --- 基金股东增强（仅对候选池，F10 逐票查询成本高） ---
    cand_codes = list(pool.index[:30]) + list(watch.index[:15]) + list(story.index[:15])
    if cand_codes:
        try:
            holders = fetch_holders_many(cand_codes, workers=10)
            for d in (pool, watch, story, res["all_scored"]):
                d["fund_holders"] = [holders.get(c, {}).get("fund_count", np.nan) for c in d.index]
        except Exception as e:
            print("基金股东查询失败（不影响主流程）:", e)

    # 距离突破触发价的空间（用当日盘中价 vs 30日最高收盘）
    for d in (pool, watch, story):
        d["dist_to_trig"] = d["last_price"] / d["trig_box"] - 1.0

    print(f"==== 核心候选池（全部硬性条件通过，共 {len(pool)} 只）====")
    cols = ["name", "industry", "close", "score", "fund_holders", "dist_to_trig"] + PARAMS_ORDER
    print(fmt(pool, cols=cols).to_string())
    print()

    if len(watch):
        print(f"==== 观察池（结构/资金/行业全过，营收同比 15%~30%，共 {len(watch)} 只）====")
        print(fmt(watch, cols=cols, n=20).to_string())
        print()

    if len(story):
        print(f"==== 高预期故事票（营收>=30%增长但最新一期或TTM亏损，共 {len(story)} 只，建议半仓+更紧止损）====")
        print(fmt(story, cols=cols, n=15).to_string())
        print()

    print("==== 当前池内持仓检查（剔除规则）====")
    if len(removed):
        for _, r in removed.iterrows():
            print(f"  [移除] {r['code']} {r['name']}（{r['industry']}）: {r['reasons']}")
    else:
        print("  持仓均未触发剔除规则（但见下方逐票诊断）。")
    print()

    # 持仓逐票诊断
    print("==== 持仓四票逐项诊断（对照硬性条件）====")
    p = PARAMS
    for c in HOLDINGS:
        if c not in panel.index:
            continue
        r = panel.loc[c]
        checks = [
            ("30日箱宽<=%.0f%%" % (p["width30_max"] * 100), r["width30"] <= p["width30_max"], f"{r['width30']:.1%}"),
            ("近30日高/一年高>=%.2f" % p["prox_min"], r["prox"] >= p["prox_min"], f"{r['prox']:.2f}"),
            ("30日均额>=%.0f亿" % (p["amt30_min"] / 1e8), r["amt30"] >= p["amt30_min"], f"{r['amt30']/1e8:.1f}亿"),
            ("30日均换手 %.1f%%~%.0f%%" % (p["turn30_min"], p["turn30_max"]),
             p["turn30_min"] <= r["turn30"] <= p["turn30_max"], f"{r['turn30']:.2f}%"),
            ("营收同比>=%.0f%%" % p["rev_yoy_min"], r["rev_yoy"] >= p["rev_yoy_min"], f"{r['rev_yoy']:.1f}%"),
            ("上期营收同比>0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"),
            ("行业未剔除", r["board"] not in p["exclude_industries"], str(r["board"])),
        ]
        line = " ".join(("✓" if ok else "✗") + n for n, ok, _v in checks)
        print(f"  {c} {r['name']}")
        print(f"     {line}")
        print(f"     距一年新高 {r['dd_from_high']:.1%} | 箱内位置 {r['pos_in_box']:.0%} | "
              f"近60日 {r['ret60']:.1%} | PE {r['pe_ttm']:.0f} | 亏损={bool(r['is_loss'])}")
        print()

    # 输出 CSV
    out = pool.reset_index()
    out.to_csv(os.path.join(OUT_DIR, "候选池.csv"), index=False, encoding="utf-8-sig")
    if len(story):
        story.reset_index().to_csv(os.path.join(OUT_DIR, "故事票.csv"), index=False, encoding="utf-8-sig")
    res["all_scored"].reset_index().to_csv(
        os.path.join(OUT_DIR, "全市场打分明细.csv"), index=False, encoding="utf-8-sig")
    print("已输出:", OUT_DIR)


if __name__ == "__main__":
    main()
