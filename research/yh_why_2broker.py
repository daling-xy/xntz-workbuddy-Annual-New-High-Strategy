# -*- coding: utf-8 -*-
"""
专项核验：中信证券(600030)、财通证券(601108) 为什么会被选出来，修正口径后为什么被排除。

对比两条口径：
  旧口径 prox      = 近30日最高收盘 / 250日最高收盘   （允许「先冲高后回落横盘」）
  新口径 prox_now  = 当前收盘       / 250日最高收盘   （= 1 + dd_from_high）
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_panel import build_panel  # noqa: E402

TARGETS = ["600030", "601108", "600837", "000776", "002736", "601377"]  # 券商样本


def main():
    panel = build_panel(asof_offset=1, days=320)
    panel["prox_now"] = 1.0 + panel["dd_from_high"]
    print("评估基准日:", panel["date"].max(), " 样本:", len(panel))
    print()

    cols = ["name", "board", "close", "prox", "prox_now", "dd_from_high",
            "width30", "amt30", "turn30", "ret60", "rev_yoy"]
    sub = panel[panel.index.isin(TARGETS)][cols].copy()
    sub = sub.assign(_o=lambda d: 1 - d["prox"])
    sub["新口径差(1-prox_now)"] = 1 - sub["prox_now"]
    print("=== 券商样本：双口径对照 ===")
    print(sub.to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== 判定 ===")
    for c, r in sub.iterrows():
        old_pass = r["prox"] >= 0.90
        new_pass = r["prox_now"] >= 0.95
        width_ok = r["width30"] <= 0.15
        amt_ok = r["amt30"] >= 4e8
        amt_ok2 = r["amt30"] >= 2e8
        turn_ok = r["turn30"] <= 6.0
        rev_ok = r["rev_yoy"] >= 30
        print(f"{c} {r['name']}:")
        print(f"   旧口径(近30日高/年高)={r['prox']:.3f} -> {'过' if old_pass else '不过'} 0.90；"
              f"新口径(当前价/年高)={r['prox_now']:.3f} -> {'过' if new_pass else '不过'} 0.95")
        print(f"   箱宽 {r['width30']:.1%} {'✓' if width_ok else '✗'}15% | "
              f"额 {r['amt30']/1e8:.1f}亿 {'✓' if amt_ok else ('✓(2亿档)' if amt_ok2 else '✗')} | "
              f"换手 {r['turn30']:.2f}% {'✓' if turn_ok else '✗'}6% | "
              f"营收同比 {r['rev_yoy']:.1f}% {'✓' if rev_ok else '✗'}30%")
        if old_pass and not new_pass:
            print(f"   → 【正是旧口径的漏洞】：30日内曾冲到年高的 {r['prox']:.0%}，"
                  f"但当前已回落到年高的 {r['prox_now']:.0%}（回撤 {r['dd_from_high']:.1%}），"
                  f"属于「冲高回落横盘」，不是「站在高位」。")
        print()


if __name__ == "__main__":
    main()
