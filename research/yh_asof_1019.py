# -*- coding: utf-8 -*-
"""
【定点复盘】把评估日定在 2026-10-19，跑一次完整选股，看那天乐鑫科技 688018 在不在池子里。

做法：build_panel(asof_offset=N) 用的是「倒数第 N+1 根 K 线」。
K 线最后几根为 ... 10-19, 10-20, 10-21, 10-22, 10-23，
所以 asof_offset=4 对应 10-19。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_panel import build_panel                                   # noqa: E402
from yh_screen import screen, fmt, params_for, resolve_regime      # noqa: E402

CODE = "688018"
HOLDINGS = ["688120", "300209", "002463", "301205"]


def main():
    # ---------- 1. 那天的市场周期 ----------
    import pickle
    st_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "regime_state.pkl")
    with open(st_path, "rb") as f:
        stt = pickle.load(f)
    stt = stt.sort_values("date")
    row = stt[stt["date"] <= "2026-10-19"].iloc[-1]
    reg = str(row["regime"])
    print("=" * 116)
    print("定点复盘：评估日 2026-10-19")
    print("=" * 116)
    print(f"  当日周期标签：{reg}　（市场宽度 {row['breadth']:.2%}）")
    p = params_for(reg)
    print(f"  采用参数：当前价/年高>={p['prox_now_min']:.2f}　箱宽<={p['width30_max']:.0%}　"
          f"额>={p['amt30_min']/1e8:.0f}亿　换手 {p['turn30_min']:.1f}~{p['turn30_max']:.0f}%")
    print()

    # ---------- 2. 以 10-19 为评估日重建面板 ----------
    panel = build_panel(asof_offset=4, days=320, force=True)
    print(f"  面板基准日：{panel['date'].max()}（应为 2026-10-19）　样本 {len(panel)} 只")
    print()

    # ---------- 3. 乐鑫逐项对照 ----------
    print("=" * 116)
    print(f"A. 乐鑫科技 {CODE} 在 2026-10-19 的逐项对照")
    print("=" * 116)
    if CODE in panel.index:
        r = panel.loc[CODE]
        pn = float(r["prox_now"]) if "prox_now" in panel.columns else 1.0 + float(r["dd_from_high"])
        checks = [
            ("当前价/年高 >= 0.95", pn >= 0.95, f"{pn:.3f}"),
            ("30日箱宽 <= 15%", r["width30"] <= 0.15, f"{r['width30']:.1%}"),
            ("30日均额 >= 4亿", r["amt30"] >= 4e8, f"{r['amt30']/1e8:.1f}亿"),
            ("换手 <= 6%", r["turn30"] <= 6.0, f"{r['turn30']:.2f}%"),
            ("营收同比 >= 30%", r["rev_yoy"] >= 30.0, f"{r['rev_yoy']:.1f}%"),
            ("上期营收同比 > 0", r["rev_yoy_prev"] > 0, f"{r['rev_yoy_prev']:.1f}%"),
            ("单期营收 >= 2亿", r["revenue"] >= 2e8, f"{r['revenue']/1e8:.1f}亿"),
        ]
        for name, ok, val in checks:
            print(f"  {'✓' if ok else '✗'} {name:<22} 实测 {val}")
        print(f"\n  其它：收盘 {r['close']:.2f}　年高 {r['hi250']:.2f}　"
              f"距年高 {r['dd_from_high']:+.1%}　箱内位置 {r['pos_in_box']:.0%}　"
              f"近20日 {r['ret20']:+.1%}　近60日 {r['ret60']:+.1%}")
        bad = [n for n, ok, _ in checks if not ok]
        print(f"\n  → 未通过：{'、'.join(bad)}")
    else:
        print("  该票不在面板里（可能停牌或数据缺失）")
    print()

    # ---------- 4. 那天真正选出谁 ----------
    res = screen(panel, params=p, holdings=HOLDINGS)
    pool = res["pool"]
    print("=" * 116)
    print(f"B. 那天（2026-10-19）v2 核心候选池：共 {len(pool)} 只")
    print("=" * 116)
    if len(pool):
        cols = ["name", "board", "close", "score", "prox_now", "width30", "amt30",
                "turn30", "rev_yoy", "rev_yoy_prev", "pos_in_box"]
        cols = [c for c in cols if c in pool.columns]
        print(fmt(pool, cols=cols, n=40).to_string() if len(pool) else "  （空）")
    else:
        print("  （空）")
    print()

    # ---------- 5. 乐鑫在全市场的位置 ----------
    print("=" * 116)
    print("C. 乐鑫在全市场里排第几（按策略综合分）")
    print("=" * 116)
    sc = res["all_scored"]
    if "score" in sc.columns and CODE in sc.index:
        rank = int((sc["score"] > sc.at[CODE, "score"]).sum()) + 1
        print(f"  乐鑫综合分 {sc.at[CODE,'score']:.1f}　全市场排名 {rank}/{len(sc)}")
        print(f"  （但排名高不代表入选 —— 硬门槛没过就是没过）")
    print()
    print("※ 数据来源：腾讯前复权日线／通达信指数日线／东方财富财报。历史统计，不构成投资建议。")


if __name__ == "__main__":
    main()
