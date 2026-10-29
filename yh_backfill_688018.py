# -*- coding: utf-8 -*-
"""
【反事实验证】如果把时间倒回去，乐鑫科技 688018 有没有哪天能被 v2 选出来？

先澄清一个误解：它不是「因为已经突破」才没被选出来 —— 突破（当前价/年高越高）恰恰是
策略要的。真正卡住的是【30日箱宽 42%】和【营收同比 18.9%】。
本脚本逐日回算过去 60 个交易日的技术门槛，看它在启动前是不是曾经满足过。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import CACHE_DIR, fetch_klines          # noqa: E402
from yh_factors import features_at                          # noqa: E402
from yh_panel import build_panel, DEFAULT_PARAMS            # noqa: E402

CODE = "688018"
DAYS = 60


def main():
    kmap = fetch_klines([CODE], days=320, workers=4, cache_tag="320")
    bars = kmap.get(CODE)
    if not bars:
        print("拿不到 K 线")
        return
    p = dict(DEFAULT_PARAMS)

    panel = build_panel(asof_offset=1, days=320)
    fin = panel.loc[CODE] if CODE in panel.index else None

    n = len(bars)
    start = max(250, n - DAYS)
    rows = []
    for idx in range(start, n):
        f = features_at(bars, p, idx=idx)
        if not f:
            continue
        b = bars[idx]
        prox_now = 1.0 + f["dd_from_high"]
        ok_prox = prox_now >= 0.95
        ok_w = f["width30"] <= 0.15
        ok_amt = f["amt30"] >= 4e8
        ok_turn = f["turn30"] <= 6.0
        ok_tech = ok_prox and ok_w and ok_amt and ok_turn
        rows.append({
            "date": b["date"], "close": b["close"],
            "prox_now": round(prox_now, 3), "箱宽%": round(f["width30"] * 100, 1),
            "额(亿)": round(f["amt30"] / 1e8, 1), "换手%": round(f["turn30"], 2),
            "箱位%": round(f["pos_in_box"] * 100, 0),
            "Prox✓": "✓" if ok_prox else "✗", "箱宽✓": "✓" if ok_w else "✗",
            "额✓": "✓" if ok_amt else "✗", "换手✓": "✓" if ok_turn else "✗",
            "技术全过": "★" if ok_tech else "",
        })

    df = pd.DataFrame(rows)
    print("=" * 118)
    print(f"A. 乐鑫科技 {CODE} 逐日回算（近 {len(df)} 个交易日）　技术门槛：年高>=0.95 / 箱宽<=15% / 额>=4亿 / 换手<=6%")
    print("=" * 118)
    print(df.to_string(index=False))
    print()

    hit = df[df["技术全过"] == "★"]
    print("=" * 118)
    print("B. 结论")
    print("=" * 118)
    if len(hit):
        print(f"  有 {len(hit)} 天技术门槛全过：{', '.join(hit['date'].tolist())}")
    else:
        print("  过去这段时间里【没有任何一天】技术门槛全过。")
    print()
    # 分段统计，看卡在哪
    print(f"  单独看每一条：")
    print(f"    当前价/年高 >=0.95 满足 {int((df['Prox✓']=='✓').sum())}/{len(df)} 天")
    print(f"    箱宽 <=15%         满足 {int((df['箱宽✓']=='✓').sum())}/{len(df)} 天")
    print(f"    额 >=4亿           满足 {int((df['额✓']=='✓').sum())}/{len(df)} 天")
    print(f"    换手 <=6%          满足 {int((df['换手✓']=='✓').sum())}/{len(df)} 天")
    print()
    if fin is not None:
        print(f"  基本面（财报 2026-06-30，期内不变）：营收同比 {fin['rev_yoy']:.1f}% "
              f"（门槛 >=30% {'✓' if fin['rev_yoy']>=30 else '✗'}）　"
              f"上期 {fin['rev_yoy_prev']:.1f}%　单期营收 {fin['revenue']/1e8:.1f}亿")
    print()
    print("=" * 118)
    print("C. 关键澄清：不是「因为突破」才没选出来")
    print("=" * 118)
    print("  策略的三条技术门槛里，【贴年高】恰恰欢迎突破 —— prox_now 越高越好（>=0.95 即可，")
    print("  创新高 = 1.0 也照样合格）。真正把它挡住的是【箱宽】：")
    print("  涨得越快，30日振幅被撑得越大，箱宽就越容易超过 15%。")
    print("  也就是说：它是被【自己涨得太猛】挡住的，不是被【突破】挡住的。")
    print()
    print("※ 数据来源：腾讯前复权日线／通达信指数日线／东方财富财报。历史统计，不构成投资建议。")


if __name__ == "__main__":
    main()
