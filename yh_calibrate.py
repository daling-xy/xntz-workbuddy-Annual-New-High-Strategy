# -*- coding: utf-8 -*-
"""
参数校准：用事件回放表检验每个条件的真实效果。
全部使用「相对当日全市场横截面均值」的超额收益，剔除牛熊行情本身的影响。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_eventstudy import build_event_table  # noqa: E402


def stat(df: pd.DataFrame, label: str, key: str = "ex_fwd60") -> dict:
    """返回一组超额收益统计"""
    if len(df) == 0:
        return {"label": label, "n": 0}
    r = df[key]
    return {
        "label": label,
        "n": int(len(df)),
        "均值": r.mean(),
        "中位": r.median(),
        "胜率": (r > 0).mean(),
        "p90": r.quantile(0.9),
        "p10": r.quantile(0.1),
        "mae中位": df["mae20"].median(),
        "mfe中位": df["mfe60"].median(),
    }


def show(rows, title=""):
    df = pd.DataFrame([r for r in rows if r.get("n", 0) > 0])
    if not len(df):
        print(title, "无样本"); return
    print(f"--- {title} （超额 60 日收益）---")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print()


def main():
    ev = build_event_table()
    print("事件总数:", len(ev), " 时间:", ev["date"].min(), "~", ev["date"].max())
    print()

    # 全市场基准
    show([stat(ev, "全市场基准")], "基准")

    # ---------- 条件 2：prox（近一年高点接近度）----------
    rows = []
    for v in [0.70, 0.80, 0.90, 0.95, 0.98, 1.0]:
        rows.append(stat(ev[ev["prox"] >= v], f"prox>={v:.2f}"))
    show(rows, "条件2：30日高/一年高（prox）")

    # ---------- 条件 1：箱体宽度 ----------
    base = ev[ev["prox"] >= 0.90]
    rows = []
    for v in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 9.9]:
        rows.append(stat(base[base["width30"] <= v], f"prox>=.90 且 宽30<={v:.0%}"))
    show(rows, "条件1：30日收盘价箱体宽度（先固定 prox>=0.90）")

    # ---------- 成交额门槛 ----------
    base2 = ev[(ev["prox"] >= 0.90) & (ev["width30"] <= 0.20)]
    rows = []
    for v in [0.5e8, 1e8, 2e8, 3e8, 4e8, 6e8]:
        rows.append(stat(base2[base2["amt30"] >= v], f"日均额>={v/1e8:.0f}亿"))
    show(rows, "条件5：30日日均成交额（固定 prox>=.90, 宽<=20%）")

    # ---------- 换手率：不是越高越好？ ----------
    base3 = ev[(ev["prox"] >= 0.90) & (ev["width30"] <= 0.20) & (ev["amt30"] >= 2e8)]
    rows = []
    q = base3["turn30"].quantile([.2, .4, .6, .8]).values
    bins = [0, q[0], q[1], q[2], q[3], np.inf]
    names = ["Q1低换手", "Q2", "Q3", "Q4", "Q5高换手"]
    for lo, hi, nm in zip(bins[:-1], bins[1:], names):
        rows.append(stat(base3[(base3["turn30"] > lo) & (base3["turn30"] <= hi)],
                         f"{nm}({lo:.1f}~{hi:.1f}%)"))
    show(rows, "换手率五分位（固定基础条件）")

    # ---------- 收缩度 contraction（10日宽/30日宽）----------
    rows = []
    for v in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        rows.append(stat(base3[base3["contraction"] <= v], f"收缩度<={v:.1f}"))
    show(rows, "末端收缩度（近10日宽/近30日宽）")

    # ---------- 自身历史宽度分位 width_pct ----------
    rows = []
    for v in [0.2, 0.3, 0.4, 0.5, 0.6]:
        rows.append(stat(base3[base3["width_pct"] <= v], f"自身宽度分位<={v:.1f}"))
    show(rows, "箱宽在自身过去一年的分位（自适应紧度）")

    # ---------- 量能收缩 vol_dry ----------
    rows = []
    for v in [0.6, 0.7, 0.8, 0.9, 1.1]:
        rows.append(stat(base3[base3["vol_dry"] <= v], f"近10日额/近30日额<={v:.1f}"))
    show(rows, "整理期缩量程度")

    # ---------- MAE/MFE：止损应设多紧 ----------
    b = base3
    print("--- 止损参考（基础条件命中事件，持有20日的盘中最大不利幅度 mae20）---")
    print("mae20 分位:",
          {q: round(float(b["mae20"].quantile(q / 100)), 3) for q in [5, 10, 25, 50, 75, 90]})
    print("mfe60 分位:",
          {q: round(float(b["mfe60"].quantile(q / 100)), 3) for q in [10, 25, 50, 75, 90]})
    print()

    # ---------- 组合对比：原版 vs 改良 ----------
    orig = ev[(ev["prox"] >= 0.90) & (ev["width30"] <= 0.20) & (ev["amt30"] >= 4e8)]
    v25 = ev[(ev["prox"] >= 0.90) & (ev["width30"] <= 0.25) & (ev["amt30"] >= 4e8)]
    imp1 = ev[(ev["prox"] >= 0.95) & (ev["width30"] <= 0.20) & (ev["amt30"] >= 2e8)]
    imp2 = ev[(ev["prox"] >= 0.95) & (ev["width30"] <= 0.20) & (ev["amt30"] >= 2e8)
              & (ev["contraction"] <= 0.6)]
    imp3 = ev[(ev["prox"] >= 0.95) & (ev["width30"] <= 0.20) & (ev["amt30"] >= 2e8)
              & (ev["contraction"] <= 0.6) & (ev["width_pct"] <= 0.5)]
    show([stat(ev, "全市场"),
          stat(orig, "原版(prox.90+宽20%+4亿)"),
          stat(v25, "原版25%箱宽试验"),
          stat(imp1, "改良A(prox.95+宽20%+2亿)"),
          stat(imp2, "改良B=A+收缩<=0.6"),
          stat(imp3, "改良C=B+自身分位<=0.5")], "组合对比")


if __name__ == "__main__":
    main()
