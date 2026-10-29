# -*- coding: utf-8 -*-
"""
Regime 标签：把「市场宽度」与「指数趋势」合成中级周期状态，并回看关键年份。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_regime_state import (CACHE, assign_state, build_breadth, index_panel)  # noqa: E402


def build(force: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE, "regime_state.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            import pickle
            return pickle.load(f)

    br = build_breadth()
    br = br[br["nstock"] >= 2000].reset_index(drop=True)   # 早期样本不足的交易日剔除
    ipx = index_panel()
    st = assign_state(br, ipx)
    st.to_pickle(path)
    return st


if __name__ == "__main__":
    st = build()
    print("样本区间:", st["date"].min(), "~", st["date"].max(), " 交易日数:", len(st))
    print()
    st["year"] = st["date"].str[:4]
    tab = st.groupby(["year", "regime"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=["UP", "CHOP", "DOWN"], fill_value=0)
    tab["天数"] = tab.sum(axis=1)
    tab["UP%"] = (tab["UP"] / tab["天数"] * 100).round(1)
    tab["DOWN%"] = (tab["DOWN"] / tab["天数"] * 100).round(1)
    print("=== 各年度 Regime 分布 ===")
    print(tab.to_string())
    print()

    pd.set_option("display.width", 200)
    print("=== 近 40 个交易日 Regime 明细 ===")
    cols = ["date", "breadth", "bull_stack"]
    cols += [c for c in st.columns if c.startswith("沪深300_")]
    print(st[cols + ["regime"]].tail(40).to_string(index=False))
    print()
    print("=== 当前状态 ===")
    last = st.iloc[-1]
    print(f"日期 {last['date']}  状态 {last['regime']}")
    print(f"  宽度(站上MA60占比) {last['breadth']:.3f}  多头排列占比 {last['bull_stack']:.3f}")
    for c in st.columns:
        if c.startswith("沪深300_"):
            print(f"  {c}: {last[c]:.4f}")
