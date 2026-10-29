# -*- coding: utf-8 -*-
"""
周期分层校准 III：把「放宽 prox」拆成可执行的两件事
  1) 下限放宽（不再要求 ≥0.90，允许 0.80 起的「中距年高带」）
  2) 上限收紧（下跌市显式排除 prox≥0.95 的追新高/创新高样本）
再叠加本轮发现的另外两条下跌市修正：限制换手过热、剔除近 60 日已经走了一大截的票。
输出 final decision table + 子样本稳定性检验。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 1200
_rng = np.random.default_rng(20261022)


def boot_ci(sub, key, n=BOOT):
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = v.mean()
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def row(label, sub):
    if sub is None or len(sub) == 0:
        return None
    e, lo, hi = boot_ci(sub, "ex_fwd60")
    ae, alo, ahi = boot_ci(sub, "ab_fwd60")
    return {
        "分组": label,
        "事件数": int(len(sub)),
        "交易日": int(sub["date"].nunique()),
        "日均票": round(len(sub) / max(sub["date"].nunique(), 1), 1),
        "去均值超额%": round(e * 100, 2),
        "90%CI": f"[{lo*100:.1f},{hi*100:.1f}]",
        "beta调整超额%": round(ae * 100, 2),
        "beta调整CI": f"[{alo*100:.1f},{ahi*100:.1f}]",
        "绝对收益%": round(sub["fwd60"].mean() * 100, 2),
        "胜率%": round((sub["fwd60"] > 0).mean() * 100, 1),
        "大亏<=-20%": round((sub["fwd60"] <= -0.20).mean() * 100, 1),
        "大赢>=+20%": round((sub["fwd60"] >= 0.20).mean() * 100, 1),
        "mae20中位%": round(sub["mae20"].median() * 100, 1),
    }


COLS = ["分组", "事件数", "交易日", "日均票", "去均值超额%", "90%CI",
        "beta调整超额%", "beta调整CI", "绝对收益%", "胜率%",
        "大亏<=-20%", "大赢>=+20%", "mae20中位%"]


def show(rows, title):
    rs = [r for r in rows if r]
    if not rs:
        print(f"--- {title} --- 无样本\n")
        return
    print(f"--- {title} ---")
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(pd.DataFrame(rs)[COLS].to_string(index=False))
    print()


def load():
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["year"] = ev["date"].astype(str).str[:4]
    return ev


def combos(s, T=3.0, P=0.40, AMT=2e8, W=0.25):
    """返回 {名称: mask}。所有组合统一带流动性地板 amt30>=AMT。"""
    base_liq = s["amt30"] >= AMT
    p = s["prox"]
    p40 = float(s["ret60"].quantile(P))
    calm = s["turn30"] <= T
    notrun = s["ret60"] <= p40
    return {
        "①下限放宽 prox>=0.85": (p >= 0.85) & (s["width30"] <= W) & base_liq,
        "②下限放宽 prox>=0.80": (p >= 0.80) & (s["width30"] <= W) & base_liq,
        "③中距带 0.80~0.95": (p >= 0.80) & (p < 0.95) & (s["width30"] <= W) & base_liq,
        "④中距带+去追高": (p >= 0.80) & (p < 0.95) & (s["width30"] <= W) & base_liq & notrun,
        "⑤中距带+控换手": (p >= 0.80) & (p < 0.95) & (s["width30"] <= W) & base_liq & calm,
        "⑥中距带+去追高+控换手": (p >= 0.80) & (p < 0.95) & (s["width30"] <= W) & base_liq & notrun & calm,
        "⑦宽版 0.70~0.95 去追高+控换手": (p >= 0.70) & (p < 0.95) & (s["width30"] <= W) & base_liq & notrun & calm,
        "⑧中距带+严箱体15%": (p >= 0.80) & (p < 0.95) & (s["width30"] <= 0.15) & base_liq & notrun & calm,
        "⑨中距带+缩量amt_act<=1.0": (p >= 0.80) & (p < 0.95) & (s["width30"] <= W) & base_liq
        & notrun & calm & (s["amt_act"] <= 1.0),
    }


def main(ev):
    print("=" * 122)
    print("G. 「放宽」的正确姿势：下限放宽 + 上限收紧 ——三周期同表对照")
    print("=" * 122)
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        cs = combos(s)
        rows = [row(f"[{r}] {k}", s[m]) for k, m in cs.items()]
        rows.insert(0, row(f"[{r}] ⓪v1原版 prox>=0.90 箱宽<=25% 额>=4亿",
                           s[(s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8)]))
        show(rows, f"[{r}] 候选组合")

    print("=" * 122)
    print("H. 交互检验：同一套参数在三个周期里是不是真的「表现不同」？（验证周期区分是否必要）")
    print("=" * 122)
    keys = ["⓪v1原版 prox>=0.90 箱宽<=25% 额>=4亿", "③中距带 0.80~0.95", "⑥中距带+去追高+控换手"]
    tab = []
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        cs = combos(s)
        cs["⓪v1原版 prox>=0.90 箱宽<=25% 额>=4亿"] = (
            (s["prox"] >= 0.90) & (s["width30"] <= 0.25) & (s["amt30"] >= 4e8))
        for k in keys:
            sub = s[cs[k]]
            if len(sub) == 0:
                continue
            ae, alo, ahi = boot_ci(sub, "ab_fwd60")
            e, lo, hi = boot_ci(sub, "ex_fwd60")
            tab.append({"周期": r, "参数": k, "事件数": len(sub),
                        "去均值超额%": round(e * 100, 2),
                        "beta调整超额%": round(ae * 100, 2),
                        "beta调整CI": f"[{alo*100:.1f},{ahi*100:.1f}]"})
    print(pd.DataFrame(tab).pivot(index="参数", columns="周期",
                                  values="去均值超额%").to_string())
    print()
    print(pd.DataFrame(tab).pivot(index="参数", columns="周期",
                                  values="beta调整超额%").to_string())
    print()

    print("=" * 122)
    print("I. 子样本稳定性：把历史切成两半，看结论是不是只由某一段撑起来")
    print("=" * 122)
    for r in ["DOWN", "UP"]:
        for k in ["③中距带 0.80~0.95", "⑥中距带+去追高+控换手"]:
            rows = []
            for nm, msk in [("2019-2022", (ev["year"] <= "2022")),
                            ("2023-2026", (ev["year"] >= "2023"))]:
                s = ev[(ev["regime"] == r) & msk]
                if len(s) < 500:
                    continue
                rows.append(row(f"[{r}] {k} | {nm}", s[combos(s)[k]]))
            show(rows, f"[{r}] {k} 分期")

    print("=" * 122)
    print("J. 连续化检验：不挑阈值，直接看 prox 与 beta 调整超额的形状（DOWN，带流动性地板+去追高+控换手）")
    print("=" * 122)
    s = ev[ev["regime"] == "DOWN"]
    b = ((s["amt30"] >= 2e8) & (s["width30"] <= 0.25)
         & (s["turn30"] <= 3.0) & (s["ret60"] <= s["ret60"].quantile(0.40)))
    edges = [0.0, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]
    names = ["<0.70", "0.70~0.75", "0.75~0.80", "0.80~0.85", "0.85~0.90", "0.90~0.95", "0.95~1.00"]
    rows = [row(f"[DOWN] 净化后 {nm}", s[(s["prox"] > lo) & (s["prox"] <= hi) & b])
            for lo, hi, nm in zip(edges[:-1], edges[1:], names)]
    show(rows, "[DOWN] prox 连续分档（已施加流动性地板+去追高+控换手）")

    s = ev[ev["regime"] == "UP"]
    b = ((s["amt30"] >= 2e8) & (s["width30"] <= 0.25)
         & (s["turn30"] <= 3.0) & (s["ret60"] <= s["ret60"].quantile(0.40)))
    rows = [row(f"[UP] 净化后 {nm}", s[(s["prox"] > lo) & (s["prox"] <= hi) & b])
            for lo, hi, nm in zip(edges[:-1], edges[1:], names)]
    show(rows, "[UP] prox 连续分档（已施加流动性地板+去追高+控换手）")


if __name__ == "__main__":
    main(load())
