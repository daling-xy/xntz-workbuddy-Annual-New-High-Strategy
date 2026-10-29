# -*- coding: utf-8 -*-
"""
周期分层校准：回答「下跌趋势下要不要放宽距年高要求」。

必须区分两个被混淆的问题：
  (A) 横截面问题 —— 同一天里，选距年高近的票还是选回调深的票？
      用 ex_fwd（当日全市场去均值后的超额收益）衡量。这是「选股能力」。
  (B) 时序问题   —— 下跌市里这套策略整体上还赚不赚钱？要不要减仓/停手？
      用 fwd（原始收益）衡量。这是「仓位/参与强度」。

统计口径：所有区块相关性极强（同一天几千只票同涨同跌），
因此用「按交易日聚类的自助法」给出置信区间，而不是按事件数算标准误。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_regime_label import build as build_regime  # noqa: E402
from yh_eventstudy_long import build_event_table  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 800
_rng = np.random.default_rng(20261022)


# --------------------------------------------------------------------------
# 统计工具
# --------------------------------------------------------------------------
def cluster_boot_ci(sub: pd.DataFrame, key: str, n_boot: int = BOOT) -> tuple:
    """按交易日整簇自助抽样 → 均值点估计与 90% 置信区间"""
    dates = sub["date"].unique()
    vals = sub.groupby("date")[key].mean().values      # 每个交易日的组内均值
    if len(vals) < 3:
        return (np.nan, np.nan, np.nan)
    est = vals.mean()
    idx = _rng.integers(0, len(vals), size=(n_boot, len(vals)))
    bs = vals[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def blk(sub: pd.DataFrame, label: str, key_ex: str = "ex_fwd60",
        key_raw: str = "fwd60", boot: bool = True) -> dict:
    if len(sub) == 0:
        return {"分组": label, "事件数": 0}
    ex, lo, hi = cluster_boot_ci(sub, key_ex) if boot else (np.nan,) * 3
    raw = sub[key_raw].mean()
    return {
        "分组": label,
        "事件数": int(len(sub)),
        "交易日": int(sub["date"].nunique()),
        "股票": int(sub["code"].nunique()),
        "超额均值%": round(sub[key_ex].mean() * 100, 2),
        "超额中位%": round(sub[key_ex].median() * 100, 2),
        "胜率%": round((sub[key_ex] > 0).mean() * 100, 1),
        "簇自助90%CI": f"[{lo*100:.1f}, {hi*100:.1f}]" if boot else "-",
        "绝对收益%": round(raw * 100, 2),
        "mae20中位%": round(sub["mae20"].median() * 100, 1),
        "mfe60中位%": round(sub["mfe60"].median() * 100, 1),
    }


def show(rows, title: str, cols=None):
    df = pd.DataFrame([r for r in rows if r.get("事件数", 0) > 0])
    if not len(df):
        print(f"--- {title} --- 无样本\n")
        return None
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    print(f"--- {title} ---")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(df.to_string(index=False))
    print()
    return df


MAIN_COLS = ["分组", "事件数", "交易日", "股票", "超额均值%", "超额中位%", "胜率%",
             "簇自助90%CI", "绝对收益%", "mae20中位%", "mfe60中位%"]


def load():
    path = os.path.join(CACHE, "events_long_regime.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    ev = build_event_table()
    st = build_regime()[["date", "breadth", "bull_stack", "regime",
                         "沪深300_pos", "沪深300_ret60", "沪深300_dd250"]]
    ev = ev.merge(st, on="date", how="inner")
    print("合并后事件数:", len(ev), " 区间:", ev["date"].min(), "~", ev["date"].max())
    with open(path, "wb") as f:
        pickle.dump(ev, f)
    return ev


# --------------------------------------------------------------------------
def part0(ev):
    print("=" * 100)
    print("第 0 部分 · 样本与周期分布")
    print("=" * 100)
    y = ev.copy(); y["year"] = y["date"].str[:4]
    print("按年：")
    print(y.groupby("year").agg(事件数=("code", "size"), 股票数=("code", "nunique"),
                                交易日=("date", "nunique")).to_string())
    print()
    d = ev.drop_duplicates("date")[["date", "regime", "breadth", "沪深300_dd250"]].copy()
    d["year"] = d["date"].str[:4]
    tab = d.groupby(["year", "regime"]).size().unstack(fill_value=0)
    for c in ["UP", "CHOP", "DOWN"]:
        if c not in tab:
            tab[c] = 0
    tab = tab[["UP", "CHOP", "DOWN"]]
    tab["天数"] = tab.sum(axis=1)
    print("交易日 Regime 分布：")
    print(tab.to_string())
    print()
    print("各 Regime 的沪深300 距年高均值：")
    print(d.groupby("regime")["沪深300_dd250"].agg(["mean", "median", "min"]).round(3).to_string())
    print()


def part1(ev):
    print("=" * 100)
    print("第 1 部分 · 时序视角：不同周期下，这套策略整体上赚不赚钱？（原始收益 fwd60）")
    print("=" * 100)
    base = (ev["width30"] <= 0.25) & (ev["amt30"] >= 2e8)
    rows = []
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        rows.append(blk(s, f"[{r}] 全市场", boot=True))
        rows.append(blk(s[s["prox"] >= 0.90], f"[{r}] prox>=0.90", boot=True))
        rows.append(blk(s[(s["prox"] >= 0.90) & base], f"[{r}] prox>=.90+宽25%+额2亿", boot=True))
    show(rows, "不同周期下的原始 60 日收益", MAIN_COLS)


def part2(ev):
    print("=" * 100)
    print("第 2 部分 · prox 距年高阈值 × 周期（横截面超额 vs 绝对收益）")
    print("=" * 100)
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        if not len(s):
            continue
        rows = []
        for v in [0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 1.0]:
            sub = s[s["prox"] >= v]
            rows.append(blk(sub, f"prox>={v:.2f}"))
        show(rows, f"[{r}] 累积式 prox 门槛（含 CI 为该分组当日平均超额的 90% 簇自助区间）", MAIN_COLS)

    print(">>> 分档式（互斥区间），更能看出单调性是否真的成立 <<<")
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        if not len(s):
            continue
        edges = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0, np.inf]
        names = ["<0.70", "0.70~0.80", "0.80~0.85", "0.85~0.90",
                 "0.90~0.95", "0.95~1.00", "创新高(>=1.00)"]
        rows = []
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            rows.append(blk(s[(s["prox"] > lo) & (s["prox"] <= hi)], nm))
        show(rows, f"[{r}] 分档式 prox", MAIN_COLS)


def part3(ev):
    print("=" * 100)
    print("第 3 部分 · 用「当前价距年高的回撤」dd_from_high 重做同一件事")
    print("=" * 100)
    print("（prox 用的是 30 日高点，dd_from_high 用当前价。下跌市里两者差异会被放大）")
    print()
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        if not len(s):
            continue
        edges = [-np.inf, -0.40, -0.30, -0.20, -0.10, -0.05, 0.0]
        names = ["回撤>40%", "回撤30~40%", "回撤20~30%", "回撤10~20%", "回撤5~10%", "新高附近(<5%)"]
        rows = []
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            rows.append(blk(s[(s["dd_from_high"] > lo) & (s["dd_from_high"] <= hi)], nm))
        show(rows, f"[{r}] 当前价距 250 日高点回撤分档", MAIN_COLS)


def part4(ev):
    print("=" * 100)
    print("第 4 部分 · 下跌不是铁板一块：按「市场已回撤深度」再切一刀")
    print("=" * 100)
    for nm, mask in [("轻度下跌 指数距年高<15%",
                      (ev["沪深300_dd250"] > -0.15)),
                     ("深度下跌 指数距年高>=15%",
                      (ev["沪深300_dd250"] <= -0.15))]:
        rows = []
        for v in [0.70, 0.80, 0.85, 0.90, 0.95]:
            sub = ev[mask & (ev["prox"] >= v)]
            rows.append(blk(sub, f"{nm} · prox>={v:.2f}"))
        show(rows, nm, MAIN_COLS)

    print(">>> 进一步：把「下跌末期的初期修复」单独拎出来 —— 宽度刚从低位回升 <<<")
    st = ev.drop_duplicates("date")[["date", "breadth"]].sort_values("date")
    st["b_chg20"] = st["breadth"].pct_change(20)
    st["b_ma60"] = st["breadth"].rolling(60).mean()
    ev2 = ev.merge(st[["date", "b_chg20", "b_ma60"]], on="date", how="left")
    for nm, mask in [("宽度低位(<=0.35) 且开始回升(b_chg20>0.5)",
                      (ev2["breadth"] <= 0.35) & (ev2["b_chg20"] > 0.5)),
                     ("宽度低位 且仍在恶化(b_chg20<-0.2)",
                      (ev2["breadth"] <= 0.35) & (ev2["b_chg20"] < -0.2))]:
        rows = []
        for v in [0.70, 0.80, 0.85, 0.90, 0.95]:
            rows.append(blk(ev2[mask & (ev2["prox"] >= v)], f"{nm} · prox>={v:.2f}"))
        show(rows, nm, MAIN_COLS)


def part5(ev):
    print("=" * 100)
    print("第 5 部分 · 其它因子在不同周期下是否需要换一套？")
    print("=" * 100)
    for r in ["UP", "DOWN"]:
        s = ev[(ev["regime"] == r) & (ev["prox"] >= 0.85)]
        if len(s) < 500:
            continue
        print(f"### [{r}] 固定 prox>=0.85 后...")
        rows = []
        for v in [0.15, 0.20, 0.25, 0.30, 0.40, 9.9]:
            rows.append(blk(s[s["width30"] <= v], f"  箱宽30<={v:.0%}"))
        show(rows, f"[{r}] 箱体宽度", MAIN_COLS)

        rows = []
        qs = s["ret60"].quantile([.2, .4, .6, .8]).values
        bins = [-np.inf] + list(qs) + [np.inf]
        nm = ["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"]
        for lo, hi, n in zip(bins[:-1], bins[1:], nm):
            rows.append(blk(s[(s["ret60"] > lo) & (s["ret60"] <= hi)], f"  近60日收益{n}"))
        show(rows, f"[{r}] 中期动量", MAIN_COLS)

        rows = []
        for thr, n in [(1.0, "量能<=1.0(缩量)"), (1.3, "量能<=1.3"), (2.0, "量能<=2.0"),
                       (99, "不限")]:
            rows.append(blk(s[s["amt_act"] <= thr], f"  {n}"))
        show(rows, f"[{r}] 相对量能 amt_act(近30日额/过去一年日均额)", MAIN_COLS)


def part6(ev):
    print("=" * 100)
    print("第 6 部分 · 直接对比候选方案：放宽 prox 到底多放了些票进来？")
    print("=" * 100)
    d = ev.drop_duplicates("date")[["date"]]
    rows = []
    for r in ["UP", "CHOP", "DOWN"]:
        s = ev[ev["regime"] == r]
        if not len(s):
            continue
        for v in [0.90, 0.85, 0.80, 0.70]:
            g = s[(s["prox"] >= v) & (s["width30"] <= 0.25) & (s["amt30"] >= 2e8)]
            if not len(g):
                continue
            per = g.groupby("date").size()
            rows.append({
                "Regime": r, "prox门槛": v,
                "日均候选数": round(float(per.mean()), 1),
                "中位": int(per.median()),
                "超额均值%": round(g["ex_fwd60"].mean() * 100, 2),
                "绝对收益%": round(g["fwd60"].mean() * 100, 2),
                "mae20中位%": round(g["mae20"].median() * 100, 1),
                "mfe60中位%": round(g["mfe60"].median() * 100, 1),
                "亏损>20%占比%": round((g["fwd60"] < -0.20).mean() * 100, 1),
                "盈利>20%占比%": round((g["fwd60"] > 0.20).mean() * 100, 1),
            })
    show(rows, "放宽门槛的代价与收益",
         ["Regime", "prox门槛", "日均候选数", "中位", "超额均值%", "绝对收益%",
          "mae20中位%", "mfe60中位%", "亏损>20%占比%", "盈利>20%占比%"])


if __name__ == "__main__":
    ev = load()
    for fn in (part0, part1, part2, part3, part4, part5, part6):
        fn(ev)
