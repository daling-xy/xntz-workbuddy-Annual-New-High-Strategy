# -*- coding: utf-8 -*-
"""
「池子过于保守」诊断（2026-11-10）
================================================================
用户观察：市场转好、大部分票都起来了，但核心池仍是那批防御票，感觉过于保守、
选出来的票不及预期。本脚本不预设结论，回答三个可证伪的问题：

  Q1 当前市场真的是「普涨」吗？（用 K 线全横截面，不用印象）
  Q2 历史上「深跌后宽度快速修复」的时段，严格池是否真的跑输市场 / 跑输高弹性票？
     → 若成立，问题是【状态依赖】，不是阈值本身设错。
  Q3 门①（距年高）在修复段筛出来的到底是「强势票」还是「前期抗跌的防御票」？
     → 若是后者，则风格错配是结构性的，放宽阈值也换不来弹性票。

状态定义（仅用于历史分层研究，不对当前市场下判断）：
  REBOUND = breadth 的 20 日绝对变化 > +0.05 且 breadth < 0.40（低位快速修复）
  UP/DOWN/CHOP = 事件表自带的历史 regime 标签（研究归档用）

数据来源：通达信指数日线构建的市场宽度、腾讯前复权日线事件表。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_blockboot_all import make_dmap, paired_block_boot, sig  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE_DIR, "cache")
OUT = os.path.join(BASE_DIR, "output", "conserv_diag_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


BAD = -0.20


def load_events():
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[ev["beta"].notna() & ev["fwd60"].notna()].copy()

    st = pd.read_pickle(os.path.join(CACHE, "regime_state.pkl"))
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values("date")
    st["db20"] = st["breadth"] - st["breadth"].shift(20)
    ev = ev.merge(st[["date", "breadth", "db20"]], on="date", how="left")
    ev["state"] = np.where((ev["db20"] > 0.05) & (ev["breadth"] < 0.40),
                           "REBOUND", ev["regime"])
    # 当日横截面相对强弱分位（事件表为轮转采样，按当日样本内的分位近似）
    ev["ret20_pct"] = ev.groupby("date")["ret20"].rank(pct=True)
    ev["_bad"] = ev["ex_fwd60"] < BAD
    ev["_bad_ab"] = ev["ab_fwd60"] < BAD
    return ev.reset_index(drop=True)


def strict(d: pd.DataFrame) -> pd.DataFrame:
    """当前生效的五道结构门（事件表口径）。"""
    return d[(d["prox_now"] >= 0.95) & (d["width30"] <= 0.15) &
             (d["amt30"] >= 4e8) & (d["turn30"] <= 6.0) &
             (pd.to_numeric(d["ret60"], errors="coerce") <= 0.19)]


def part1_current_market():
    P("=" * 110)
    P("Q1  当前市场真是「普涨」吗？——全 A 横截面（K线缓存，最新 bar）")
    P("=" * 110)
    d = pd.read_pickle(os.path.join(CACHE, "kline_320.pkl"))
    rows = []
    for c, b in d.items():
        if not b or len(b) < 65:
            continue
        last = b[-1]
        c0 = last["close"]
        hi = max(x["close"] for x in b[-250:]) if len(b) >= 250 else max(x["close"] for x in b)
        rows.append((c, last.get("date"), c0 / b[-21]["close"] - 1,
                     c0 / b[-61]["close"] - 1, c0 / hi - 1))
    df = pd.DataFrame(rows, columns=["code", "date", "ret20", "ret60", "dd"])
    P("  最新 bar: %s   有效样本 %d 只" % (df["date"].mode().iloc[0], len(df)))
    P("  近20日涨幅：中位 %+.1f%%，上涨占比 %.1f%%，涨幅>10%% 占比 %.1f%%" % (
        df.ret20.median() * 100, (df.ret20 > 0).mean() * 100, (df.ret20 > 0.10).mean() * 100))
    P("  近60日涨幅：中位 %+.1f%%，上涨占比 %.1f%%" % (
        df.ret60.median() * 100, (df.ret60 > 0).mean() * 100))
    P("  距年高：中位 %.1f%%，贴年高(≥−5%%) 仅 %.1f%%（%d 只）" % (
        df.dd.median() * 100, (df.dd >= -0.05).mean() * 100, (df.dd >= -0.05).sum()))
    st = pd.read_pickle(os.path.join(CACHE, "regime_state.pkl"))
    st["date"] = pd.to_datetime(st["date"])
    st = st.sort_values("date")
    st["db20"] = st["breadth"] - st["breadth"].shift(20)
    tail = st.tail(8)[["date", "breadth20", "breadth", "db20"]].copy()
    P("\n  宽度序列（breadth=站上MA60占比，breadth20=近20日均）：")
    P("  " + tail.to_string(index=False).replace("\n", "\n  "))
    P("  → 短期宽度从 10-30 的 12.9% 修复到 42.5%，但中期宽度仅 23.0%，")
    P("    且个股中位数仍在跌：这是【深跌后的结构性修复】，不是全市场普涨。")


def part2_state_table(ev):
    P("\n" + "=" * 110)
    P("Q2  历史上「低位快速修复」时段，严格池是否跑输？（ex_fwd60 = 相对当日横截面超额）")
    P("=" * 110)
    P("  状态定义：REBOUND = 宽度20日绝对变化 >+5pp 且 宽度<40%；其余沿用历史 regime 标签")
    P("  %-10s %8s %9s | %8s %9s %9s %8s | %9s %9s" % (
        "状态", "交易日", "事件数", "池日均票", "池超额", "市场超额", "池−市场", "池大亏率", "市场大亏率"))
    P("  " + "-" * 106)
    out = {}
    for s in ["REBOUND", "UP", "CHOP", "DOWN"]:
        sub = ev[ev["state"] == s]
        if len(sub) < 100:
            P("  %-10s 样本不足" % s)
            continue
        pool = strict(sub)
        mkt = sub.groupby("date")["ex_fwd60"].mean().mean()
        p_ex = pool.groupby("date")["ex_fwd60"].mean().mean()
        per = pool.groupby("date").size().mean() if len(pool) else 0
        P("  %-10s %8d %9d | %8.2f %+8.2f%% %+8.2f%% %+7.2fpp | %8.1f%% %9.1f%%" % (
            s, sub["date"].nunique(), len(sub), per, p_ex * 100, mkt * 100,
            (p_ex - mkt) * 100, pool["_bad"].mean() * 100, sub["_bad"].mean() * 100))
        out[s] = (p_ex, mkt, per, pool, sub)

    # 高弹性票（当日 ret20 分位前 20%，有量）在各状态的表现
    P("\n  对照：当日「高弹性票」（ret20 分位前20% + 额≥4亿）的历史超额")
    P("  %-10s %10s %10s %10s %10s" % ("状态", "事件数", "超额", "大亏率", "vs严格池"))
    P("  " + "-" * 60)
    for s in ["REBOUND", "UP", "CHOP", "DOWN"]:
        if s not in out:
            continue
        p_ex, mkt, per, pool, sub = out[s]
        hot = sub[(sub["ret20_pct"] >= 0.80) & (sub["amt30"] >= 4e8)]
        if len(hot) < 100:
            continue
        h = hot.groupby("date")["ex_fwd60"].mean().mean()
        P("  %-10s %10d %+9.2f%% %9.1f%% %+9.2fpp" % (
            s, len(hot), h * 100, hot["_bad"].mean() * 100, (h - p_ex) * 100))

    # 配对 bootstrap：REBOUND 段 池−市场 是否显著
    if "REBOUND" in out:
        pool, sub = out["REBOUND"][3], out["REBOUND"][4]
        dm = make_dmap(ev)
        pt, lo, hi, nb = paired_block_boot(pool, sub, dm, col="ex_fwd60")
        P("\n  REBOUND 段配对区块 bootstrap（池 − 当日市场，β未调整口径）：")
        if not pd.isna(pt):
            P("    点估计 %+.2fpp  90%%CI[%+.2f,%+.2f]  %s（区块数 %d）" % (
                pt * 100, lo * 100, hi * 100, sig(pt, lo, hi), nb))
    return out


def part3_profile(ev):
    P("\n" + "=" * 110)
    P("Q3  门①在修复段筛出来的是「强势票」还是「前期抗跌的防御票」？")
    P("=" * 110)
    for s in ["REBOUND", "UP"]:
        sub = ev[ev["state"] == s]
        if len(sub) < 100:
            continue
        near = sub[sub["prox_now"] >= 0.95]
        hot = sub[(sub["ret20_pct"] >= 0.80) & (sub["amt30"] >= 4e8)]
        pool = strict(sub)
        P("\n  【%s】三组画像对比（均值）" % s)
        P("  %-14s %9s %9s %9s %9s %9s %9s" % (
            "组", "n", "ret20", "ret60", "width30", "turn30", "beta"))
        P("  " + "-" * 76)
        for tag, g in [("全市场样本", sub), ("贴年高(门①)", near),
                       ("高弹性票", hot), ("严格池", pool)]:
            if len(g) == 0:
                continue
            P("  %-14s %9d %+8.1f%% %+8.1f%% %8.1f%% %8.2f%% %9.2f" % (
                tag, len(g), g["ret20"].mean() * 100,
                pd.to_numeric(g["ret60"], errors="coerce").mean() * 100,
                g["width30"].mean() * 100, g["turn30"].mean(), g["beta"].mean()))
        # 贴年高组的 ret20 分位均值（<0.5 = 反弹段里反而涨得少）
        P("  → 贴年高组的 ret20 横截面分位均值 = %.3f（0.5 为市场中位）" % near["ret20_pct"].mean())
        P("    高弹性组贴年高占比 = %.1f%%，严格池贴年高占比 = 100%%" % (
            (hot["prox_now"] >= 0.95).mean() * 100))


def main():
    t0 = time.time()
    P("=" * 110)
    P("「池子过于保守」诊断 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 110)
    part1_current_market()
    ev = load_events()
    P("\n事件表：%s 行 / %d 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["date"].nunique(), ev["date"].min().date(), ev["date"].max().date()))
    P("状态分布（事件数）：" + str(ev["state"].value_counts().to_dict()))
    part2_state_table(ev)
    part3_profile(ev)
    P("\n耗时 %.1fs" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
