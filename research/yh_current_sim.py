# -*- coding: utf-8 -*-
"""
当前截面落地验证（2026-11-10）
================================================================
把历史实证的结论落到「今天到底会选到谁」：
  1. 池子实绩：11-02~11-09 核心池等权 vs 全A等权 vs 用户关注的三只弹性票
  2. 用 11-09 收盘的完整横截面重建六道门，复现现有池
  3. 模拟「截面自适应分位门槛」会放进哪些票，看是否覆盖想抓的弹性票
  4. 用户关注票在各门下的卡点

数据来源：腾讯前复权日线（K线缓存），基准日 2026-11-09 收盘。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE_DIR, "cache")
OUT = os.path.join(BASE_DIR, "output", "current_sim_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


POOL9 = ["600900", "000333", "600674", "000651", "601919",
         "601985", "000538", "601088", "600886"]
WATCH = {"688235": "百济神州", "603127": "昭衍新药", "600795": "国电电力"}


def build_cross(d):
    """用截至 11-09 收盘（倒数第2根bar）的数据重建当日横截面。"""
    rows = []
    for c, b in d.items():
        if not b or len(b) < 252:
            continue
        bars = b[:-1]                      # 丢掉 11-10 盘中未完成 bar
        cl = np.array([x["close"] for x in bars], dtype=float)
        am = np.array([x.get("amount", 0) or 0 for x in bars], dtype=float)
        tn = np.array([x.get("turnover", 0) or 0 for x in bars], dtype=float)
        c0 = cl[-1]
        hi250 = cl[-250:].max()
        w = cl[-30:]
        rows.append((c, bars[-1].get("date"), c0, c0 / hi250,
                     w.max() / w.min() - 1.0, am[-30:].mean(), tn[-30:].mean(),
                     c0 / cl[-21] - 1.0, c0 / cl[-61] - 1.0))
    return pd.DataFrame(rows, columns=["code", "date", "close", "prox_now", "width30",
                                       "amt30", "turn30", "ret20", "ret60"])


def main():
    d = pd.read_pickle(os.path.join(CACHE, "kline_320.pkl"))
    P("=" * 108)
    P("当前截面落地验证 · 2026-11-10 · 数据来源：腾讯前复权日线 / 通达信指数")
    P("=" * 108)

    # ---------- 1. 池子实绩 ----------
    P("\n【1】11-02 收盘 → 11-09 收盘，各组表现")
    def px(bars, date):
        for x in bars:
            if x.get("date") == date:
                return x["close"]
        return np.nan
    def ret_between(code, d0, d1):
        b = d.get(code)
        if not b:
            return np.nan
        p0, p1 = px(b, d0), px(b, d1)
        return p1 / p0 - 1 if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else np.nan
    allret = pd.Series({c: ret_between(c, "2026-11-02", "2026-11-09") for c in d}).dropna()
    P("  全A等权（%d 只）      %+7.2f%%   中位数 %+7.2f%%   上涨占比 %.1f%%" % (
        len(allret), allret.mean() * 100, allret.median() * 100, (allret > 0).mean() * 100))
    pr = [ret_between(c, "2026-11-02", "2026-11-09") for c in POOL9]
    P("  11-09 核心池9只等权   %+7.2f%%" % (np.nanmean(pr) * 100))
    for c, n in WATCH.items():
        P("  关注票 %-8s %+7.2f%%" % (n, ret_between(c, "2026-11-02", "2026-11-09") * 100))
    # 11-02 时点的强势票（近20日涨幅前20%）之后表现
    r20 = {}
    for c, b in d.items():
        if b and len(b) >= 22:
            idx = next((i for i, x in enumerate(b) if x.get("date") == "2026-11-02"), None)
            if idx and idx >= 20:
                r20[c] = b[idx]["close"] / b[idx - 20]["close"] - 1
    r20s = pd.Series(r20)
    hot = r20s[r20s >= r20s.quantile(0.80)].index
    hr = allret.reindex(hot).dropna()
    P("  11-02 时点强势票(近20日前20%%, %d只) 此后 %+7.2f%%" % (len(hr), hr.mean() * 100))

    # ---------- 2. 重建当日横截面与固定门 ----------
    df = build_cross(d)
    P("\n【2】11-09 收盘横截面：%d 只（%s）" % (len(df), df["date"].mode().iloc[0]))
    df["w_rank"] = df["width30"].rank(pct=True)
    df["t_rank"] = df["turn30"].rank(pct=True)
    other = (df["prox_now"] >= 0.95) & (df["amt30"] >= 4e8) & (df["ret60"] <= 0.19)
    fixed = df[other & (df["width30"] <= 0.15) & (df["turn30"] <= 6.0)]
    P("  固定门槛（prox≥0.95 / 箱宽≤15%% / 换手≤6%% / 额≥4亿 / ret60≤19%%）：%d 只" % len(fixed))
    P("  → 再叠加财报清洗层 + 行业剔除后即当前 9 只核心池（口径差异来自财报/行业层）")
    P("  全市场箱宽分位：15%% 绝对值对应当日分位 %.3f" % (
        (df["width30"] <= 0.15).mean()))
    P("  全市场换手分位：6%% 绝对值对应当日分位 %.3f" % (
        (df["turn30"] <= 6.0).mean()))
    P("  ★ 关键：固定 15%% 箱宽在当日只放行 %.1f%% 的票（历史平静期通常 25~35%%）" % (
        (df["width30"] <= 0.15).mean() * 100))

    # ---------- 3. 自适应分位门槛模拟 ----------
    P("\n【3】截面自适应门槛模拟（把绝对箱宽/换手换成当日横截面分位）")
    P("  %-8s %8s %s" % ("分位q", "入池数", "新增票（相对固定门，按 prox 排序取前12）"))
    P("  " + "-" * 100)
    for q in [0.15, 0.25, 0.35, 0.50]:
        ad = df[other & (df["w_rank"] <= q) & (df["t_rank"] <= q)]
        new = ad[~ad["code"].isin(fixed["code"])].sort_values("prox_now", ascending=False)
        names = []
        for c in new["code"].head(12):
            nm = WATCH.get(str(c).zfill(6))
            names.append(nm if nm else str(c).zfill(6))
        P("  q=%.2f   %8d   %s" % (q, len(ad), "、".join(names)))

    # ---------- 4. 关注票卡点 ----------
    P("\n【4】三只关注票在 11-09 横截面中的位置")
    P("  %-10s %9s %9s %9s %9s %9s" % ("票", "prox_now", "箱宽", "箱宽分位", "换手", "换手分位"))
    P("  " + "-" * 64)
    for c, n in WATCH.items():
        r = df[df["code"].astype(str).str.zfill(6) == c]
        if len(r) == 0:
            P("  %-10s 不在横截面" % n)
            continue
        r = r.iloc[0]
        P("  %-10s %9.3f %8.1f%% %9.3f %8.2f%% %9.3f" % (
            n, r["prox_now"], r["width30"] * 100, r["w_rank"], r["turn30"], r["t_rank"]))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
