# -*- coding: utf-8 -*-
"""按 TSP regime_builder 的口径，用本地数据重建 5 档市场状态。

口径来源（逐条对齐，不是近似）：
  - 涨跌停判定价: backend/app/price_limits.py::polars_limit_price
      分子 = round((1 + 涨跌幅) * 100)
      分   = floor(prev_close * 100 + 0.5)
      价   = ((分 * 分子 + 50) // 100) / 100
  - 板块涨幅: 主板 10% / 创业板20%(2020-08-24起) / 科创20% / 北交所30% / 主板ST 5%(2026-07-06前)
  - 除权日前收盘: prev = 前复权昨收 / (今前复权收 / 今未复权收)
  - 涨停信号: raw_close >= limit_up_price
  - 5 档评分: profit/speculation/resilience/trend 四维 + 权重 0.35/0.25/0.20/0.20
  - 分档: >=70 strong, >=55 lean_strong, >=45 range, >=30 lean_weak, else weak

输出: cache/regime_rebuilt.pkl（DataFrame，与 TSP 的 regime_history 同结构）
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

BASE = r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
CACHE = os.path.join(BASE, "cache")

# 阈值常量（与 TSP 一致）
GEM_20PCT_START = pd.Timestamp("2020-08-24")
MAIN_ST_CHANGE = pd.Timestamp("2026-07-06")
STATE_STRONG, STATE_LEAN_STRONG, STATE_RANGE, STATE_LEAN_WEAK = 70, 55, 45, 30
WEIGHTS = {"profit": 0.35, "speculation": 0.25, "resilience": 0.20, "trend": 0.20}


def limit_pct(code: str, d: pd.Timestamp, is_st: bool) -> float:
    is_bj = code.startswith(("4", "8"))
    is_gem = code.startswith(("300", "301", "302"))
    is_star = code.startswith(("688", "689"))
    if is_bj:
        return 0.30
    if is_gem and d < GEM_20PCT_START:
        return 0.10
    if is_gem or is_star:
        return 0.20
    if is_st and d < MAIN_ST_CHANGE:
        return 0.05
    return 0.10


def limit_price(prev: float, pct: float, up: bool) -> float:
    sign = 1 if up else -1
    numerator = int(round((1 + sign * pct) * 100))
    cents = int(np.floor(prev * 100 + 0.5))
    return ((cents * numerator + 50) // 100) / 100


def score(value, lo, hi):
    if not np.isfinite(value):
        return 50.0
    if hi <= lo:
        return 50.0
    return float(max(0, min(100, round((value - lo) / (hi - lo) * 100))))


def load_adj():
    """前复权数据：close + amount + turnover，用于涨跌幅/成交额/换手。"""
    out = {}
    for fn in ("kline_long2.pkl", "kline_320.pkl"):
        p = os.path.join(CACHE, fn)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        for code, rows in d.items():
            for r in rows:
                out.setdefault(code, {})[r["date"]] = r
        print(f"  载入 {fn}: {len(d)} 只", flush=True)
    return out


def main():
    print("载入未复权数据 ...", flush=True)
    with open(os.path.join(CACHE, "kline_raw.pkl"), "rb") as f:
        raw = pickle.load(f)
    with open(os.path.join(CACHE, "names_raw.pkl"), "rb") as f:
        names = pickle.load(f)
    print(f"  {len(raw)} 只", flush=True)

    print("载入前复权数据 ...", flush=True)
    adj = load_adj()

    # 全市场最早交易日（用于识别“老股”，避免把 2018-07-31 当成上市日）
    global_start = min(r[0]["date"] for r in raw.values() if r)
    print(f"  全局起始日: {global_start}", flush=True)

    per_stock = []
    skipped = 0
    for i, (code, rows) in enumerate(raw.items(), 1):
        if not rows:
            skipped += 1
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        a = adj.get(code, {})
        if not a:
            skipped += 1
            continue
        df["adj_close"] = df["date"].dt.strftime("%Y-%m-%d").map(lambda d: (a.get(d) or {}).get("close"))
        df["amount"] = df["date"].dt.strftime("%Y-%m-%d").map(lambda d: (a.get(d) or {}).get("amount"))
        df["turnover"] = df["date"].dt.strftime("%Y-%m-%d").map(lambda d: (a.get(d) or {}).get("turnover"))
        df = df.dropna(subset=["adj_close", "amount"])

        if len(df) < 30:
            skipped += 1
            continue

        name = names.get(code) or ""
        is_st = "ST" in name.upper()

        # 除权检测：前复权/未复权 比值变化
        ratio = df["adj_close"] / df["raw_close"].replace(0, np.nan) if "raw_close" in df else df["adj_close"] / df["close"]
        ratio_prev = ratio.shift(1)
        changed = (ratio - ratio_prev).abs() > 1e-6

        prev_raw = df["close"].shift(1)                      # 未复权昨收
        prev_adj_adjusted = df["adj_close"].shift(1) / ratio  # 除权参考价
        df["prev_raw"] = np.where(changed, prev_adj_adjusted, prev_raw)

        # 涨跌幅（用前复权）
        df["change_pct"] = df["adj_close"] / df["adj_close"].shift(1) - 1

        # 涨停/跌停价 + 信号
        df["_pct"] = [limit_pct(code, d, is_st) for d in df["date"]]
        df["limit_up_price"] = [limit_price(p, q, True) if pd.notna(p) else np.nan
                                for p, q in zip(df["prev_raw"], df["_pct"])]
        df["limit_down_price"] = [limit_price(p, q, False) if pd.notna(p) else np.nan
                                  for p, q in zip(df["prev_raw"], df["_pct"])]

        rc = df["close"]          # 未复权收盘
        rh = df["high"]
        df["signal_limit_up"] = (rc > 0) & (rc >= df["limit_up_price"])
        df["signal_limit_down"] = (rc > 0) & (rc <= df["limit_down_price"])
        df["signal_broken"] = (rh > 0) & (rc < df["limit_up_price"]) & (rh >= df["limit_up_price"])

        # 连板数
        seq = df["signal_limit_up"].fillna(False).astype(int).values
        consec = np.zeros(len(seq), dtype=int)
        run = 0
        for j, v in enumerate(seq):
            run = run + 1 if v else 0
            consec[j] = run
        df["consecutive_limit_ups"] = consec

        # MA20（前复权）
        df["ma20"] = df["adj_close"].rolling(20).mean()

        per_stock.append(df[[
            "date", "change_pct", "amount", "signal_limit_up", "signal_limit_down",
            "signal_broken", "consecutive_limit_ups", "adj_close", "ma20",
        ]])

        if i % 800 == 0:
            print(f"  处理 {i}/{len(raw)} ...", flush=True)

    print(f"有效标的: {len(per_stock)} | 跳过: {skipped}", flush=True)
    allrows = pd.concat(per_stock, ignore_index=True)
    del per_stock
    print(f"总行数: {len(allrows):,}", flush=True)

    print("按日聚合 ...", flush=True)
    g = allrows.groupby("date")
    daily = pd.DataFrame({
        "up_count": g["change_pct"].apply(lambda s: (s > 0).sum()),
        "down_count": g["change_pct"].apply(lambda s: (s < 0).sum()),
        "total_count": g.size(),
        "avg_pct": g["change_pct"].mean(),
        "median_pct": g["change_pct"].median(),
        "strong_up_count": g["change_pct"].apply(lambda s: (s >= 0.03).sum()),
        "strong_down_count": g["change_pct"].apply(lambda s: (s <= -0.03).sum()),
        "limit_up": g["signal_limit_up"].sum(),
        "limit_down": g["signal_limit_down"].sum(),
        "broken_limit": g["signal_broken"].sum(),
        "max_consecutive": g["consecutive_limit_ups"].max(),
        "total_amount": g["amount"].sum(),
    })
    valid_ma = allrows[allrows["ma20"].notna() & (allrows["ma20"] > 0)]
    above = valid_ma.assign(ab=valid_ma["adj_close"] > valid_ma["ma20"]).groupby("date")["ab"].mean()
    daily["above_ma20_pct"] = above
    daily = daily.fillna({"above_ma20_pct": 0.0}).reset_index()

    # 派生指标
    daily["up_pct"] = daily["up_count"] / daily["total_count"] * 100
    daily["down_pct"] = daily["down_count"] / daily["total_count"] * 100
    daily["strong_up_pct"] = daily["strong_up_count"] / daily["total_count"] * 100
    daily["strong_down_pct"] = daily["strong_down_count"] / daily["total_count"] * 100
    daily["strong_diff_pct"] = daily["strong_up_pct"] - daily["strong_down_pct"]
    daily["seal_rate"] = np.where(
        (daily["limit_up"] + daily["broken_limit"]) > 0,
        daily["limit_up"] / (daily["limit_up"] + daily["broken_limit"]), 0.5)

    # 指数涨幅（沪深300，本地缓存）
    idx_path = os.path.join(CACHE, "tdx_idx_000300.pkl")
    if os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        idx = idx[["date", "close"]].copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.sort_values("date")
        idx["index_pct"] = idx["close"] / idx["close"].shift(1) - 1
        daily = daily.merge(idx[["date", "index_pct"]], on="date", how="left")
    else:
        daily["index_pct"] = 0.0
    daily["index_pct"] = daily["index_pct"].fillna(0.0)

    print("计算 5 档状态 ...", flush=True)
    rows = []
    for r in daily.itertuples(index=False):
        d = r._asdict()
        profit = (score(d["up_pct"], 21, 75) * 0.45
                  + score(d["avg_pct"] * 100, -1.2, 1.3) * 0.25
                  + score(d["median_pct"] * 100, -1.2, 1.3) * 0.20
                  + score(d["strong_diff_pct"], -13, 14) * 0.10)
        spec = (score(d["limit_up"], 35, 97) * 0.30
                + score((d["seal_rate"] or 0.5) * 100, 57, 75) * 0.40
                + score(d["max_consecutive"], 4, 9) * 0.30)
        resil = 100 - score(d["strong_down_pct"], 2, 18)
        trend = (score(d["index_pct"] * 100, -2.5, 2.5) * 0.50
                 + score((d["above_ma20_pct"] or 0.5) * 100, 22, 76) * 0.50)
        total = (profit * WEIGHTS["profit"] + spec * WEIGHTS["speculation"]
                 + resil * WEIGHTS["resilience"] + trend * WEIGHTS["trend"])
        total = max(0, min(100, round(total)))
        if total >= STATE_STRONG:
            state = "strong"
        elif total >= STATE_LEAN_STRONG:
            state = "lean_strong"
        elif total >= STATE_RANGE:
            state = "range"
        elif total >= STATE_LEAN_WEAK:
            state = "lean_weak"
        else:
            state = "weak"
        rows.append({**d, "profit": profit, "speculation": spec, "resilience": resil,
                     "trend": trend, "score": total, "state": state})

    out = pd.DataFrame(rows)
    out_path = os.path.join(CACHE, "regime_rebuilt.pkl")
    out.to_pickle(out_path)
    print(f"\n完成: {len(out)} 天  {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"输出: {out_path}")
    print("\n状态分布:")
    print(out["state"].value_counts().to_string())


if __name__ == "__main__":
    main()
