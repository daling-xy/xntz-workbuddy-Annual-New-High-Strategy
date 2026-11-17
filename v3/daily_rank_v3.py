# -*- coding: utf-8 -*-
"""daily_rank_v3.py —— 年度新高策略 v3（推倒重来版）每日选股
================================================================
设计（2026-11-13 定版）：
  宇宙：当日成交额>=2亿 & 非涨停 & 非ST/退 & 非9类行业黑名单
        & vol20 <= 1.0 x 自身过去一年 vol20 均值（按两位小数比较，含四舍五入；2026-11-17 晚按用户要求回退自 1.2）
        & vol20 <= 35%（绝对保险丝）
  条件：prox >= 0.90 & ret10 <= 8% & turn30 <= 5% & ret60 >= 0
  排序：prox（双向样本外唯一通过的排序特征）
  分档：A >= 0.98 / B >= 0.95 / C >= 0.90（不设名额）

用法：
  1) 先刷新行情（仓库根目录）：python run_daily.py --fresh --asof 0
  2) 再跑本脚本：python v3/daily_rank_v3.py
  输出：output/v3/daily_rank_v3_YYYYMMDD.csv

依赖（可用环境变量覆盖）：
  YW_CACHE  kline_320.pkl 路径（默认 cache/kline_320.pkl）
  YW_BOARD  全市场打分明细 路径（默认 output/v2/全市场打分明细_v2.csv；
            缺失时退回 v3/data/board_map.csv 快照）
  YW_ID     行业身份证表（默认 v3/data/deliver_identity_table.csv）
  YW_V3_OUT 输出目录（默认 output/v3）
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("YW_CACHE") or os.path.join(BASE, "cache", "kline_320.pkl")
BOARD = os.environ.get("YW_BOARD")
if not BOARD:
    for _cand in (os.path.join(BASE, "cache", "board_map.csv"),
                  os.path.join(BASE, "output", "v2", "全市场打分明细_v2.csv"),
                  os.path.join(BASE, "v3", "data", "board_map.csv")):
        if os.path.exists(_cand):
            BOARD = _cand
            break
if not BOARD:
    BOARD = os.path.join(BASE, "v3", "data", "board_map.csv")
ID = os.environ.get("YW_ID") or os.path.join(BASE, "v3", "data", "deliver_identity_table.csv")
OUTDIR = os.environ.get("YW_V3_OUT") or os.path.join(BASE, "output", "v3")

BLACK = {"银行Ⅱ", "基础建设", "房屋建设Ⅱ", "专业工程", "工程咨询服务Ⅱ",
         "铁路公路", "物流", "航空机场", "通信服务"}

# 参数（2026-11-13 定版，详见 v3/README_v3.md）
P_VOL_RATIO = 1.0      # vol20 / 自身一年均值 上限（按两位小数比较，即四舍五入后 ≤1.0）
P_VOL_CAP = 0.35       # vol20 绝对上限
P_PROX = 0.90          # 距年高下限
P_RET10 = 0.08         # 近10日涨幅上限
P_TURN30 = 5.0         # 30日平均换手上限（%）
P_RET60 = 0.0          # 近60日涨幅下限
TIER_A, TIER_B = 0.98, 0.95


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    kl = pickle.load(open(CACHE, "rb"))
    codes = sorted(kl.keys())
    all_dates = sorted({b["date"] for v in kl.values() for b in v})
    di = {d: i for i, d in enumerate(all_dates)}
    nd, nc = len(all_dates), len(codes)
    close = np.full((nd, nc), np.nan, np.float32)
    amt = np.full((nd, nc), np.nan, np.float32)
    turn = np.full((nd, nc), np.nan, np.float32)
    for j, c in enumerate(codes):
        for b in kl[c]:
            i = di[b["date"]]
            close[i, j] = b["close"]
            amt[i, j] = b["amount"]
            turn[i, j] = b.get("turnover", np.nan)
    dates = pd.to_datetime(all_dates)
    px = pd.DataFrame(close, index=dates)
    am = pd.DataFrame(amt, index=dates)
    tn = pd.DataFrame(turn, index=dates)

    ret = px.pct_change(fill_method=None)
    vol20 = (ret.rolling(20).std().values) * np.sqrt(252)
    vol_hist = pd.DataFrame(vol20).rolling(250).mean().values
    volratio = vol20 / np.where(vol_hist > 0, vol_hist, np.nan)
    prox = (px / px.rolling(250).max()).values
    ret10 = (px / px.shift(10) - 1).values
    ret60 = (px / px.shift(60) - 1).values
    turn30 = tn.rolling(30).mean().values

    idn = pd.read_csv(ID, dtype={"code": str}).set_index("code")
    name = np.array([idn.loc[x, "name"] if x in idn.index else "" for x in codes])
    st = np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
    bm = pd.read_csv(BOARD, dtype={"code": str}, usecols=["code", "board"])         .drop_duplicates("code").set_index("code")["board"]
    board = bm.reindex(codes).values
    bad = np.array([(isinstance(b, str) and b in BLACK) for b in board])
    clean = ~st & ~bad

    asof_offset = int(os.environ.get("YH_ASOF", "0"))   # 0=最新bar（收盘后）；1=倒数第2根（盘中安全）
    i = nd - 1 - asof_offset
    c = px.values[i]
    today_amt = am.values[i] / 1e8
    today_chg = c / px.values[i - 1] - 1
    not_limit_up = today_chg < 0.098        # 当日涨幅 >=9.8% 视为涨停/接近涨停，买不进
    liq = (today_amt >= 2.0) & ~np.isnan(c) & clean & not_limit_up
    base = (liq
            & (np.round(volratio[i], 2) <= P_VOL_RATIO) & (vol20[i] <= P_VOL_CAP)
            & (prox[i] >= P_PROX)
            & (ret10[i] <= P_RET10) & (turn30[i] <= P_TURN30) & (ret60[i] >= P_RET60))
    idx = np.where(base)[0]

    def tier(p: float) -> str:
        return "A" if p >= TIER_A else ("B" if p >= TIER_B else "C")

    rows = []
    for j in idx:
        r = idn.loc[codes[j]] if codes[j] in idn.index else None
        rows.append(dict(
            date=dates[i].date(), tier=tier(float(prox[i, j])),
            prox=round(float(prox[i, j]), 4), code=codes[j], name=name[j],
            board=board[j] if isinstance(board[j], str) else "",
            industry=r["主行业(赛道)"] if r is not None else "",
            anchor=r["定价锚(申万三级)"] if r is not None else "",
            close=round(float(c[j]), 2),
            chg=round(float(c[j] / px.values[i - 1, j] - 1) * 100, 2),
            vol20=round(float(vol20[i, j]) * 100, 1),
            volratio=round(float(volratio[i, j]), 2),
            ret10=round(float(ret10[i, j]) * 100, 1),
            ret60=round(float(ret60[i, j]) * 100, 1),
            turn30=round(float(turn30[i, j]), 2),
            amt_yi=round(float(today_amt[j]), 2),
        ))
    R = pd.DataFrame(rows).sort_values("prox", ascending=False)
    out = os.path.join(OUTDIR, f"daily_rank_v3_{dates[i].strftime('%Y%m%d')}.csv")
    R.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"asof {dates[i].date()} | 清洗排除 ST/退 {int(st.sum())}、黑名单 {int(bad.sum())}")
    print(f"v3 候选 {len(R)} 只 | A {(R.tier == 'A').sum()} B {(R.tier == 'B').sum()} C {(R.tier == 'C').sum()}")
    if len(R):
        print(R[["tier", "prox", "code", "name", "close", "chg", "vol20", "volratio",
                 "ret10", "ret60", "turn30", "amt_yi", "industry"]].to_string(index=False))
    print(f"\n输出：{out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
