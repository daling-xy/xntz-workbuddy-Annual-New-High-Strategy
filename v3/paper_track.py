# -*- coding: utf-8 -*-
"""paper_track.py —— v3 / v3.1 平行跟踪记录与结果回填
用法：python v3/paper_track.py   （run_daily 末尾也会自动调用）
记录文件：output/track/picks.csv（每行：日期/版本/档位/prox/代码/名称/当日收盘/20日收益）
"""
from __future__ import annotations
import glob
import os
import pickle
import re
import datetime as dt

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache", "kline_320.pkl")
TRACK_DIR = os.path.join(BASE, "output", "track")
TRACK = os.path.join(TRACK_DIR, "picks.csv")
COLS = ["rec_date", "version", "tier", "prox", "code", "name", "close", "fwd20", "bigloss", "updated"]


def _panel():
    kl = pickle.load(open(CACHE, "rb"))
    codes = sorted(kl.keys())
    all_dates = sorted({b["date"] for v in kl.values() for b in v})
    di = {d: i for i, d in enumerate(all_dates)}
    px = np.full((len(all_dates), len(codes)), np.nan, np.float32)
    for j, c in enumerate(codes):
        for b in kl[c]:
            px[di[b["date"]], j] = b["close"]
    return all_dates, codes, px


def _scan():
    rows = []
    for pat, ver in ((os.path.join(BASE, "output", "v3", "daily_rank_v3_*.csv"), "v3"),
                     (os.path.join(BASE, "output", "v31", "daily_rank_v31_*.csv"), "v3.1")):
        for f in glob.glob(pat):
            m = re.search(r"(\d{8})", os.path.basename(f))
            if not m:
                continue
            rec = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
            try:
                df = pd.read_csv(f, dtype={"code": str})
            except Exception:
                continue
            for _, r in df.iterrows():
                rows.append(dict(rec_date=rec, version=ver, tier=r.get("tier", ""),
                                 prox=r.get("prox", np.nan), code=str(r["code"]),
                                 name=r.get("name", ""), close=r.get("close", np.nan),
                                 fwd20=np.nan, bigloss=np.nan,
                                 updated=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return pd.DataFrame(rows, columns=COLS)


def update():
    os.makedirs(TRACK_DIR, exist_ok=True)
    old = pd.read_csv(TRACK, dtype={"code": str}) if os.path.exists(TRACK) else pd.DataFrame(columns=COLS)
    new = _scan()
    frames = [f for f in (old, new) if f is not None and len(f)]
    allr = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    allr = allr.drop_duplicates(subset=["rec_date", "version", "code"], keep="first")
    dates, codes, px = _panel()
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: j for j, c in enumerate(codes)}
    for k, r in allr.iterrows():
        i = di.get(r["rec_date"]); j = ci.get(str(r["code"]))
        if i is None or j is None or i + 20 >= len(dates):
            continue
        c0 = px[i, j]; c1 = px[i + 20, j]
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
            allr.at[k, "fwd20"] = round(float(c1 / c0 - 1) * 100, 2)
            allr.at[k, "bigloss"] = int(c1 / c0 - 1 <= -0.20)
            allr.at[k, "updated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    allr.to_csv(TRACK, index=False, encoding="utf-8-sig")
    print(f"[paper-track] 记录 {len(allr)} 条 -> {TRACK}")


def summary():
    if not os.path.exists(TRACK):
        return
    df = pd.read_csv(TRACK, dtype={"code": str})
    if df.empty:
        return
    print("\n=== 平行跟踪汇总（fwd20 为毛收益，未扣成本）===")
    for ver, g in df.groupby("version"):
        done = g[g["fwd20"].notna()]
        daily = g.groupby("rec_date").size()
        avg = f"{done['fwd20'].mean():+6.2f}%" if len(done) else "     —"
        bl = f"{(done['bigloss'] == 1).mean() * 100:.1f}%" if len(done) else "  —"
        print(f"{ver:5s} 名单数={len(g):4d}  交易日={len(daily):3d}  日均={daily.mean():5.1f}  "
              f"已有20日结果={len(done):4d}  平均fwd20={avg}  大亏率={bl}")


if __name__ == "__main__":
    update()
    summary()
