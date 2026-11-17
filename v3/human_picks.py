# -*- coding: utf-8 -*-
"""human_picks.py —— 记录“人工+模型”的每日最终股票池，并回填 20 日结果
记录文件：output/track/human_picks.csv
用法：
  python v3/human_picks.py --date 2026-11-16 300760 603233 600018 601000
  python v3/human_picks.py --summary
"""
from __future__ import annotations
import argparse, glob, os, pickle, re
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache", "kline_320.pkl")
ID = os.path.join(BASE, "v3", "data", "deliver_identity_table.csv")
BOARD = os.path.join(BASE, "cache", "board_map.csv")
TRACK_DIR = os.path.join(BASE, "output", "track")
OUT = os.path.join(TRACK_DIR, "human_picks.csv")
COLS = ["rec_date","code","name","board","industry","close","amt_yi","prox","vol20","volratio",
        "ret10","ret60","turn30","z10","close_gt_ma60","ma5_gt_ma20","in_v3","in_v31",
        "fwd20","bigloss","updated"]

def load_panel():
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
            close[i, j] = b["close"]; amt[i, j] = b["amount"]; turn[i, j] = b.get("turnover", np.nan)
    px = pd.DataFrame(close, index=pd.to_datetime(all_dates))
    am = pd.DataFrame(amt, index=pd.to_datetime(all_dates))
    tn = pd.DataFrame(turn, index=pd.to_datetime(all_dates))
    ret = px.pct_change(fill_method=None)
    vol20 = (ret.rolling(20).std().values) * np.sqrt(252)
    vol_hist = pd.DataFrame(vol20).rolling(250).mean().values
    volratio = vol20 / np.where(vol_hist > 0, vol_hist, np.nan)
    prox = (px / px.rolling(250).max()).values
    ret10 = (px / px.shift(10) - 1).values
    ret60 = (px / px.shift(60) - 1).values
    turn30 = tn.rolling(30).mean().values
    vol20d = vol20 / np.sqrt(252)
    z10 = ret10 / np.where(vol20d * np.sqrt(10) > 0, vol20d * np.sqrt(10), np.nan)
    ma5 = px.rolling(5).mean().values; ma20 = px.rolling(20).mean().values; ma60 = px.rolling(60).mean().values
    return dict(codes=codes, dates=[str(d.date()) for d in px.index], close=close,
                prox=prox, vol20=vol20, volratio=volratio, ret10=ret10, ret60=ret60,
                turn30=turn30, z10=z10, ma5=ma5, ma20=ma20, ma60=ma60, amt=amt)

def _lists(rec):
    out = {"v3": set(), "v31": set()}
    for ver, pat in (("v3", os.path.join(BASE, "output", "v3", f"daily_rank_v3_{rec.replace('-', '')}.csv")),
                     ("v31", os.path.join(BASE, "output", "v31", f"daily_rank_v31_{rec.replace('-', '')}.csv"))):
        if os.path.exists(pat):
            df = pd.read_csv(pat, dtype={"code": str})
            out[ver] = set(df["code"].astype(str))
    return out

def record(rec, codes):
    P = load_panel()
    if rec not in P["dates"]:
        raise SystemExit(f"缓存中没有 {rec}（最新 {P['dates'][-1]}），先刷新行情")
    i = P["dates"].index(rec); ci = {c: j for j, c in enumerate(P["codes"])}
    idn = pd.read_csv(ID, dtype={"code": str}).set_index("code")
    bm = pd.read_csv(BOARD, dtype={"code": str}).drop_duplicates("code").set_index("code")["board"]
    lists = _lists(rec)
    old = pd.read_csv(OUT, dtype={"code": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    rows = []
    for c in codes:
        c = str(c).zfill(6)
        if c not in ci:
            print(f"  [跳过] {c} 不在行情缓存中")
            continue
        j = ci[c]
        rows.append(dict(rec_date=rec, code=c,
                         name=idn.loc[c, "name"] if c in idn.index else "",
                         board=bm.get(c, ""), industry=idn.loc[c, "主行业(赛道)"] if c in idn.index else "",
                         close=round(float(P["close"][i, j]), 2),
                         amt_yi=round(float(P["amt"][i, j]) / 1e8, 2),
                         prox=round(float(P["prox"][i, j]), 4),
                         vol20=round(float(P["vol20"][i, j]) * 100, 1),
                         volratio=round(float(P["volratio"][i, j]), 2),
                         ret10=round(float(P["ret10"][i, j]) * 100, 1),
                         ret60=round(float(P["ret60"][i, j]) * 100, 1),
                         turn30=round(float(P["turn30"][i, j]), 2),
                         z10=round(float(P["z10"][i, j]), 2),
                         close_gt_ma60=bool(P["close"][i, j] > P["ma60"][i, j]),
                         ma5_gt_ma20=bool(P["ma5"][i, j] > P["ma20"][i, j]),
                         in_v3=int(c in lists["v3"]), in_v31=int(c in lists["v31"]),
                         fwd20=np.nan, bigloss=np.nan,
                         updated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
    new = pd.DataFrame(rows, columns=COLS)
    frames = [f for f in (old, new) if len(f)]
    allr = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["rec_date", "code"], keep="last")
    # 回填 20 日结果
    for k, r in allr.iterrows():
        if r["rec_date"] not in P["dates"]: continue
        i2 = P["dates"].index(r["rec_date"]); j2 = ci.get(str(r["code"]))
        if j2 is None or i2 + 20 >= len(P["dates"]): continue
        c0, c1 = P["close"][i2, j2], P["close"][i2 + 20, j2]
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0:
            allr.at[k, "fwd20"] = round(float(c1 / c0 - 1) * 100, 2)
            allr.at[k, "bigloss"] = int(c1 / c0 - 1 <= -0.20)
    allr.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[human] 记录 {len(new)} 只（{rec}）-> {OUT}")
    return new

def summary():
    if not os.path.exists(OUT):
        print("尚未记录人工股票池"); return
    df = pd.read_csv(OUT, dtype={"code": str})
    print("\n=== 人工最终池 vs 模型名单（fwd20 毛收益，未扣成本）===")
    done = df[df["fwd20"].notna()]
    print(f"总记录 {len(df)} 条，已出 20 日结果 {len(done)} 条")
    if len(done):
        print(f"整体: 平均fwd20={done['fwd20'].mean():+.2f}% 大亏率={(done['bigloss']==1).mean()*100:.1f}%")
        for name, mask in [("在v3名单内", done["in_v3"] == 1), ("在v3.1名单内", done["in_v31"] == 1),
                           ("不在v3也不在v3.1", (done["in_v3"] == 0) & (done["in_v31"] == 0))]:
            g = done[mask]
            if len(g):
                print(f"  {name:14s} n={len(g):3d} 平均fwd20={g['fwd20'].mean():+.2f}% 大亏率={(g['bigloss']==1).mean()*100:.1f}%")
    print("\n最近记录：")
    print(df.sort_values(["rec_date"]).tail(20).to_string(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="记录日期 YYYY-MM-DD")
    ap.add_argument("codes", nargs="*", help="股票代码（可带名字，只取代码）")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(); return
    if not a.date or not a.codes:
        ap.error("需要 --date 和至少一个股票代码")
    import re
    codes = [re.sub(r"\D", "", x)[-6:] for x in a.codes]
    record(a.date, codes)
    summary()

if __name__ == "__main__":
    main()
