# -*- coding: utf-8 -*-
"""box_zone_filter.py —— 借鉴便携版「缠论中枢」箱体口径，检验它对 v3 交易的过滤价值
中枢定义：最近 3 段（4 个转折点）的重叠区；ZG=min(各段高)，ZD=max(各段低)；overlap=ZD<=ZG
偏离闸：现价 ∈ [ZD×0.92, ZG×1.08]（引擎：>ZG+8% 或 <ZD-8% 剔除）
样本：v3(V1) 已校准 5min 收益的 1,205 笔；特征用 kline_long2 OHLC 在选股日 t 计算
"""
import os, sys, numpy as np, pandas as pd
from scipy.signal import find_peaks
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"
P=M.build()
trades=[x for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])]
meta={}
for x in trades:
    meta[(x["code"], P["dates"][x["ed"]])]=(x["j"], x["t"], x["top"])

def swings(h, l, closes, prom_pct=0.015, dist=2):
    if len(closes)<60: return None
    prom=prom_pct*float(np.mean(closes))
    pk,_=find_peaks(h, prominence=prom, distance=dist)
    tr,_=find_peaks(-l, prominence=prom, distance=dist)
    pts=[]
    for i in pk:
        pts.append((int(i),"peak",float(h[i])))
    for i in tr:
        pts.append((int(i),"trough",float(l[i])))
    pts.sort()
    out=[]; last=None
    for p in pts:
        if last!=p[1]:
            out.append(p); last=p[1]
    if len(out)<4: return None
    if out[0][1]!="peak": out=out[1:]
    if out[-1][1]!="trough": out=out[:-1]
    if len(out)<4 or len(out)%2!=0: return None
    return out

def zone_at(P, j, t, win=250):
    lo=max(0,t-win+1)
    h=P["H"][lo:t+1,j]; l=P["L"][lo:t+1,j]; c=P["C"][lo:t+1,j]
    sw=swings(h,l,c)
    if sw is None: return dict(ok=False)
    pts=sw[-4:]
    highs=[]; lows=[]
    for a,b in zip(pts[:-1],pts[1:]):
        highs.append(max(a[2],b[2])); lows.append(min(a[2],b[2]))
    zg=min(highs); zd=max(lows); overlap=zd<=zg
    width=(zg-zd)/zd if (zd>0 and overlap) else None
    close=float(P["C"][t,j])
    return dict(ok=True, zg=zg, zd=zd, width=width, overlap=bool(overlap),
                dev_zg=(close/zg-1) if zg>0 else None, dev_zd=(close/zd-1) if zd>0 else None,
                ret10=float(P["C"][t,j]/P["C"][t-10,j]-1) if t>=10 else None,
                ret30=float(P["C"][t,j]/P["C"][t-30,j]-1) if t>=30 else None)

cal=pd.read_csv(os.path.join(LEARN,"analysis","min5_calib_trades_all.csv"),dtype={"code":str})
rows=[]
for _,r in cal.iterrows():
    key=(r["code"], pd.Timestamp(r["ed"]).strftime("%Y-%m-%d"))
    if key not in meta: continue
    j,t,top=meta[key]
    z=zone_at(P,j,t)
    if not z.get("ok"): z=dict(ok=False)
    rows.append(dict(code=r["code"], ed=key[1], ret=r["ret_min"], half=r["half_min"],
                     zone_ok=z.get("ok"), overlap=z.get("overlap"), width=z.get("width"),
                     dev_zg=z.get("dev_zg"), dev_zd=z.get("dev_zd"), ret10=z.get("ret10"), ret30=z.get("ret30")))
d=pd.DataFrame(rows)
print(f"匹配 {len(d)} 笔（有中枢特征）")
d.to_csv(os.path.join(LEARN,"analysis","box_zone_features.csv"),index=False,encoding="utf-8-sig")

def stat(lab, m):
    s=d[m]
    if len(s)==0: print(f"{lab:34s} n=0"); return
    print(f"{lab:34s} n={len(s):4d} 平均 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 大亏 {(s.ret<=-0.20).mean()*100:4.1f}% 半仓 {s.half.mean()*100:5.1f}%")

print("\n=== 中枢相关过滤（5min 口径收益）===")
stat("全部", d.index==d.index)
stat("有有效中枢(overlap)", d.overlap==True)
stat("无有效中枢(单边/转折不足)", d.overlap!=True)
stat("中枢宽度<=10%", (d.width.notna())&(d.width<=0.10))
stat("中枢宽度<=20%", (d.width.notna())&(d.width<=0.20))
stat("偏离在 [ZD-8%, ZG+8%]", (d.dev_zg.notna())&(d.dev_zg<=0.08)&(d.dev_zd.notna())&(d.dev_zd>=-0.08))
stat("近10日跌幅>=-7%", (d.ret10.notna())&(d.ret10>=-0.07))
stat("中枢+宽度<=20%+偏离内", (d.overlap==True)&(d.width.notna())&(d.width<=0.20)&(d.dev_zg<=0.08)&(d.dev_zd>=-0.08))
stat("中枢+宽度<=20%+偏离内+近10日>=-7%", (d.overlap==True)&(d.width.notna())&(d.width<=0.20)&(d.dev_zg<=0.08)&(d.dev_zd>=-0.08)&(d.ret10>=-0.07))
stat("仅保留无中枢(趋势/单边)", (d.overlap!=True))
