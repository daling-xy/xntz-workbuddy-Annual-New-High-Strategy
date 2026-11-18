# -*- coding: utf-8 -*-
"""min5_sell_sens.py —— 5min 口径下卖出规则敏感性（止损幅度 / 半仓触发方式 / ③a量能阈值）
复用 data/min_calib 缓存；V1(v3现状) 全部 1,324 笔交易。
"""
import os, sys, time
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"

def prep(df, ed_date, nd_set=None):
    days=sorted(df["date"].unique())
    g={}
    for d in days:
        b=df[df.date==d].sort_values("hhmm")
        if len(b)==0: continue
        g[d]=dict(hhmm=b["hhmm"].to_numpy(), o=b["open"].to_numpy(), h=b["high"].to_numpy(),
                  l=b["low"].to_numpy(), c=b["close"].to_numpy(), v=b["volume"].to_numpy(),
                  prevc=None, prevv=None)
    return days, g

def sim(days, g, ed_date, top, s=0.87, half_mode="both", vol70=0.7):
    ds=[d for d in days if d>=ed_date]
    if not ds: return None
    # 找触发
    entry=None; ei=None; eday=None
    for dd in range(len(ds)):
        d=ds[dd]
        if dd>1: break
        b=g[d]; trig=1.07
        m=(b["h"]>=1.07*top_g) if False else None
    return None

def simulate(days, g, ed_date, top, s=0.87, half_mode="both", vol70=0.7):
    """top 为 5min 不复权口径的箱顶（调用方已乘 factor）；entry 与 top 同口径"""
    ds=[d for d in days if d>=ed_date]
    if len(ds)<1: return None
    trig=1.07*top; cap=1.15*top
    entry=None; eday=None; ehhmm=None
    for k in range(min(2,len(ds))):
        d=ds[k]; b=g[d]
        ok=(b["h"]>=trig)&(b["l"]<=cap)
        if ok.any():
            i=int(np.argmax(ok))
            entry=float(min(max(b["o"][i],trig),cap)); eday=d; ehhmm=int(b["hhmm"][i]); break
    if entry is None: return None
    half=None; half_done=False; xp=None; xday=None
    for k,d in enumerate(ds):
        if d<eday: continue
        b=g[d]
        # 前一日收盘/成交量
        if k==0:
            continue
        pd_=ds[k-1]; pb=g[pd_]; prevc=float(pb["c"][-1]); prevv=float(pb["v"].sum())
        # ②a 盘中涨幅>=9%
        if half_mode in ("both","a"):
            start=ehhmm if d==eday else 0
            for i in range(len(b["hhmm"])):
                if d==eday and b["hhmm"][i]<=start: continue
                prof=float(b["c"][i])/entry-1
                if prof>0.30 and float(b["c"][i])/prevc-1>=0.09:
                    if not half_done: half=float(b["c"][i]); half_done=True
                    else: xp=float(b["c"][i]); xday=d; break
            if xp is not None: break
        # ③a 10:30 前量>=vol70*前日，且跌幅>7%
        m=b["hhmm"]<=1030
        if m.any():
            cum=float(b["v"][m].sum()); px=float(b["c"][m][-1])
            if prevv>0 and cum>=vol70*prevv and px/prevc-1<-0.07:
                xp=px; xday=d; break
        # ③b 14:30 累计量>=前日，且跌幅>7%
        m=b["hhmm"]<=1430
        if m.any():
            cum=float(b["v"][m].sum()); px=float(b["c"][m][-1])
            if prevv>0 and cum>=prevv and px/prevc-1<-0.07:
                xp=px; xday=d; break
            # ①止损 14:30
            if px<entry*s:
                xp=px; xday=d; break
        # ②b 收盘口径：盈利>30% 且连续两日涨幅<=3% → 14:45
        if half_mode in ("both","b"):
            m=b["hhmm"]<=1445
            if m.any():
                px1445=float(b["c"][m][-1]); prof=px1445/entry-1
                if prof>0.30 and float(b["c"][-1])/prevc-1<=0.03:
                    pp=g[ds[k-1]]
                    if float(pp["c"][-1])/prevc-1<=0.03:
                        if not half_done: half=px1445; half_done=True
                        else: xp=px1445; xday=d; break
    if xp is None:
        xday=ds[-1]; xp=float(g[ds[-1]]["c"][-1])
    r=(0.5*half+0.5*xp)/entry-1 if half is not None else xp/entry-1
    return dict(ret=float(r), half=half is not None, xday=xday, entry=entry)

def main():
    t0=time.time(); P=M.build(); trades=[x for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])]
    prepared=[]
    for x in trades:
        ed=x["ed"]; j=x["j"]
        d0=P["dates"][ed-1]; d1=P["dates"][min(ed+80,P["nd"]-1)]
        df=M.get5(x["code"],d0,d1)
        if df is None or len(df)==0: continue
        days,g=prep(df,P["dates"][ed]); prepared.append(dict(x=x,days=days,g=g,P=P))
    print(f"加载 {len(prepared)} 笔 [{time.time()-t0:.0f}s]",flush=True)
    variants=[("基准 s=0.87 ②both",dict(s=0.87,half_mode="both",vol70=0.7)),
              ("止损 -10% s=0.90",dict(s=0.90,half_mode="both",vol70=0.7)),
              ("止损 -15% s=0.85",dict(s=0.85,half_mode="both",vol70=0.7)),
              ("只保留②a",dict(s=0.87,half_mode="a",vol70=0.7)),
              ("只保留②b",dict(s=0.87,half_mode="b",vol70=0.7)),
              ("不要半仓",dict(s=0.87,half_mode="none",vol70=0.7)),
              ("③a量能阈值100%",dict(s=0.87,half_mode="both",vol70=1.0))]
    rows=[]
    for lab,kw in variants:
        rs=[]; hs=[]; held=[]
        for it in prepared:
            x=it["x"]; j=x["j"]; ed=x["ed"]; top=x["top"]
            days=it["days"]; g=it["g"]
            # factor: 前一日 5min 收盘 / qfq 收盘
            pre=[d for d in days if d<P["dates"][ed]]
            if not pre: continue
            pclose=float(g[pre[-1]]["c"][-1]); qfq_prev=float(P["C"][ed-1,j])
            if qfq_prev<=0: continue
            factor=pclose/qfq_prev
            r=simulate(days,g,P["dates"][ed],top*factor,**kw)
            if r is None: continue
            rs.append(r["ret"]); hs.append(r["half"]); held.append(days.index(r["xday"])-days.index(pre[-1]) if r["xday"] in days else np.nan)
        a=np.array(rs); h=np.array(hs)
        rows.append(dict(variant=lab,n=len(a),mean=a.mean()*100,median=np.median(a)*100,win=(a>0).mean()*100,big=(a<=-0.20).mean()*100,half=h.mean()*100,hold=np.nanmean(held)))
        print(f"  {lab:22s} n={len(a):4d} 平均 {a.mean()*100:6.2f}% 中位 {np.median(a)*100:6.2f}% 胜率 {(a>0).mean()*100:5.1f}% 大亏 {(a<=-0.20).mean()*100:4.1f}% 半仓 {h.mean()*100:5.1f}% | [{time.time()-t0:.0f}s]",flush=True)
    R=pd.DataFrame(rows)
    print("\n=== 汇总 ===")
    print(R.round(2).to_string(index=False))
    R.to_csv(os.path.join(LEARN,"analysis","min5_sell_sensitivity.csv"),index=False,encoding="utf-8-sig")
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__":
    main()
