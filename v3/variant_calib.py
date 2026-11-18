# -*- coding: utf-8 -*-
"""variant_calib.py —— 预注册变体：prox0.85 / ret10 5% / 两者，补 5min 校准（去重 key=code+触发日+箱顶）
买卖点与 v3 完全一致；只改选股条件。
"""
import os, sys, time, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"

def masks(P):
    base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["ret10"]<=0.08)&(P["turn30"]<=5)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    return {
      "V0": base&(P["prox"]>=0.90),
      "V1": base&(P["prox"]>=0.85),
      "V2": base&(P["prox"]>=0.90)&(P["ret10"]<=0.05),
      "V3": base&(P["prox"]>=0.85)&(P["ret10"]<=0.05),
    }

def build(P, mask):
    dates=np.array(P["dates"]); nd=P["nd"]
    i0=int(np.searchsorted(dates,"2019-01-01")); mask=mask.copy(); mask[:i0,:]=False
    H=P["H"]; O=P["O"]; C=P["C"]; hi30=pd.DataFrame(C).rolling(30).max().values
    ii,jj=np.where(mask); out={}
    for k in range(len(ii)):
        t=ii[k]; j=jj[k]; top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        trig=top*1.07; cap=top*1.15; entry=None; ed=None
        for d in (t+1,t+2):
            if d>=nd: break
            if np.isfinite(H[d,j]) and H[d,j]>=trig and np.isfinite(O[d,j]) and O[d,j]<=cap:
                entry=max(float(O[d,j]),trig); ed=d; break
        if entry is None: continue
        key=(P["codes"][j], dates[ed], round(float(top),4))
        if key in out: continue
        out[key]=dict(code=P["codes"][j], j=j, t=t, ed=ed, top=float(top), entry=entry)
    return out

def main():
    t0=time.time(); P=M.build()
    masks_=masks(P)
    D={v:build(P,m) for v,m in masks_.items()}
    for v in D: print(v,"trades(dedup):",len(D[v]),flush=True)
    cal=pd.read_csv(os.path.join(LEARN,"analysis","min5_calib_trades_all.csv"),dtype={"code":str})
    existing=set()
    for _,r in cal.iterrows():
        existing.add((r["code"], pd.Timestamp(r["ed"]).strftime("%Y-%m-%d"), round(float(r["top"]),4)))
    need={}
    for v in ["V1","V2","V3"]:
        for k,x in D[v].items():
            if k not in existing: need[k]=x
    print("需要新校准:",len(need),flush=True)
    if not need: return
    import baostock as bs
    bs.login(); out=[]
    for idx,(k,x) in enumerate(need.items()):
        ed=x["ed"]; j=x["j"]
        d0=P["dates"][ed-1]; d1=P["dates"][min(ed+80,P["nd"]-1)]
        df=M.get5(x["code"], d0, d1)
        if df is None or len(df)==0: continue
        rm=M.sim_minute(P, x, df, float(P["C"][ed-1,j]))
        if rm is None: continue
        out.append(dict(code=k[0], ed=k[1], top=k[2], ret_min=rm["ret"], half=rm["half"]))
        if (idx+1)%50==0: print(f"  {idx+1}/{len(need)} [{time.time()-t0:.0f}s]",flush=True)
    bs.logout()
    r=pd.DataFrame(out); r.to_csv(os.path.join(LEARN,"analysis","variant5min_new.csv"),index=False,encoding="utf-8-sig")
    print("完成",len(r),"输出 analysis/variant5min_new.csv")

if __name__=="__main__": main()
