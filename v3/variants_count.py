# -*- coding: utf-8 -*-
"""variants_count.py —— 预估 prox0.85 / ret10 5% 变体的交易数与需要新校准的笔数"""
import numpy as np, pandas as pd, os, sys
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build(); dates=np.array(P["dates"])
cal=pd.read_csv(r"D:\about\learn\analysis\min5_calib_trades_all.csv",dtype={"code":str})
calk=set(zip(cal.code, cal.ed))
def make(variant):
    nd=P["nd"]; i0=int(np.searchsorted(dates,"2019-01-01"))
    base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["prox"]>=0.90)&(P["ret10"]<=0.08)&(P["turn30"]<=5)
    base&=(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    if variant=="V1": base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["prox"]>=0.85)&(P["ret10"]<=0.08)&(P["turn30"]<=5)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    if variant=="V2": base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["prox"]>=0.90)&(P["ret10"]<=0.05)&(P["turn30"]<=5)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    if variant=="V3": base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["prox"]>=0.85)&(P["ret10"]<=0.05)&(P["turn30"]<=5)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    base[:i0,:]=False
    H=P["H"]; O=P["O"]; hi30=pd.DataFrame(P["C"]).rolling(30).max().values
    ii,jj=np.where(base); trades=[]
    for k in range(len(ii)):
        t=ii[k]; j=jj[k]; top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        trig=top*1.07; cap=top*1.15; entry=None; ed=None
        for d in (t+1,t+2):
            if d>=nd: break
            if np.isfinite(H[d,j]) and H[d,j]>=trig and np.isfinite(O[d,j]) and O[d,j]<=cap:
                entry=max(float(O[d,j]),trig); ed=d; break
        if entry is None: continue
        trades.append((P["codes"][j], dates[ed]))
    return trades
for v in ["V0","V1","V2","V3"]:
    tr=make(v); keys=set(tr)
    new=keys-calk
    print(f"{v}: trades={len(tr)} unique={len(keys)} 需新5min校准={len(new)} (2019:{sum(1 for k in new if k[1][:4]=='2019')} 2020-22:{sum(1 for k in new if '2020'<=k[1][:4]<='2022')} 2023-26:{sum(1 for k in new if k[1][:4]>='2023')})")
