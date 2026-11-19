# -*- coding: utf-8 -*-
import os, sys, time, numpy as np, pandas as pd
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build(); dates=np.array(P["dates"]); idx={d:i for i,d in enumerate(dates)}
need=[]
for f in ["buys_entry_fixed.csv","buys_first_fixed.csv"]:
    p=os.path.join(r"D:\about\learn\analysis",f)
    if not os.path.exists(p): continue
    d=pd.read_csv(p,dtype={"code":str}); d["entry_date"]=pd.to_datetime(d.entry_date)
    for _,r in d.iterrows():
        s=r.entry_date.strftime("%Y-%m-%d")
        if s not in idx: continue
        i=idx[s]; sd=dates[i-1]; ed=dates[min(i+120,P["nd"]-1)]
        fp=os.path.join(M.C5, f"{r.code}_{sd}_{ed}.csv")
        if not os.path.exists(fp): need.append((r.code,sd,ed))
need=list(dict.fromkeys(need))
print("需要补 5min 窗口:",len(need),flush=True)
import baostock as bs; bs.login(); ok=fail=0; t0=time.time()
for k,(code,sd,ed) in enumerate(need):
    df=M.get5(code,sd,ed)
    if df is not None and len(df): ok+=1
    else: fail+=1
    if (k+1)%20==0: print(f"  {k+1}/{len(need)} ok={ok} fail={fail} [{time.time()-t0:.0f}s]",flush=True)
bs.logout(); print("done ok",ok,"fail",fail,f"[{time.time()-t0:.0f}s]")
