# -*- coding: utf-8 -*-
"""sf5min_calib.py —— 结构优先合并版(SF+vol+turn30+ret10) 的 5min 校准（抽样）"""
import os, sys, time, random, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"
N=int(sys.argv[1]) if len(sys.argv)>1 else 300
P=M.build(); dates=np.array(P["dates"]); jmap={c:i for i,c in enumerate(P["codes"])}
d=pd.read_csv(os.path.join(LEARN,"analysis","structure_first_trades_2023_ret.csv"),dtype={"code":str})
d["t"]=pd.to_datetime(d["t"]); d["ed"]=pd.to_datetime(d["ed"])
rows=[]
for _,r in d.iterrows():
    j=jmap.get(r["code"]); t=int(np.searchsorted(dates, r["t"].strftime("%Y-%m-%d")))
    vr=P["vr"][t,j]; vol20=P["vol20"][t,j]; ret10=P["ret10"][t,j]; turn30=P["turn30"][t,j]
    if not (np.isfinite(vr) and np.round(vr,2)<=1.0 and vol20<=0.35 and ret10<=0.08 and turn30<=5): continue
    rows.append(dict(code=r["code"], j=j, t=t, ed=int(np.searchsorted(dates, r["ed"].strftime("%Y-%m-%d"))), top=r["top"], entry=float(r["entry"])))
tr=[x for x in rows if M.bs_sym(x["code"])]
print("合并版候选交易:",len(tr),flush=True)
random.seed(11); random.shuffle(tr); sample=tr[:N]
import baostock as bs
bs.login(); out=[]
for k,x in enumerate(sample):
    ed=x["ed"]; j=x["j"]
    d0=dates[ed-1]; d1=dates[min(ed+80, P["nd"]-1)]
    df=M.get5(x["code"], d0, d1)
    if df is None or len(df)==0: continue
    rm=M.sim_minute(P, x, df, float(P["C"][ed-1,j]))
    if rm is None: continue
    rd=M.sim_daily(P, x)
    out.append(dict(code=x["code"], ed=dates[ed], ret_min=rm["ret"], ret_daily=rd["ret"], half=rm["half"]))
    if (k+1)%50==0: print(f"  {k+1}/{len(sample)}",flush=True)
bs.logout()
r=pd.DataFrame(out)
print(f"完成 {len(r)} 笔")
if len(r):
    print(f"5min: 平均 {r.ret_min.mean()*100:.2f}% 中位 {r.ret_min.median()*100:.2f}% 胜率 {(r.ret_min>0).mean()*100:.1f}% 半仓 {r.half.mean()*100:.1f}%")
    print(f"日线: 平均 {r.ret_daily.mean()*100:.2f}% 中位 {r.ret_daily.median()*100:.2f}% 胜率 {(r.ret_daily>0).mean()*100:.1f}%")
r.to_csv(os.path.join(LEARN,"analysis","sf5min_sample.csv"),index=False,encoding="utf-8-sig")
