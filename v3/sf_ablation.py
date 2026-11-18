# -*- coding: utf-8 -*-
"""sf_ablation.py —— 结构优先交易上逐条加 v3 门，看哪条有价值"""
import numpy as np, pandas as pd, os, sys
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build(); dates=np.array(P["dates"]); jmap={c:i for i,c in enumerate(P["codes"])}
d=pd.read_csv(r"D:\about\learn\analysis\structure_first_trades_2023_ret.csv",dtype={"code":str})
d["t"]=pd.to_datetime(d["t"]); d["ed"]=pd.to_datetime(d["ed"])
rows=[]
for _,r in d.iterrows():
    j=jmap.get(r["code"]); t=int(np.searchsorted(dates, r["t"].strftime("%Y-%m-%d")))
    rows.append(dict(ret=r["ret_d"], m=r["ed"].to_period("M"),
        vol20=P["vol20"][t,j], vr=P["vr"][t,j], prox=P["prox"][t,j],
        ret10=P["ret10"][t,j], ret60=P["ret60"][t,j], turn30=P["turn30"][t,j]))
x=pd.DataFrame(rows)
gates={"vol门": (np.round(x.vr,2)<=1.0)&(x.vol20<=0.35),
       "prox>=0.90": x.prox>=0.90,
       "ret10<=8%": x.ret10<=0.08,
       "turn30<=5": x.turn30<=5,
       "ret60>=0": x.ret60>=0}
allgate=np.ones(len(x),bool)
for k in gates: allgate&=gates[k].values
print("结构优先交易数:",len(x))
def stat(lab,m):
    s=x[m]
    if len(s)==0: print(f"{lab:28s} n=0"); return
    dm=s.groupby(s.m).ret.mean(); 
    print(f"{lab:28s} n={len(s):5d} 单笔均值 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% | 月均 {dm.mean()*100:6.2f}% 月胜率 {(dm>0).mean()*100:5.1f}%")
print("\n=== 逐条加门（日线近似收益）===")
stat("SF 全部", x.ret.notna())
for k,m in gates.items(): stat("SF + "+k, m)
stat("SF + 全部v3门", allgate)
stat("SF + vol门+prox", gates["vol门"]&gates["prox>=0.90"])
stat("SF + vol门+prox+ret60", gates["vol门"]&gates["prox>=0.90"]&gates["ret60>=0"])
print("\n=== 反例：去掉某条（全部门去掉一条）===")
for k in gates:
    m=allgate.copy()
    for kk in gates:
        if kk!=k: m&=gates[kk].values
    stat("全门 去掉"+k, m)
x.to_csv(r"D:\about\learn\analysis\sf_ablation.csv",index=False,encoding="utf-8-sig")
