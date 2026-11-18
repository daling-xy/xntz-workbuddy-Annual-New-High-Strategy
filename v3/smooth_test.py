# -*- coding: utf-8 -*-
"""smooth_test.py —— 探索 A/Q 差异的本质：30日“平滑度”特征能否直接用于 v3 选票"""
import pandas as pd, numpy as np, os, sys
from scipy.signal import find_peaks
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build()
tr={(x["code"], P["dates"][x["ed"]]):(x["j"],x["t"]) for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])}
feat=pd.read_csv(r"D:\about\learn\analysis\borrow_portable_features.csv",dtype={"code":str})
feat["ed"]=pd.to_datetime(feat["ed"]); feat["m"]=feat.ed.dt.to_period("M")
rows=[]
for _,r in feat.iterrows():
    j,t=tr[(r["code"], r["ed"].strftime("%Y-%m-%d"))]
    lo=max(0,t-30); c=P["C"][lo:t+1,j]; h=P["H"][lo:t+1,j]; l=P["L"][lo:t+1,j]
    if len(c)<25: continue
    dif=np.diff(c); er=abs(c[-1]-c[0])/np.sum(np.abs(dif)) if np.sum(np.abs(dif))>0 else np.nan
    dirchg=int(np.sum(np.sign(dif[1:])!=np.sign(dif[:-1]))) if len(dif)>1 else 0
    prom=0.015*float(np.mean(c))
    pk,_=find_peaks(h,prominence=prom,distance=2); trs,_=find_peaks(-l,prominence=prom,distance=2)
    rows.append(dict(ret=r.ret, status=r.status, m=r.m, er=er, dirchg=dirchg, sw=len(pk)+len(trs)))
d=pd.DataFrame(rows)
print("n",len(d))
rng=np.random.default_rng(42); allm=d.groupby("m").ret.mean()
def qstats(col,q=5):
    d["q"]=pd.qcut(d[col],q,labels=False,duplicates="drop")
    print(f"\n--- {col} 五分位（5min 平均）---")
    for k in range(q):
        s=d[d.q==k]
        print(f"  Q{k+1}: n={len(s):4d} 平均 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 大亏 {(s.ret<=-0.20).mean()*100:4.1f}%")
def boot(m,label):
    a=d[m]; am=a.groupby("m").ret.mean()
    df=pd.concat([am,allm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
    if len(diff)<5: print(f"{label}: 月数不足"); return
    bs=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)]
    lo,hi=np.percentile(bs,[5,95]); print(f"  {label:26s} {diff.mean()*100:+6.2f}pp CI90 [{lo*100:+.2f},{hi*100:+.2f}]")
qstats("er"); d["q"]=pd.qcut(d.er,5,labels=False,duplicates="drop")
boot(d.q==4,"ER最高五分位"); boot(d.q==0,"ER最低五分位")
qstats("dirchg"); d["q"]=pd.qcut(d.dirchg,5,labels=False,duplicates="drop")
boot(d.q==0,"方向变化最少"); boot(d.q==4,"方向变化最多")
qstats("sw"); d["q"]=pd.qcut(d.sw,5,labels=False,duplicates="drop")
boot(d.q==0,"转折点最少"); boot(d.q==4,"转折点最多")
print("\n--- A/Q 与简单特征的关系 ---")
print(d.groupby("status")[["er","dirchg","sw"]].mean().round(3).to_string())
d.to_csv(r"D:\about\learn\analysis\smooth_features.csv",index=False,encoding="utf-8-sig")
