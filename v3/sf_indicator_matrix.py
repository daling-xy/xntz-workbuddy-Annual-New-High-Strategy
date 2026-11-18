# -*- coding: utf-8 -*-
"""sf_indicator_matrix.py —— 结构优先交易上的单指标穷举（2023-2026，日线近似）
回答：volratio 还是 vol20？prox 要不要严？turn30/ret10 是否真有效？逐个阈值测，含月度块 CI 与两段 OOS。
"""
import numpy as np, pandas as pd
x=pd.read_csv(r"D:\about\learn\analysis\sf_ablation.csv")
x["m"]=pd.PeriodIndex(x["m"],freq="M")
x["y"]=x.m.dt.year
rng=np.random.default_rng(42)
base_month=x.groupby("m").ret.mean()
def evalv(lab,mask, half=None):
    s=x[mask] if half is None else x[mask&(x.y>=half[0])&(x.y<=half[1])]
    b=x if half is None else x[(x.y>=half[0])&(x.y<=half[1])]
    if len(s)==0: print(f"{lab:34s} n=0"); return
    sm=s.groupby("m").ret.mean(); bm=b.groupby("m").ret.mean()
    df=pd.concat([sm,bm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
    bs=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)] if len(diff)>=5 else None
    lo,hi=(np.percentile(bs,[5,95]) if bs is not None else (np.nan,np.nan))
    print(f"{lab:34s} n={len(s):5d} 均值 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 月均 {sm.mean()*100:6.2f}% | 差 {diff.mean()*100:+5.2f}pp CI[{lo*100:+.2f},{hi*100:+.2f}]")
print("=== 基准 ===")
evalv("SF 全部", np.ones(len(x),bool))
print("\n=== volratio 单指标 ===")
for th in [1.0,1.1,1.2,1.5]: evalv(f"volratio<={th}", np.round(x.vr,2)<=th)
print("\n=== vol20 单指标 ===")
for th in [0.30,0.35,0.40,0.45]: evalv(f"vol20<={int(th*100)}%", x.vol20<=th)
print("\n=== volratio+vol20 组合 ===")
evalv("ratio<=1.0 仅", np.round(x.vr,2)<=1.0)
evalv("vol20<=35 仅", x.vol20<=0.35)
evalv("两者都要", (np.round(x.vr,2)<=1.0)&(x.vol20<=0.35))
print("\n=== turn30 ===")
for th in [3,5,8,10]: evalv(f"turn30<={th}", x.turn30<=th)
print("\n=== ret10 ===")
for th in [0.05,0.08,0.10,0.15]: evalv(f"ret10<={int(th*100)}%", x.ret10<=th)
print("\n=== prox ===")
for th in [0.80,0.85,0.90,0.95]: evalv(f"prox>={th}", x.prox>=th)
print("\n=== ret60 ===")
for lab,m in [("ret60>=-10%",x.ret60>=-0.10),("ret60>=0",x.ret60>=0),("ret60>=10%",x.ret60>=0.10)]: evalv(lab,m)
print("\n=== 两段 OOS（2023-24 / 2025-26）关键变体 ===")
vol=(np.round(x.vr,2)<=1.0)&(x.vol20<=0.35)
for half in [(2023,2024),(2025,2026)]:
    print(f"-- {half[0]}-{half[1]} --")
    evalv("SF 全部", np.ones(len(x),bool), half)
    evalv("SF+vol", vol, half)
    evalv("SF+vol+t30+r10", vol&(x.turn30<=5)&(x.ret10<=0.08), half)
    evalv("SF+all v3 gates", vol&(x.turn30<=5)&(x.ret10<=0.08)&(x.prox>=0.90)&(x.ret60>=0), half)
