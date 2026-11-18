# -*- coding: utf-8 -*-
"""variant_eval.py —— prox0.85 / ret10 5% 事件级 5min 对比（2021+，去重 code+ed+top）"""
import numpy as np, pandas as pd, os, sys
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
import variant_calib as VC
LEARN=r"D:\about\learn"
P=M.build(); dates=np.array(P["dates"])
base_cal=pd.read_csv(os.path.join(LEARN,"analysis","min5_calib_trades_all.csv"),dtype={"code":str})
new_cal=pd.read_csv(os.path.join(LEARN,"analysis","variant5min_new.csv"),dtype={"code":str})
def key(r): return (r["code"], pd.Timestamp(r["ed"]).strftime("%Y-%m-%d"), round(float(r["top"]),4))
cal={}
for _,r in base_cal.iterrows(): cal.setdefault(key(r), r["ret_min"])
for _,r in new_cal.iterrows(): cal.setdefault(key(r), r["ret_min"])
ms=VC.masks(P); res={}
for v,m in ms.items():
    tr=VC.build(P,m); rows=[]
    for k,x in tr.items():
        ed=dates[x["ed"]]
        if ed<"2021-01-01": continue
        r=cal.get(k)
        rows.append(dict(code=k[0], ed=ed, top=k[2], ret=r))
    d=pd.DataFrame(rows); res[v]=d
    dd=d.dropna(subset=["ret"]).copy(); dd["m"]=pd.to_datetime(dd.ed).dt.to_period("M"); dd["y"]=pd.to_datetime(dd.ed).dt.year
    print(f"{v}: 交易 {len(d)} 笔，5min校准 {len(dd)} 笔")
    if len(dd):
        print(f"   单笔 均值 {dd.ret.mean()*100:6.2f}% 中位 {dd.ret.median()*100:6.2f}% 胜率 {(dd.ret>0).mean()*100:5.1f}% 大亏 {(dd.ret<=-0.20).mean()*100:4.1f}%")
        dm=dd.groupby("m").ret.mean(); print(f"   月度(日等权) 均值 {dm.mean()*100:6.2f}% 月胜率 {(dm>0).mean()*100:5.1f}% 月数 {len(dm)}")
base=res["V0"].dropna(subset=["ret"]).copy(); base["m"]=pd.to_datetime(base.ed).dt.to_period("M")
rng=np.random.default_rng(42)
print("\n=== 相对 V0 的月度块 bootstrap（2021+）===")
for v in ["V1","V2","V3"]:
    d=res[v].dropna(subset=["ret"]).copy(); d["m"]=pd.to_datetime(d.ed).dt.to_period("M")
    am=d.groupby("m").ret.mean(); bm=base.groupby("m").ret.mean()
    df=pd.concat([am,bm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
    bs=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(3000)] if len(diff)>=5 else None
    lo,hi=(np.percentile(bs,[5,95]) if bs is not None else (np.nan,np.nan))
    print(f"{v} vs V0: 月度差 {diff.mean()*100:+.2f}pp CI90 [{lo*100:+.2f},{hi*100:+.2f}] 月数 {len(diff)}")
print("\n=== 边际样本 ===")
k0=set((r.code,r.ed,r.top) for r in res["V0"].itertuples())
for v in ["V1","V2"]:
    marg=res[v][~res[v].apply(lambda r:(r.code,r.ed,r.top) in k0,axis=1)].dropna(subset=["ret"])
    if len(marg):
        print(f"{v} 相对V0的边际 {len(marg)} 笔: 均值 {marg.ret.mean()*100:6.2f}% 中位 {marg.ret.median()*100:6.2f}% 胜率 {(marg.ret>0).mean()*100:5.1f}%")
print("\n=== 两段 OOS（2021-2023 / 2024-2026）===")
for half in [(2021,2023),(2024,2026)]:
    print(f"-- {half[0]}-{half[1]} --")
    for v in ["V0","V1","V2","V3"]:
        d=res[v].dropna(subset=["ret"]); d=d[(pd.to_datetime(d.ed).dt.year>=half[0])&(pd.to_datetime(d.ed).dt.year<=half[1])]
        if len(d): print(f"   {v}: n={len(d):4d} 均值 {d.ret.mean()*100:6.2f}% 胜率 {(d.ret>0).mean()*100:5.1f}%")
print("\n=== V0 分年（5min，2021+）===")
b=res["V0"].dropna(subset=["ret"]).copy(); b["y"]=pd.to_datetime(b.ed).dt.year
print(b.groupby("y").agg(n=("ret","size"),mean=("ret",lambda x:x.mean()*100),win=("ret",lambda x:(x>0).mean()*100)).round(2).to_string())
