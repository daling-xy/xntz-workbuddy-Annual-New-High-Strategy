# -*- coding: utf-8 -*-
"""vol_gate_design.py —— 波动率门怎么定：绝对阈值 vs 自身历史相对 vs 行业相对
宇宙：当日成交额>=2亿 & 非涨停 & 有效（2019-2026，learn 数据）
"""
import numpy as np, pandas as pd, time
t0=time.time()
FULL=r"D:\about\learn\data\full_a_share_v2.csv"; ID=r"D:\about\learn\data\deliver_identity_table.csv"
d=pd.read_csv(FULL,usecols=["date","code","close_hfq","amount","pct_chg"],dtype={"code":str})
d["date"]=pd.to_datetime(d["date"]); d=d[d.date<=pd.Timestamp("2026-11-04")]
d.loc[d.close_hfq<=0,"close_hfq"]=np.nan
px=d.pivot(index="date",columns="code",values="close_hfq").astype("float32").sort_index()
am=d.pivot(index="date",columns="code",values="amount").astype("float32").sort_index()
pc=d.pivot(index="date",columns="code",values="pct_chg").astype("float32").sort_index()
px=px.loc["2017-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index)
keep=px.columns[px.notna().sum()>=400]; px,am,pc=px[keep],am[keep],pc[keep]
px=px.loc["2018-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index)
dates=px.index; codes=px.columns; nd,nc=px.shape
ret=px.pct_change(fill_method=None)
vol20=(ret.rolling(20).std().values)*np.sqrt(252)
vol_hist=pd.DataFrame(vol20).rolling(250).mean().values
ratio_own=vol20/np.where(vol_hist>0,vol_hist,np.nan)
idn=pd.read_csv(ID,dtype={"code":str}).set_index("code")
ind=idn["主行业(赛道)"].reindex(codes).values
uniq=[u for u in pd.unique(ind) if isinstance(u,str)]
M=np.zeros((len(uniq),nc),np.float32)
for k,u in enumerate(uniq): M[k,ind==u]=1.0
cnt=M.sum(axis=1)[:,None]
ind_vol=(M@np.nan_to_num(vol20, nan=0.0).T)/np.maximum(cnt,1e-9)  # (n_ind, nd)
idx_map={u:k for k,u in enumerate(uniq)}
ind_of=np.array([idx_map.get(u,-1) if isinstance(u,str) else -1 for u in ind])
print(f"[{time.time()-t0:.0f}s] factors",flush=True)
fwd20=(px.shift(-20)/px-1).values
amt=am.values
valid=~np.isnan(px.values)
amt30=am.rolling(30).mean().values
ar=pd.DataFrame(amt30).rank(axis=1,pct=True).values.astype(np.float32)
bkt=np.clip(np.floor(np.nan_to_num(ar,nan=-1.0)*5.0),0,4).astype(np.int8); bkt[np.isnan(amt30)]=-1
vf=~np.isnan(fwd20); bench=np.full((5,nd),np.nan,np.float32)
for b in range(5):
    m=(bkt==b)&vf; bench[b]=np.nanmean(np.where(m,fwd20,np.nan),axis=1)
ex20=fwd20-bench[np.clip(bkt,0,4),np.arange(nd)[:,None]]; ex20[bkt<0]=np.nan
i0=int(dates.searchsorted(pd.Timestamp("2019-01-01")))
base=(amt>=2e8)&valid&~(pc.values>=9.8)
def bins_table(name, arr, edges, mask):
    m=mask&~np.isnan(arr)&~np.isnan(ex20); m[:i0]=False
    ii,jj=np.where(m)
    print(f"\n=== {name}（n={len(ii)}）===")
    print(f"{'区间':>14s} {'n':>8s} {'日均':>6s} {'fwd20均%':>9s} {'中位%':>7s} {'胜率%':>6s} {'大亏%':>6s}")
    ndays=nd-i0
    for lo,hi in zip(edges[:-1],edges[1:]):
        sel=(arr[ii,jj]>=lo)&(arr[ii,jj]<hi)
        ee=ex20[ii[sel],jj[sel]]
        if len(ee)==0: continue
        print(f"{lo:6.2f}-{hi:6.2f} {len(ee):8d} {len(ee)/ndays:6.1f} {ee.mean()*100:9.2f} {np.median(ee)*100:7.2f} {(ee>0).mean()*100:6.1f} {(ee<-0.20).mean()*100:6.1f}")
E=[0,0.15,0.20,0.25,0.30,0.35,0.40,0.50,1.0]
bins_table("vol20 绝对区间", vol20, E, base)
ER=[0,0.6,0.8,1.0,1.2,1.5,2.0,3.0,9]
bins_table("vol20 / 自身一年均值", ratio_own, ER, base)
ind_vol_mat=ind_vol[np.clip(ind_of,0,None),:].T   # (nd, nc)
ratio_ind=vol20/np.where(ind_of[None,:]>=0, ind_vol_mat, np.nan)
bins_table("vol20 / 行业均值", ratio_ind, ER, base)
print("\n=== 交叉：绝对高波(>30%) × 自身相对低(<1.0) ===")
for lab,mk in [("abs>30% & ratio<1.0",(vol20>0.30)&(ratio_own<1.0)),
               ("abs>30% & ratio>=1.0",(vol20>0.30)&(ratio_own>=1.0)),
               ("abs<=25% & ratio<1.0",(vol20<=0.25)&(ratio_own<1.0)),
               ("abs<=25% & ratio>=1.0",(vol20<=0.25)&(ratio_own>=1.0))]:
    m=base&mk&~np.isnan(ex20); m[:i0]=False
    e=ex20[m]; print(f"  {lab:24s} 日均 {m.sum()/max(nd-i0,1):6.1f} 只  fwd20均 {e.mean()*100:6.2f}%  大亏 {(e<-0.20).mean()*100:4.1f}%")
print("\n=== 候选门对比 ===")
med_own=vol_hist
gates={
 "G1 vol20<=25%":(vol20<=0.25),
 "G2 vol20<=30%":(vol20<=0.30),
 "G3 ratio_own<=1.0":(ratio_own<=1.0),
 "G4 ratio_own<=1.0 & vol20<=40%":(ratio_own<=1.0)&(vol20<=0.40),
 "G5 vol20<=max(自身均值,25%) & <=40%":(vol20<=np.maximum(med_own,0.25))&(vol20<=0.40),
}
for lab,mk in gates.items():
    m=base&mk&~np.isnan(ex20); m[:i0]=False
    e=ex20[m]; print(f"  {lab:34s} 日均 {m.sum()/max(nd-i0,1):6.1f} 只  fwd20均 {e.mean()*100:6.2f}%  大亏 {(e<-0.20).mean()*100:4.1f}%")
print(f"\n[{time.time()-t0:.0f}s] done")
