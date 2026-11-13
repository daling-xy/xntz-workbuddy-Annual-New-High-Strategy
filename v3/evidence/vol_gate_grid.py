# -*- coding: utf-8 -*-
"""vol_gate_grid.py —— 波动率门的组合网格对比（ratio x 绝对上限）
评估：日均只数、fwd20/fwd60 超额、大亏、分年、前后半样本外
"""
import numpy as np, pandas as pd, time
t0=time.time()
FULL=r"D:\about\learn\data\full_a_share_v2.csv"
d=pd.read_csv(FULL,usecols=["date","code","close_hfq","amount","pct_chg"],dtype={"code":str})
d["date"]=pd.to_datetime(d["date"]); d=d[d.date<=pd.Timestamp("2026-11-04")]
d.loc[d.close_hfq<=0,"close_hfq"]=np.nan
px=d.pivot(index="date",columns="code",values="close_hfq").astype("float32").sort_index()
am=d.pivot(index="date",columns="code",values="amount").astype("float32").sort_index()
pc=d.pivot(index="date",columns="code",values="pct_chg").astype("float32").sort_index()
px=px.loc["2017-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index)
keep=px.columns[px.notna().sum()>=400]; px,am,pc=px[keep],am[keep],pc[keep]
px=px.loc["2018-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index)
dates=px.index; nd,nc=px.shape
ret=px.pct_change(fill_method=None)
vol20=(ret.rolling(20).std().values)*np.sqrt(252)
vol_hist=pd.DataFrame(vol20).rolling(250).mean().values
ratio=vol20/np.where(vol_hist>0,vol_hist,np.nan)
print(f"[{time.time()-t0:.0f}s] factors",flush=True)
fwd20=(px.shift(-20)/px-1).values; fwd60=(px.shift(-60)/px-1).values
amt=am.values; valid=~np.isnan(px.values)
amt30=am.rolling(30).mean().values
ar=pd.DataFrame(amt30).rank(axis=1,pct=True).values.astype(np.float32)
bkt=np.clip(np.floor(np.nan_to_num(ar,nan=-1.0)*5.0),0,4).astype(np.int8); bkt[np.isnan(amt30)]=-1
def ex_of(fwd):
    vf=~np.isnan(fwd); bench=np.full((5,nd),np.nan,np.float32)
    for b in range(5):
        m=(bkt==b)&vf; bench[b]=np.nanmean(np.where(m,fwd,np.nan),axis=1)
    ex=fwd-bench[np.clip(bkt,0,4),np.arange(nd)[:,None]]; ex[bkt<0]=np.nan; return ex
ex20=ex_of(fwd20); ex60=ex_of(fwd60)
i0=int(dates.searchsorted(pd.Timestamp("2019-01-01")))
base=(amt>=2e8)&valid&~(pc.values>=9.8)
yr=dates.year.values
print("\n=== 波动率门组合网格（宇宙=当日成交额≥2亿，扣0.2%成本）===")
hdr=f"{'组合':28s} {'日均':>6s} {'fwd20':>7s} {'中位':>7s} {'胜率':>5s} {'大亏':>5s} {'fwd60':>7s} {'19-22':>7s} {'23-26':>7s} {'2021':>6s} {'2022':>6s}"
print(hdr)
rows=[]
for rt,rt_lab in [(1.0,"1.0x"),(1.2,"1.2x"),(9,"无")]:
    for cap in [0.25,0.30,0.35,0.40]:
        m=base&(ratio<=rt if rt<9 else True)&(vol20<=cap)&~np.isnan(ex20); m[:i0]=False
        e=ex20[m]-0.002
        e6=ex60[m]-0.002
        def ym(y0,y1):
            sel=(yr>=y0)&(yr<=y1); mm=m&sel[:,None]; ee=ex20[mm]-0.002
            return ee.mean()*100 if len(ee) else np.nan
        lab=f"ratio<={rt_lab} & vol<={cap*100:.0f}%"
        print(f"{lab:28s} {m.sum()/max(nd-i0,1):6.0f} {e.mean()*100:7.2f} {np.median(e)*100:7.2f} {(e>0).mean()*100:5.1f} {(e<-0.20).mean()*100:5.1f} {np.nanmean(e6)*100:7.2f} {ym(2019,2022):7.2f} {ym(2023,2026):7.2f} {ym(2021,2021):6.2f} {ym(2022,2022):6.2f}")
        rows.append(dict(ratio=rt_lab,cap=cap,n_day=m.sum()/max(nd-i0,1),fwd20=e.mean()*100,
                         fwd60=np.nanmean(e6)*100,bigloss=(e<-0.20).mean()*100,first=ym(2019,2022),second=ym(2023,2026)))
R=pd.DataFrame(rows)
R["score"]=R.fwd20-0.5*R.bigloss
print("\n按 (fwd20 − 0.5×大亏率) 排序：")
print(R.sort_values("score",ascending=False)[["ratio","cap","n_day","fwd20","fwd60","bigloss","first","second","score"]].round(2).to_string(index=False))
print(f"\n[{time.time()-t0:.0f}s] done")
