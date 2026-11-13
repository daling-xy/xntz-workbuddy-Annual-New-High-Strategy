# -*- coding: utf-8 -*-
"""rebuild_conditions.py —— 三个候选宇宙上重建条件组合（去掉 ct/C/H/F）
宇宙：A=1.2x&35%  B=1.0x&35%  C=1.0x&40%
条件（逐步叠加）：prox / w30 / ret10 / turn30 / ret60
评估：fwd20/fwd60 超额（同流动性，扣0.2%）、大亏、分前后半样本
"""
import numpy as np, pandas as pd, time
t0=time.time()
FULL=r"D:\about\learn\data\full_a_share_v2.csv"; ID=r"D:\about\learn\data\deliver_identity_table.csv"
BOARD=r"C:\Users\xntz\Desktop\test\yearly_high_strategy\output\v2\全市场打分明细_v2.csv"
d=pd.read_csv(FULL,usecols=["date","code","close_hfq","amount","pct_chg","turnover"],dtype={"code":str})
d["date"]=pd.to_datetime(d["date"]); d=d[d.date<=pd.Timestamp("2026-11-04")]
d.loc[d.close_hfq<=0,"close_hfq"]=np.nan
px=d.pivot(index="date",columns="code",values="close_hfq").astype("float32").sort_index()
am=d.pivot(index="date",columns="code",values="amount").astype("float32").sort_index()
pc=d.pivot(index="date",columns="code",values="pct_chg").astype("float32").sort_index()
tn=d.pivot(index="date",columns="code",values="turnover").astype("float32").sort_index()
px=px.loc["2017-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index); tn=tn.reindex(px.index)
keep=px.columns[px.notna().sum()>=400]; px,am,pc,tn=px[keep],am[keep],pc[keep],tn[keep]
px=px.loc["2018-06-01":]; am=am.reindex(px.index); pc=pc.reindex(px.index); tn=tn.reindex(px.index)
dates=px.index; codes=px.columns; nd,nc=px.shape
ret=px.pct_change(fill_method=None)
vol20=(ret.rolling(20).std().values)*np.sqrt(252)
vol_hist=pd.DataFrame(vol20).rolling(250).mean().values
ratio=vol20/np.where(vol_hist>0,vol_hist,np.nan)
prox=(px/px.rolling(250).max()).values
w30=(px.rolling(30).max()/px.rolling(30).min()-1).values
ret10=(px/px.shift(10)-1).values; ret60=(px/px.shift(60)-1).values
amt=am.values; valid=~np.isnan(px.values)
turn30=tn.rolling(30).mean().values
idn=pd.read_csv(ID,dtype={"code":str}).set_index("code")
name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
bm=pd.read_csv(BOARD,dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
board=bm.reindex(codes).values
BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
bad=np.array([(isinstance(b,str) and b in BLACK) for b in board])
clean=~st&~bad
print(f"[{time.time()-t0:.0f}s] factors",flush=True)
fwd20=(px.shift(-20)/px-1).values; fwd60=(px.shift(-60)/px-1).values
amt30=am.rolling(30).mean().values
ar=pd.DataFrame(amt30).rank(axis=1,pct=True).values.astype(np.float32)
bkt=np.clip(np.floor(np.nan_to_num(ar,nan=-1.0)*5.0),0,4).astype(np.int8); bkt[np.isnan(amt30)]=-1
def ex_of(fwd):
    vf=~np.isnan(fwd); bench=np.full((5,nd),np.nan,np.float32)
    for b in range(5):
        m=(bkt==b)&vf; bench[b]=np.nanmean(np.where(m,fwd,np.nan),axis=1)
    ex=fwd-bench[np.clip(bkt,0,4),np.arange(nd)[:,None]]; ex[bkt<0]=np.nan; return ex
ex20=ex_of(fwd20); ex60=ex_of(fwd60)
i0=int(dates.searchsorted(pd.Timestamp("2019-01-01"))); yr=dates.year.values
nf=~(pc.values>=9.8)
base0=(amt>=2e8)&valid&nf&clean
universes={"A 1.2x&35%":(ratio<=1.2)&(vol20<=0.35),"B 1.0x&35%":(ratio<=1.0)&(vol20<=0.35),"C 1.0x&40%":(ratio<=1.0)&(vol20<=0.40)}
sets={
 "S0 仅宇宙":None,
 "S1 prox>=0.90":(prox>=0.90),
 "S1b prox>=0.85":(prox>=0.85),
 "S1c prox>=0.95":(prox>=0.95),
 "S2 prox.90+ret10<=8":(prox>=0.90)&(ret10<=0.08),
 "S3 S2+turn<=5%":(prox>=0.90)&(ret10<=0.08)&(turn30<=5),
 "S4 S3+ret60>=0":(prox>=0.90)&(ret10<=0.08)&(turn30<=5)&(ret60>=0),
 "S5 S2+w30<=15%":(prox>=0.90)&(ret10<=0.08)&(w30<=0.15),
 "S6 S3+ret60>=0.10":(prox>=0.90)&(ret10<=0.08)&(turn30<=5)&(ret60>=0.10),
}
print("\n=== 三宇宙 × 条件组合（fwd20/fwd60 超额，扣0.2%，分前后半样本）===")
print(f"{'宇宙':12s}{'条件':16s}{'日均':>6s}{'fwd20':>7s}{'fwd60':>7s}{'大亏':>5s}{'胜率':>5s}{'19-22':>7s}{'23-26':>7s}{'t':>6s}")
for un,ug in universes.items():
    for sn,cond in sets.items():
        m=base0&ug
        if cond is not None: m=m&cond
        m=m&~np.isnan(ex20); m[:i0]=False
        ii,jj=np.where(m); e=ex20[ii,jj]-0.002; e6=ex60[ii,jj]-0.002
        if len(e)==0: continue
        dsum=np.bincount(ii,weights=e,minlength=nd); dc=np.bincount(ii,minlength=nd)
        dm=dsum[dc>0]/dc[dc>0]; tt=dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>2 else np.nan
        def ym(y0,y1):
            sel=(yr[ii]>=y0)&(yr[ii]<=y1); return e[sel].mean()*100 if sel.sum() else np.nan
        print(f"{un:12s}{sn:16s}{len(e)/(nd-i0):6.1f}{e.mean()*100:7.2f}{np.nanmean(e6)*100:7.2f}{(e<-0.20).mean()*100:5.1f}{(e>0).mean()*100:5.1f}{ym(2019,2022):7.2f}{ym(2023,2026):7.2f}{tt:6.2f}")
    print()
print(f"[{time.time()-t0:.0f}s] done")
