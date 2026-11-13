# -*- coding: utf-8 -*-
"""base_oos.py —— 新底座（宇宙B + 四条条件）的正式双向样本外
宇宙B：当日成交额>=2亿 & 非涨停 & 清洗 & vol20<=1.0x自身一年均值 & vol20<=35%
候选条件集：预先登记 8 组；每个切分内用训练年挑"fwd20 最高且大亏<=1.5%、日均>=5只"，再看测试年
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
amt=am.values; valid=~np.isnan(px.values); turn30=tn.rolling(30).mean().values
idn=pd.read_csv(ID,dtype={"code":str}).set_index("code")
name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
bm=pd.read_csv(BOARD,dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
board=bm.reindex(codes).values
BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
bad=np.array([(isinstance(b,str) and b in BLACK) for b in board]); clean=~st&~bad
print(f"[{time.time()-t0:.0f}s] factors",flush=True)
fwd20=(px.shift(-20)/px-1).values
amt30=am.rolling(30).mean().values
ar=pd.DataFrame(amt30).rank(axis=1,pct=True).values.astype(np.float32)
bkt=np.clip(np.floor(np.nan_to_num(ar,nan=-1.0)*5.0),0,4).astype(np.int8); bkt[np.isnan(amt30)]=-1
vf=~np.isnan(fwd20); bench=np.full((5,nd),np.nan,np.float32)
for b in range(5):
    m=(bkt==b)&vf; bench[b]=np.nanmean(np.where(m,fwd20,np.nan),axis=1)
ex20=fwd20-bench[np.clip(bkt,0,4),np.arange(nd)[:,None]]; ex20[bkt<0]=np.nan
base=(amt>=2e8)&valid&~(pc.values>=9.8)&clean&(ratio<=1.0)&(vol20<=0.35)&~np.isnan(ex20)
i0=int(dates.searchsorted(pd.Timestamp("2019-01-01"))); yr=dates.year.values
sets={
 "P90":(prox>=0.90),
 "P90+R10":(prox>=0.90)&(ret10<=0.08),
 "P90+R10+T5":(prox>=0.90)&(ret10<=0.08)&(turn30<=5),
 "P90+R10+T5+M0":(prox>=0.90)&(ret10<=0.08)&(turn30<=5)&(ret60>=0),
 "P90+M0":(prox>=0.90)&(ret60>=0),
 "P95+R10+T5":(prox>=0.95)&(ret10<=0.08)&(turn30<=5),
 "P92+R10+T8+M0":(prox>=0.92)&(ret10<=0.10)&(turn30<=8)&(ret60>=0),
 "P88":(prox>=0.88),
}
def stats(mask, years):
    m=mask&base.copy(); m[:i0]=False
    sel=np.isin(yr,years)[:,None]; m=m&sel
    ii,jj=np.where(m); e=ex20[ii,jj]-0.002
    if len(e)==0: return None
    dsum=np.bincount(ii,weights=e,minlength=nd); dc=np.bincount(ii,minlength=nd)
    dm=dsum[dc>0]/dc[dc>0]
    t=dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>2 else np.nan
    ndays=len([y for y in years])
    return dict(n=len(e), per_day=len(e)/max(int(sel.sum()),1), fwd20=e.mean()*100,
                bigloss=(e<-0.20).mean()*100, win=(e>0).mean()*100, t=t)
for name_,train,test in [("S1 训练19-22→测试23-26",[2019,2020,2021,2022],[2023,2024,2025,2026]),
                         ("S2 训练23-26→测试19-22",[2023,2024,2025,2026],[2019,2020,2021,2022])]:
    print(f"\n=========== {name_} ===========")
    print(f"{'条件集':16s}{'训练n/日':>8s}{'训练fwd20':>9s}{'训练大亏':>8s}{'训练t':>7s} || {'测试n/日':>8s}{'测试fwd20':>9s}{'测试大亏':>8s}{'测试t':>7s}")
    best=None
    for k,cond in sets.items():
        tr=stats(cond,train); te=stats(cond,test)
        if tr is None or te is None: continue
        print(f"{k:16s}{tr['per_day']:8.1f}{tr['fwd20']:9.2f}{tr['bigloss']:8.1f}{tr['t']:7.2f} || {te['per_day']:8.1f}{te['fwd20']:9.2f}{te['bigloss']:8.1f}{te['t']:7.2f}")
        if tr['n']>200 and tr['bigloss']<=1.5 and tr['per_day']>=5:
            if best is None or tr['fwd20']>best[1]['fwd20']: best=(k,tr,te)
    if best: print(f"→ 训练年选中最优：{best[0]}（训练 fwd20 {best[1]['fwd20']:.2f}%）→ 测试 fwd20 {best[2]['fwd20']:.2f}%、大亏 {best[2]['bigloss']:.1f}%、t {best[2]['t']:.2f}")
print(f"\n[{time.time()-t0:.0f}s] done")
