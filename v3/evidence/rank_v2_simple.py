# -*- coding: utf-8 -*-
"""rank_v2_simple.py —— 新底座上只用"两半样本都同号"的稳定特征做排序，OOS 检验
稳定特征：prox(+)、dist20(+)。对照：prox 单因子、prox+dist20。
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
volratio=vol20/np.where(vol_hist>0,vol_hist,np.nan)
prox=(px/px.rolling(250).max()).values
ret10=(px/px.shift(10)-1).values; ret60=(px/px.shift(60)-1).values
h20=px.rolling(20).max().values; l20=px.rolling(20).min().values
pos20=(px.values-l20)/np.where((h20-l20)>0,(h20-l20),np.nan)
dist20=px.values/np.where(h20>0,h20,np.nan)
amt=am.values; valid=~np.isnan(px.values); turn30=tn.rolling(30).mean().values
idn=pd.read_csv(ID,dtype={"code":str}).set_index("code")
name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
bm=pd.read_csv(BOARD,dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
board=bm.reindex(codes).values
BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
bad=np.array([(isinstance(b,str) and b in BLACK) for b in board]); clean=~st&~bad
fwd20=(px.shift(-20)/px-1).values
amt30=am.rolling(30).mean().values
ar=pd.DataFrame(amt30).rank(axis=1,pct=True).values.astype(np.float32)
bkt=np.clip(np.floor(np.nan_to_num(ar,nan=-1.0)*5.0),0,4).astype(np.int8); bkt[np.isnan(amt30)]=-1
vf=~np.isnan(fwd20); bench=np.full((5,nd),np.nan,np.float32)
for b in range(5):
    m=(bkt==b)&vf; bench[b]=np.nanmean(np.where(m,fwd20,np.nan),axis=1)
ex20=fwd20-bench[np.clip(bkt,0,4),np.arange(nd)[:,None]]; ex20[bkt<0]=np.nan
base=(amt>=2e8)&valid&~(pc.values>=9.8)&clean&(volratio<=1.0)&(vol20<=0.35)&(prox>=0.90)&(ret10<=0.08)&(turn30<=5)&(ret60>=0)&~np.isnan(ex20)
i0=int(dates.searchsorted(pd.Timestamp("2019-01-01"))); base[:i0]=False
ii,jj=np.where(base)
E=pd.DataFrame({"day":ii,"code":jj,"ex20":ex20[ii,jj],"year":dates.year[ii].astype(int)})
F={"prox":prox,"dist20":dist20,"pos20":pos20,"ret20":(px/px.shift(20)-1).values}
for k,v in F.items(): E[k]=v[ii,jj]
for f in F: E["z_"+f]=E.groupby("day")[f].transform(lambda s:(s-s.mean())/(s.std(ddof=0)+1e-9))
print(f"candidates {len(E)} ({len(E)/(nd-i0):.1f}/day)  [{time.time()-t0:.0f}s]",flush=True)
def ev(sub,score,ta,tb,label):
    sub=sub.copy(); sub["score"]=score; ics=[]
    for day,g in sub.groupby("day"):
        if len(g)<8: continue
        r=g.score.rank().corr(g.ex20.rank())
        if not np.isnan(r): ics.append(r)
    ic=pd.Series(ics); ndays=sub.day.nunique()
    print(f"  {label}: IC={ic.mean():.3f} t={ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic))):.2f}")
    for lo,hi,nm in [(ta,9,"A"),(tb,ta,"B"),(-9,tb,"C")]:
        s=sub[(sub.score>=lo)&(sub.score<hi)]; e=s.ex20.values
        if len(e)==0: continue
        print(f"    {nm}: 日均{len(s)/ndays:5.1f} mean={e.mean()*100:6.2f}% 大亏={(e<-0.20).mean()*100:4.1f}%")
for label,expr in [("prox",["prox"]),("prox+dist20",["prox","dist20"]),("prox+dist20+pos20",["prox","dist20","pos20"])]:
    print(f"\n=== 评分 = {'+'.join(expr)} ===")
    sc=sum(E["z_"+f].values for f in expr)
    for name_,train,test in [("19-22→23-26",[2019,2020,2021,2022],[2023,2024,2025,2026]),
                             ("23-26→19-22",[2023,2024,2025,2026],[2019,2020,2021,2022])]:
        Etr=E[E.year.isin(train)]; Ete=E[E.year.isin(test)]
        s_tr=pd.Series(sc,index=E.index).loc[Etr.index].values
        s_te=pd.Series(sc,index=E.index).loc[Ete.index].values
        ta=float(pd.Series(s_tr).quantile(0.85)); tb=float(pd.Series(s_tr).quantile(0.50))
        print(f"  切分 {name_}：")
        ev(Etr,s_tr,ta,tb,"训练"); ev(Ete,s_te,ta,tb,"测试★")
print(f"\n[{time.time()-t0:.0f}s] done")
