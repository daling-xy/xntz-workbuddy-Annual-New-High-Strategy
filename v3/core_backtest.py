# -*- coding: utf-8 -*-
"""core_backtest.py —— 只比较 v3 自己的参数变体（真实交易口径，日线近似）
选股底座(v3)：当日成交额>=2亿 & 清洗 & 非涨停 & prox>=0.90 & ret10<=8% & turn30<=5%
变体：V1 v3现状(volratio<=1.0 & vol20<=35 & ret60>=0)；V2 volratio<=1.2；V3 ret30>=0；V4 无波动门
买入/卖出按 CORE_RULES.md（30日箱顶×1.07-1.15 触发；-13%止损；盈利>30%两档卖一半；跌幅>7%量能确认全卖）
"""
import numpy as np, pandas as pd, pickle, time, os
t0=time.time()
B=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
kl=pickle.load(open(os.path.join(B,"cache","kline_long2.pkl"),"rb"))
codes=sorted(kl.keys()); dates=sorted({b["date"] for v in kl.values() for b in v})
di={d:i for i,d in enumerate(dates)}; nd,nc=len(dates),len(codes)
O=np.full((nd,nc),np.nan,np.float32); H=np.full((nd,nc),np.nan,np.float32)
L=np.full((nd,nc),np.nan,np.float32); C=np.full((nd,nc),np.nan,np.float32)
V=np.full((nd,nc),np.nan,np.float32); A=np.full((nd,nc),np.nan,np.float32); T=np.full((nd,nc),np.nan,np.float32)
for j,c in enumerate(codes):
    for b in kl[c]:
        i=di[b["date"]]; O[i,j]=b["open"]; H[i,j]=b["high"]; L[i,j]=b["low"]; C[i,j]=b["close"]; V[i,j]=b["volume"]; A[i,j]=b["amount"]; T[i,j]=b.get("turnover",np.nan)
print(f"[{time.time()-t0:.0f}s] panel {nd}x{nc}",flush=True)
ret=C/np.roll(C,1,axis=0)-1
vol20=(pd.DataFrame(ret).rolling(20).std().values)*np.sqrt(252)
vol_hist=pd.DataFrame(vol20).rolling(250).mean().values
volratio=vol20/np.where(vol_hist>0,vol_hist,np.nan)
prox=C/np.where(pd.DataFrame(C).rolling(250).max().values>0,pd.DataFrame(C).rolling(250).max().values,np.nan)
hi30=pd.DataFrame(C).rolling(30).max().values; lo30=pd.DataFrame(C).rolling(30).min().values
ret10=(C/np.roll(C,10,axis=0)-1); ret30=(C/np.roll(C,30,axis=0)-1); ret60=(C/np.roll(C,60,axis=0)-1)
turn30=pd.DataFrame(T).rolling(30).mean().values
prevV=np.roll(V,1,axis=0)
idn=pd.read_csv(os.path.join(B,"v3","data","deliver_identity_table.csv"),dtype={"code":str}).set_index("code")
name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
bm=pd.read_csv(os.path.join(B,"v3","data","board_map.csv"),dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
bad=np.array([(isinstance(b,str) and b in BLACK) for b in bm.reindex(codes).values]); clean=~st&~bad
valid=np.isfinite(C)
mkt=np.nanmean(np.where(valid,ret,np.nan),axis=1); mkt=np.where(np.isfinite(mkt),mkt,0.0)
mktcum=np.cumprod(1+mkt)
base=valid&clean&(A>=2e8)&(prox>=0.90)&(ret10<=0.08)&(turn30<=5)&(C>0)
tp=np.where(hi30>0,hi30,np.nan)*1.07; bt=np.where(hi30>0,hi30,np.nan)*0.80
tradeable=valid&(C<tp)&(C>=bt)
VARS={
 "V1 v3现状(volratio<=1.0)":base&(volratio<=1.0)&(vol20<=0.35)&(ret60>=0),
 "V2 volratio<=1.2":base&(np.round(volratio,2)<=1.2)&(vol20<=0.35)&(ret60>=0),
 "V3 ret30>=0替代ret60":base&(volratio<=1.0)&(vol20<=0.35)&(ret30>=0),
 "V4 无波动门(ret60>=0)":base&(ret60>=0),
}
i0=int(np.searchsorted(np.array(dates),'2019-01-01')); days=nd-i0
def run(mask,lab):
    m=mask.copy(); m[:i0,:]=False
    ii,jj=np.where(m); n=len(ii)
    trad=tradeable[ii,jj]; past=valid[ii,jj]&(C[ii,jj]>=tp[ii,jj]); below=valid[ii,jj]&(C[ii,jj]<bt[ii,jj])
    # 只对可交易候选模拟
    ti=ii[trad]; tj=jj[trad]; trig=0; trs=[]; hds=[]
    for k in range(len(ti)):
        t=ti[k]; j=tj[k]; top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        trigp=top*1.07; cap=top*1.15; entry=None; ed=None
        for d in (t+1,t+2):
            if d>=nd: break
            if not np.isfinite(H[d,j]) or not np.isfinite(O[d,j]): continue
            if H[d,j]>=trigp and O[d,j]<=cap:
                entry=max(O[d,j],trigp); ed=d; break
        if entry is None: continue
        trig+=1; stop=entry*0.87; half=None; xp=None; xd=None; d=ed+1
        while d<nd:
            if not np.isfinite(C[d,j]): d+=1; continue
            pc=ret[d,j]; pv=V[d,j]; pvv=prevV[d,j]; prof=C[d,j]/entry-1
            if np.isfinite(L[d,j]) and L[d,j]<=stop: xp=stop; xd=d; break
            if np.isfinite(pc) and pc<-0.07 and np.isfinite(pv) and np.isfinite(pvv) and pvv>0 and pv>=pvv*0.7: xp=C[d,j]; xd=d; break
            if half is None and prof>0.30:
                if np.isfinite(pc) and pc>=0.09: half=C[d,j]
                elif d-1>=ed+1 and np.isfinite(ret[d-1,j]) and ret[d-1,j]<=0.03 and pc<=0.03: half=C[d,j]
            d+=1
        if xp is None: xd=nd-1; xp=C[xd,j]
        r=(0.5*half+0.5*xp)/entry-1 if half is not None else xp/entry-1
        mret=mktcum[xd]/mktcum[ed]-1; trs.append((r,mret)); hds.append(xd-ed)
    arr=np.array(trs); 
    print(f"\n--- {lab} ---")
    print(f"  候选 {n}（日均 {n/days:.1f}）| 可交易(箱顶×0.8~1.07) {int(trad.sum())} ({trad.mean()*100:.1f}%) | 已过触发线 {int(past.sum())} ({past.mean()*100:.1f}%) | 破箱底 {int(below.sum())} ({below.mean()*100:.1f}%)")
    print(f"  触发 {trig} 笔（可交易中 {trig/max(int(trad.sum()),1)*100:.1f}%）")
    if len(arr):
        r=arr[:,0]; mret=arr[:,1]; exc=(1+r)/(1+mret)-1
        print(f"  平均 {r.mean()*100:6.2f}% 中位 {np.median(r)*100:6.2f}% 胜率 {(r>0).mean()*100:5.1f}% 大亏 {(r<=-0.20).mean()*100:4.1f}% | 持有 {np.mean(hds):5.1f} 日 | 超额 {exc.mean()*100:5.2f}% 中位 {np.median(exc)*100:5.2f}%")
for lab,mk in VARS.items(): run(mk,lab)
print(f"\n[{time.time()-t0:.0f}s] done")
