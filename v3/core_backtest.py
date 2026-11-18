# -*- coding: utf-8 -*-
"""core_backtest.py —— 按权威买卖点做真实交易口径回测（日线近似）
选股：箱体w30<=25% & prox>=0.90 & 当日成交额>=2亿 & 清洗 & 箱顶×0.80<=close<箱顶×1.07
买入：触发后2个交易日内，high>=箱顶×1.07 且 open<=箱顶×1.15 → 成交价=max(open, 箱顶×1.07)
卖出：① low<=买入价×0.87 → 止损；② 盈利>30% & 当日涨幅>=9% → 卖一半；盈利>30% & 连续两日涨幅<=3% → 卖一半；
      ③ 跌幅>7% & 量>=前日×0.7（或>=前日） → 全卖；剩余到期/数据末按收盘平仓
对比：C0 核心选股；C1 核心+波动overlay；C2 核心+v3过滤
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
print(f"[{time.time()-t0:.0f}s] panel {nd}x{nc} {dates[0]}~{dates[-1]}",flush=True)
ret=C/np.roll(C,1,axis=0)-1; ret[0,:]=np.nan
vol20=(pd.DataFrame(ret).rolling(20).std().values)*np.sqrt(252)
vol_hist=pd.DataFrame(vol20).rolling(250).mean().values
volratio=vol20/np.where(vol_hist>0,vol_hist,np.nan)
prox=C/np.where(pd.DataFrame(C).rolling(250).max().values>0,pd.DataFrame(C).rolling(250).max().values,np.nan)
hi30=pd.DataFrame(C).rolling(30).max().values; lo30=pd.DataFrame(C).rolling(30).min().values
w30=hi30/np.where(lo30>0,lo30,np.nan)-1
ret10=(C/np.roll(C,10,axis=0)-1); ret60=(C/np.roll(C,60,axis=0)-1)
turn30=pd.DataFrame(T).rolling(30).mean().values
amt=A
prevV=np.roll(V,1,axis=0); prevC=np.roll(C,1,axis=0)
idn=pd.read_csv(os.path.join(B,"v3","data","deliver_identity_table.csv"),dtype={"code":str}).set_index("code")
name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
bm=pd.read_csv(os.path.join(B,"v3","data","board_map.csv"),dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
bad=np.array([(isinstance(b,str) and b in BLACK) for b in bm.reindex(codes).values]); clean=~st&~bad
valid=np.isfinite(C)
# 市场等权日收益（用于超额）
mkt=np.nanmean(np.where(valid,ret,np.nan),axis=1)
mkt=np.where(np.isfinite(mkt),mkt,0.0)
mktcum=np.concatenate([[1.0],np.cumprod(1+mkt)])[1:]  # 从0到i累计
core=valid&clean&(amt>=2e8)&(w30<=0.25)&(prox>=0.90)&(C<np.where(hi30>0,hi30,np.nan)*1.07)&(C>=np.where(hi30>0,hi30,np.nan)*0.80)
volov=(volratio<=1.0)&(vol20<=0.35)
v3f=volov&(ret10<=0.08)&(turn30<=5)&(ret60>=0)
VARS={"C0 核心选股":core,"C1 核心+波动overlay":core&volov,"C2 核心+v3过滤":core&v3f}
i0=int(np.searchsorted(np.array(dates),'2019-01-01'))
def simulate(mask,lab):
    m=mask.copy(); m[:i0,:]=False
    ii,jj=np.where(m)
    n_cand=len(ii); trig=0; trades=[]; holds=[]
    for k in range(n_cand):
        t=ii[k]; j=jj[k]
        top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        trigp=top*1.07; cap=top*1.15
        entry=None; ed=None
        for d in (t+1,t+2):
            if d>=nd: break
            if not np.isfinite(H[d,j]) or not np.isfinite(O[d,j]): continue
            if H[d,j]>=trigp and O[d,j]<=cap:
                entry=max(O[d,j],trigp); ed=d; break
        if entry is None: continue
        trig+=1
        stop=entry*0.87; half=None; exit_px=None; xd=None
        d=ed+1
        while d<nd:
            if not np.isfinite(C[d,j]): d+=1; continue
            pc=ret[d,j]; pv=V[d,j]; pvv=prevV[d,j]
            prof=C[d,j]/entry-1
            if np.isfinite(L[d,j]) and L[d,j]<=stop:
                exit_px=stop; xd=d; break
            if np.isfinite(pc) and pc<-0.07 and np.isfinite(pv) and np.isfinite(pvv) and pvv>0 and pv>=pvv*0.7:
                exit_px=C[d,j]; xd=d; break
            if half is None and prof>0.30:
                if np.isfinite(pc) and pc>=0.09:
                    half=C[d,j]
                elif d-1>=ed+1 and np.isfinite(ret[d-1,j]) and ret[d-1,j]<=0.03 and pc<=0.03:
                    half=C[d,j]
            d+=1
        if exit_px is None:
            xd=nd-1; exit_px=C[xd,j]
        if half is not None:
            r=(0.5*half+0.5*exit_px)/entry-1
        else:
            r=exit_px/entry-1
        mret=mktcum[xd]/mktcum[ed]-1
        trades.append((r,mret)); holds.append(xd-ed)
    arr=np.array(trades); holds=np.array(holds)
    r=arr[:,0]; mret=arr[:,1]; exc=(1+r)/(1+mret)-1
    print(f"\n--- {lab} ---")
    print(f"  候选 {n_cand}（日均 {n_cand/max(nd-i0,1):.1f}） 触发 {trig}（触发率 {trig/max(n_cand,1)*100:.1f}%）")
    if len(r):
        print(f"  交易 {len(r)} 笔 | 平均 {r.mean()*100:.2f}% 中位 {np.median(r)*100:.2f}% 胜率 {(r>0).mean()*100:.1f}% 大亏(<=-20%) {(r<=-0.20).mean()*100:.1f}% | 平均持有 {holds.mean():.1f} 日 | 平均超额(等权) {exc.mean()*100:.2f}% 超额中位 {np.median(exc)*100:.2f}%")
    return dict(n=n_cand,trig=trig,r=r,holds=holds,exc=exc)
res={}
for lab,mk in VARS.items():
    res[lab]=simulate(mk,lab)
print(f"\n[{time.time()-t0:.0f}s] done")
