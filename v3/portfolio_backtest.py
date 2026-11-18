# -*- coding: utf-8 -*-
"""portfolio_backtest.py —— 组合层回测：最多 10 只，卖出腾出仓位后买下一个（按 prox 排序）
样本：2021-01-04 起；日线近似执行（买卖点按权威规则，5min 精度问题另行标注）
变体：V0 现状 / V1 prox>=0.85 / V2 ret10<=5% / V3 两者
"""
import numpy as np, pandas as pd, os, sys, time
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M

def masks(P):
    base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["ret10"]<=0.08)&(P["turn30"]<=5)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    return {"V0":base&(P["prox"]>=0.90),
            "V1":base&(P["prox"]>=0.85),
            "V2":base&(P["prox"]>=0.90)&(P["ret10"]<=0.05),
            "V3":base&(P["prox"]>=0.85)&(P["ret10"]<=0.05)}

def build_candidates(P, mask):
    dates=np.array(P["dates"]); nd=P["nd"]; i0=int(np.searchsorted(dates,"2020-12-01"))
    mask=mask.copy(); mask[:i0,:]=False
    hi30=pd.DataFrame(P["C"]).rolling(30).max().values
    prox=P["prox"]
    byday={}
    ii,jj=np.where(mask)
    for k in range(len(ii)):
        t=ii[k]; j=jj[k]; top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        byday.setdefault(t,[]).append((j,float(top),float(prox[t,j])))
    return byday

def simulate(P, byday, start="2021-01-04", maxpos=10, capital=1.0):
    dates=np.array(P["dates"]); nd=P["nd"]
    O=P["O"]; H=P["H"]; L=P["L"]; C=P["C"]; V=P["V"]
    ret=C/np.roll(C,1,axis=0)-1
    s=int(np.searchsorted(dates,start))
    cash=capital; pos={}  # code -> dict(j, entry, shares, half_done, entry_d, hold)
    eq=[]; eq_dates=[]; buys=sells=0; holds=[]
    for d in range(s, nd):
        # 卖出（收盘口径）
        for code,p in list(pos.items()):
            j=p["j"]; c=C[d,j]
            if not np.isfinite(c): continue
            pct=ret[d,j]; pv=V[d,j]; pvv=V[d-1,j]; prof=c/p["entry"]-1
            exit_all=False; exit_px=None
            if np.isfinite(L[d,j]) and L[d,j]<=p["entry"]*0.87:
                exit_px=p["entry"]*0.87; exit_all=True
            elif np.isfinite(pct) and pct<-0.07 and np.isfinite(pv) and np.isfinite(pvv) and pvv>0 and pv>=0.7*pvv:
                exit_px=c; exit_all=True
            elif prof>0.30:
                a=(np.isfinite(pct) and pct>=0.09)
                b=(d-1>=s and np.isfinite(ret[d-1,j]) and ret[d-1,j]<=0.03 and pct<=0.03)
                if a or b:
                    if not p["half_done"]:
                        cash+=p["shares"]*0.5*c; p["shares"]*=0.5; p["half_done"]=True; sells+=1
                    else:
                        exit_px=c; exit_all=True
            if exit_all:
                cash+=p["shares"]*exit_px; sells+=1; holds.append(d-p["entry_d"]); del pos[code]
        # 买入：只用当日开盘前可用现金（卖出所得次日可用）
        free=maxpos-len(pos)
        if free>0 and cash>0:
            cands=[]
            for t in (d-1,d-2):
                for (j,top,rank) in byday.get(t,[]):
                    code=P["codes"][j]
                    if code in pos: continue
                    if not np.isfinite(H[d,j]) or not np.isfinite(O[d,j]): continue
                    trig=top*1.07; cap=top*1.15
                    if H[d,j]>=trig and L[d,j]<=cap:
                        entry=(max(O[d,j],trig) if O[d,j]<=cap else cap)
                        if np.isfinite(entry) and entry>0:
                            cands.append((rank,code,j,entry))
            # 同一只票只保留排名最高的一条，且已持仓的不再买
            seen={}
            for rank,code,j,entry in cands:
                if code not in seen or rank>seen[code][0]:
                    seen[code]=(rank,j,entry)
            cands=[(v[0],k,v[1],v[2]) for k,v in seen.items()]
            cands.sort(key=lambda x:-x[0])
            eq_now=cash+sum(p["shares"]*C[d,p["j"]] for p in pos.values() if np.isfinite(C[d,p["j"]]))
            for rank,code,j,entry in cands:
                if free<=0: break
                if code in pos: continue
                size=min(cash, eq_now/maxpos)
                if size<=0: break
                sh=size/entry
                pos[code]=dict(j=j,entry=entry,shares=sh,half_done=False,entry_d=d); cash-=size; free-=1; buys+=1
        # 估值
        val=sum(p["shares"]*C[d,p["j"]] for p in pos.values() if np.isfinite(C[d,p["j"]]))
        eq.append(cash+val); eq_dates.append(dates[d])
    eq=np.array(eq); eqd=pd.to_datetime(eq_dates)
    r=pd.Series(eq).pct_change().dropna()
    tot=eq[-1]/capital-1; n=len(eq); cagr=(eq[-1]/capital)**(252/n)-1
    mdd=float((pd.Series(eq)/pd.Series(eq).cummax()-1).min())
    sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
    mon=pd.Series(eq,index=eqd).resample("ME").last().pct_change().dropna()
    return dict(total=tot,cagr=cagr,mdd=mdd,sharpe=sharpe,n_trades=buys,sells=sells,
                avg_hold=float(np.mean(holds)) if holds else np.nan,avg_pos=None,
                monthly=mon)

def main():
    t0=time.time(); P=M.build()
    ms=masks(P)
    res={}
    for v,m in ms.items():
        by=build_candidates(P,m)
        r=simulate(P,by)
        res[v]=r["monthly"]
        print(f"{v}: 总收益 {r['total']*100:7.1f}% CAGR {r['cagr']*100:6.1f}% 最大回撤 {r['mdd']*100:6.1f}% Sharpe {r['sharpe']:.2f} 买入{r['n_trades']}笔 卖出{r['sells']}笔 平均持有{r['avg_hold']:.1f}日 月数{len(r['monthly'])} | {time.time()-t0:.0f}s",flush=True)
    R=pd.DataFrame(res)
    R.to_csv(r"D:\about\learn\analysis\portfolio_variants_monthly.csv",encoding="utf-8-sig")
    print("\n月度相关性:")
    print(R.corr().round(2).to_string())
    print("\n分年收益:")
    for y in sorted(set(R.index.year)):
        s=R[R.index.year==y]
        print(y, {c: f"{(1+s[c]).prod()-1:+.1%}" for c in R.columns})

if __name__=="__main__": main()