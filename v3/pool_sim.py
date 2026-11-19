# -*- coding: utf-8 -*-
"""pool_sim.py —— 票池 + 10 仓位 + 5min 执行的持仓模拟器（Stage 0）
机制（用户确认）：
  收盘后：满足 v3 条件 → 进票池（不在池/不持仓）；
  盘中监控池内票：价格进入 [1.07,1.15]×昨日视角的30日最高收盘 → 有空仓位+现金 → 按 prox 排序买入；
  买入即移出票池；卖出后若当日收盘仍满足选股条件 → 重新进池；
  出池（仅两条）：① 收盘 > 首次触发价×1.20；② 收盘 < 最近30日最高收盘×0.80。
  买卖执行：5min（107–115% 触发、14:30 止损、②a/②b 半仓、③a/③b 顶部确认），用 sim_minute 离线模拟整笔并排期。
  --daily：用日线近似（仅用于快速对照，不作最终结论）。
  资金：10 个等权仓位；size=min(cash, 昨收权益/10)；不计成本。
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
from sim_fast import sim_minute_fast

RANK_MODE="prox"      # prox / lowprox / random
TRIG_MODE="rolling"   # rolling / entry_fixed / first_fixed
WAIT_LIMIT=None       # 进池后允许等待的最大交易日数（None=无限）

def build_data(P):
    C=P["C"]; nd,nc=C.shape
    dfC=pd.DataFrame(C)
    top_prev=dfC.rolling(30).max().shift(1).values
    top30_exit=dfC.rolling(30).max().values
    prox_prev=(dfC.shift(1)/dfC.rolling(250).max().shift(1)).values
    base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["prox"]>=0.90)&(P["ret10"]<=0.08)&(P["turn30"]<=5)&(P["ret60"]>=0)&((C/np.roll(C,1,axis=0)-1)*100<9.8)
    return dict(top_prev=top_prev, top30_exit=top30_exit, prox_prev=prox_prev, mask=base)

def simulate(P, dd, start="2021-01-04", end=None, verbose=True, daily=False):
    dates=np.array(P["dates"]); nd=P["nd"]; O=P["O"]; H=P["H"]; L=P["L"]; C=P["C"]; codes=P["codes"]
    s=int(np.searchsorted(dates,start)); e=nd if end is None else int(np.searchsorted(dates,end))
    idx={d:i for i,d in enumerate(dates)}
    pool={}; pos={}; cash=1.0; buys=sells=0; eqs=[]; holds=[]; buy_log=[]; trades=[]
    for d in range(s,e):
        # 1) 到期的半仓/清仓事件
        for code,p in list(pos.items()):
            if p["half_day"]==d and not p["half_done"]:
                hv=p["shares"]*0.5*p["half_px"]; cash+=hv; p["half_cash"]=p.get("half_cash",0.0)+hv
                p["shares"]*=0.5; p["half_done"]=True
            if p["exit_day"]==d:
                proceeds=p.get("half_cash",0.0)+p["shares"]*p["exit_px"]
                cash+=p["shares"]*p["exit_px"]; holds.append(d-p["entry_day"])
                trades.append(dict(code=code, entry_day=p["entry_day"], exit_day=d,
                                   entry_px=p["entry_px"], size=p["size"], ret=proceeds/p["size"]-1,
                                   half=bool(p.get("half_cash",0.0)>0)))
                del pos[code]; sells+=1
        # 2) 买点监控（盘中，日线区间近似判定是否触达；实际成交用 5min）
        trigs=[]
        for code,p in pool.items():
            j=p["j"]
            if TRIG_MODE in ("entry_fixed","first_fixed") and p.get("top_fixed") is not None:
                tp=p["top_fixed"]
            else:
                tp=dd["top_prev"][d,j]
            if not np.isfinite(tp) or tp<=0: continue
            if np.isfinite(H[d,j]) and np.isfinite(L[d,j]) and H[d,j]>=1.07*tp and L[d,j]<=1.15*tp:
                if TRIG_MODE=="first_fixed" and p.get("top_fixed") is None:
                    p["top_fixed"]=float(tp)
                first=1.07*tp
                if p["first"] is None or first>p["first"]: p["first"]=first
                pv=dd["prox_prev"][d,j]
                if RANK_MODE=="lowprox": rk=-pv if np.isfinite(pv) else 9
                elif RANK_MODE=="random": rk=hash((code,d))%100000
                else: rk=pv if np.isfinite(pv) else -9
                trigs.append((rk,code,j,tp,d-p.get("entry_day",d)))
        trigs=[x for x in trigs if x[1] not in pos]
        if WAIT_LIMIT is not None: trigs=[x for x in trigs if x[4]<=WAIT_LIMIT]
        trigs.sort(key=lambda x:-x[0])
        free=10-len(pos)
        for rank,code,j,tp,wt in trigs:
            if free<=0: break
            if cash<=1e-12: break
            equity_prev=cash+sum(pp["shares"]*C[d-1,pp["j"]] for pp in pos.values() if np.isfinite(C[d-1,pp["j"]]))
            size=min(cash, equity_prev/10)
            if size<=1e-9: break
            if daily:
                trig=1.07*float(tp); cap=1.15*float(tp)
                if not np.isfinite(H[d,j]) or not np.isfinite(O[d,j]): continue
                entry=(max(float(O[d,j]),trig) if O[d,j]<=cap else cap)
                rd=M.sim_daily(P, dict(j=j, ed=d, entry=entry))
                if rd is None: continue
                pos[code]=dict(j=j,entry_day=d,shares=size/entry,entry_px=entry,size=size,half_cash=0.0,
                               half_day=None,half_px=None,half_done=False,
                               exit_day=idx[rd["xday"]],exit_px=entry*(1+rd["ret"]))
                cash-=size; free-=1; buys+=1; buy_log.append((dates[d],code,entry,size)); del pool[code]
                continue
            # 5min 模拟整笔交易
            sd=dates[d-1]; ed=dates[min(d+120, nd-1)]
            res=None
            for attempt in range(4):
                df=M.get5(code, sd, ed)
                if df is None or len(df)==0: break
                res=sim_minute_fast(P, dict(j=j, ed=d, top=float(tp)), df, float(P["C"][d-1,j]))
                if res is None: break
                last=str(df["date"].max())
                if res["xday"]==last and idx[last] < nd-1 and attempt<3:
                    ed=dates[min(idx[last]+121, nd-1)]; continue
                break
            if res is None: continue
            sh=size/res["entry_qfq"]
            pos[code]=dict(j=j,entry_day=d,shares=sh,entry_px=res["entry_qfq"],size=size,half_cash=0.0,
                           half_day=(idx[res["half_day"]] if res["half_day"] else None),half_px=res["half_px_qfq"],half_done=False,
                           exit_day=idx[res["xday"]],exit_px=res["exit_qfq"])
            cash-=size; free-=1; buys+=1; buy_log.append((dates[d],code,res["entry_qfq"],size)); del pool[code]
        # 3.5) 出池（收盘后，两条；买入之后用当日收盘判断，避免盘中前视）
        for code,p in list(pool.items()):
            j=p["j"]; c=C[d,j]
            if not np.isfinite(c): continue
            if p["first"] is not None and c>p["first"]*1.20: del pool[code]; continue
            tp=dd["top30_exit"][d,j]
            if np.isfinite(tp) and c<tp*0.80: del pool[code]
        # 4) 进池（收盘后，满足条件）
        m=dd["mask"][d]
        for j in np.where(m)[0]:
            code=codes[j]
            if code in pool or code in pos: continue
            pool[code]=dict(j=j,first=None,entry_day=d,
                            top_fixed=(float(dd["top30_exit"][d,j]) if TRIG_MODE=="entry_fixed" and np.isfinite(dd["top30_exit"][d,j]) else None))
        # 5) 估值
        val=sum(p["shares"]*C[d,p["j"]] for p in pos.values() if np.isfinite(C[d,p["j"]]))
        eqs.append((d,cash,val,cash+val,len(pos),len(pool)))
        assert cash>=-1e-6, f"negative cash day {d}"
    e_arr=np.array([x[3] for x in eqs]); eqd=pd.to_datetime([dates[x[0]] for x in eqs])
    r=pd.Series(e_arr).pct_change().dropna()
    tot=e_arr[-1]/1.0-1; n=len(e_arr); cagr=(e_arr[-1])**(252/n)-1
    mdd=float((pd.Series(e_arr)/pd.Series(e_arr).cummax()-1).min())
    sharpe=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan
    mon=pd.Series(e_arr,index=eqd).resample("ME").last().pct_change().dropna()
    tr=pd.DataFrame(trades)
    if len(tr):
        tr.to_csv(r"D:/about/learn/analysis/pool_trades.csv",index=False,encoding="utf-8-sig")
        wy=pd.to_datetime(dates[tr.exit_day]).year
        print(f"已实现交易 {len(tr)} 笔: 均值 {tr.ret.mean()*100:.2f}% 中位 {tr.ret.median()*100:.2f}% 胜率 {(tr.ret>0).mean()*100:.1f}%")
        print("  分年:", {int(y):f"{tr.ret[wy==y].mean()*100:+.1f}%(n={int((wy==y).sum())})" for y in sorted(set(wy))})
    if verbose:
        print(f"[{start}~{end}] 总收益 {tot*100:.1f}% CAGR {cagr*100:.1f}% MDD {mdd*100:.1f}% Sharpe {sharpe:.2f}")
        print(f"买入 {buys} 卖出 {sells} 平均持有 {np.mean(holds) if holds else np.nan:.1f}日 | 平均持仓 {np.mean([x[4] for x in eqs]):.1f} 平均池子 {np.mean([x[5] for x in eqs]):.0f}")
        print("分年:", {int(y):f"{(1+mon[mon.index.year==y]).prod()-1:+.1%}" for y in sorted(set(mon.index.year))})
    return dict(total=tot,cagr=cagr,mdd=mdd,sharpe=sharpe,buys=buys,sells=sells,monthly=mon,eq=pd.Series(e_arr,index=eqd))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",default="2021-01-04"); ap.add_argument("--end",default=None)
    ap.add_argument("--daily",action="store_true")
    ap.add_argument("--rank",default="prox",choices=["prox","lowprox","random"])
    ap.add_argument("--trig",default="rolling",choices=["rolling","entry_fixed","first_fixed"])
    ap.add_argument("--wait",type=int,default=None)
    a=ap.parse_args(); global RANK_MODE,WAIT_LIMIT,TRIG_MODE; RANK_MODE=a.rank; WAIT_LIMIT=a.wait; TRIG_MODE=a.trig
    t0=time.time(); P=M.build(); dd=build_data(P)
    import baostock as bs
    if not a.daily: bs.login()
    try:
        r=simulate(P,dd,a.start,a.end,daily=a.daily)
        r["eq"].to_csv(r"D:\about\learn\analysis\pool_equity.csv",encoding="utf-8-sig")
    finally:
        if not a.daily: bs.logout()
    print(f"[{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
