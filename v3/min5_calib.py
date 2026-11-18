# -*- coding: utf-8 -*-
"""min5_calib.py —— 用 baostock 5 分钟数据校准 v3 真实交易口径回测
只校准三件事：①107% 触发的实际成交价；②14:30 -13% 止损；③10:30 前量能 70% 的顶部确认
其余卖出（②a/②b/③b）仍按日线近似，便于对比
用法：python v3/min5_calib.py --n 5      # 小样本校验
      python v3/min5_calib.py --n 120    # 正式样本
"""
import argparse, os, pickle, random, time
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARN = r"D:\about\learn"
C5 = os.path.join(LEARN, "data", "min_calib"); os.makedirs(C5, exist_ok=True)
FULL = os.path.join(LEARN, "data", "full_a_share_v2.csv")
ID = os.path.join(BASE, "v3", "data", "deliver_identity_table.csv")
BOARD = os.path.join(BASE, "v3", "data", "board_map.csv")
KL5 = os.path.join(BASE, "cache", "kline_long2.pkl")

def load_panel():
    kl = pickle.load(open(KL5, "rb"))
    codes = sorted(kl.keys()); dates = sorted({b["date"] for v in kl.values() for b in v})
    di = {d: i for i, d in enumerate(dates)}; nd, nc = len(dates), len(codes)
    O=np.full((nd,nc),np.nan,np.float32); H=np.full((nd,nc),np.nan,np.float32)
    L=np.full((nd,nc),np.nan,np.float32); C=np.full((nd,nc),np.nan,np.float32)
    V=np.full((nd,nc),np.nan,np.float32); A=np.full((nd,nc),np.nan,np.float32); T=np.full((nd,nc),np.nan,np.float32)
    for j,c in enumerate(codes):
        for b in kl[c]:
            i=di[b["date"]]; O[i,j]=b["open"]; H[i,j]=b["high"]; L[i,j]=b["low"]; C[i,j]=b["close"]; V[i,j]=b["volume"]; A[i,j]=b["amount"]; T[i,j]=b.get("turnover",np.nan)
    return codes, dates, O,H,L,C,V,A,T

def build():
    codes, dates, O,H,L,C,V,A,T = load_panel()
    nd,nc = C.shape
    ret=C/np.roll(C,1,axis=0)-1
    vol20=(pd.DataFrame(ret).rolling(20).std().values)*np.sqrt(252)
    vh=pd.DataFrame(vol20).rolling(250).mean().values
    vr=vol20/np.where(vh>0,vh,np.nan)
    prox=C/np.where(pd.DataFrame(C).rolling(250).max().values>0,pd.DataFrame(C).rolling(250).max().values,np.nan)
    hi30=pd.DataFrame(C).rolling(30).max().values
    ret10=C/np.roll(C,10,axis=0)-1; ret30=C/np.roll(C,30,axis=0)-1; ret60=C/np.roll(C,60,axis=0)-1
    turn30=pd.DataFrame(T).rolling(30).mean().values
    idn=pd.read_csv(ID,dtype={"code":str}).set_index("code")
    name=np.array([idn.loc[x,"name"] if x in idn.index else "" for x in codes])
    st=np.array([("ST" in str(n)) or ("退" in str(n)) for n in name])
    bm=pd.read_csv(BOARD,dtype={"code":str},usecols=["code","board"]).drop_duplicates("code").set_index("code")["board"]
    BLACK={"银行Ⅱ","基础建设","房屋建设Ⅱ","专业工程","工程咨询服务Ⅱ","铁路公路","物流","航空机场","通信服务"}
    bad=np.array([(isinstance(b,str) and b in BLACK) for b in bm.reindex(codes).values]); clean=~st&~bad
    valid=np.isfinite(C)
    return dict(codes=codes,dates=dates,O=O,H=H,L=L,C=C,V=V,A=A,T=T,nd=nd,nc=nc,
        vol20=vol20,vr=vr,prox=prox,hi30=hi30,ret10=ret10,ret30=ret30,ret60=ret60,turn30=turn30,
        name=name,clean=clean,valid=valid)

def make_trades(P):
    C=P["C"]; nd=P["nd"]; nc=P["nc"]; i0=int(np.searchsorted(np.array(P["dates"]),"2019-01-01"))
    base=P["valid"]&P["clean"]&(P["A"]>=2e8)&(P["prox"]>=0.90)&(P["ret10"]<=0.08)&(P["turn30"]<=5)
    base&= (P["vr"]<=1.0)&(P["vol20"]<=0.35)&(P["ret60"]>=0)
    base[:i0,:]=False
    H=P["H"]; L=P["L"]; V=P["V"]; hi30=P["hi30"]
    ii,jj=np.where(base); trades=[]
    for k in range(len(ii)):
        t=ii[k]; j=jj[k]; top=hi30[t,j]
        if not np.isfinite(top) or top<=0: continue
        trig=top*1.07; cap=top*1.15; entry=None; ed=None
        for d in (t+1,t+2):
            if d>=nd: break
            if np.isfinite(H[d,j]) and H[d,j]>=trig and np.isfinite(P["O"][d,j]) and P["O"][d,j]<=cap:
                entry=max(P["O"][d,j],trig); ed=d; break
        if entry is None: continue
        trades.append(dict(j=j, code=P["codes"][j], t=t, ed=ed, top=float(top), entry=float(entry)))
    return trades

def bs_sym(code):
    if code.startswith(("6","68")): return "sh."+code
    if code.startswith(("0","3")): return "sz."+code
    return None

def get5(code, sd, ed):
    f=os.path.join(C5, f"{code}_{sd}_{ed}.csv")
    if os.path.exists(f):
        try: return pd.read_csv(f)
        except: pass
    import baostock as bs
    sym=bs_sym(code)
    if sym is None: return None
    rs=bs.query_history_k_data_plus(sym,"date,time,open,high,low,close,volume",start_date=sd,end_date=ed,frequency="5",adjustflag="1")
    if rs.error_code!="0": return None
    rows=[]
    while rs.next(): rows.append(rs.get_row_data())
    if not rows: return None
    d=pd.DataFrame(rows,columns=["date","time","open","high","low","close","volume"])
    for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["hhmm"]=d["time"].str[8:12].astype(int)
    d.to_csv(f,index=False)
    return d

def sim_minute(P, tr, df, qfq_close_prevd, verbose=False):
    """用 5min 数据重放一笔交易：返回 (ret, entry, exit_reason, exit_day)"""
    j=tr["j"]; ed=tr["ed"]; top=tr["top"]
    dts=P["dates"]; nd=P["nd"]
    # 前一日收盘（5min 最后 bar）与 qfq 收盘 → 换算因子
    days=sorted(df["date"].unique())
    pre=[d for d in days if d<dts[ed]]
    if not pre: return None
    pday=pre[-1]; pclose=float(df[df.date==pday].iloc[-1]["close"]); pvol=float(df[df.date==pday]["volume"].sum())
    factor=pclose/qfq_close_prevd if qfq_close_prevd>0 else np.nan
    if not np.isfinite(factor): return None
    trig=1.07*top*factor; cap=1.15*top*factor
    # 找触发 bar（ed 和 ed+1 两个交易日）
    entry=None; eday=None; ebar=None
    for dd in (0,1):
        d=dts[ed+dd] if ed+dd<nd else None
        if d is None or d not in days: continue
        b=df[df.date==d].sort_values("hhmm")
        for idx,r in b.iterrows():
            if r["high"]>=trig and r["low"]<=cap:
                entry=float(min(max(r["open"],trig),cap)); eday=d; ebar=idx; break
        if entry is not None: break
    if entry is None: return None
    half=None; half_done=False; xr=None; xp=None; xday=None
    # 逐日重放
    for d in [x for x in days if x>=eday]:
        b=df[df.date==d].sort_values("hhmm")
        if len(b)==0: continue
        prevc = pclose if d==pday else float(df[df.date<d].iloc[-1]["close"]) if False else None
        # 当日参考前收
        prev_rows=df[df.date<d]
        prevc=float(prev_rows.iloc[-1]["close"]) if len(prev_rows) else pclose
        prevv=float(prev_rows.iloc[-1]["volume"]) if False else float(prev_rows[prev_rows.date==prev_rows.iloc[-1]["date"]]["volume"].sum())
        # ②a：盘中涨幅>=9% 且盈利>30% → 半仓/全卖
        for _,r in b.iterrows():
            if d==eday and r["hhmm"]<=int(df.loc[ebar,"hhmm"]): continue
            prof=float(r["close"])/entry-1
            if prof>0.30 and float(r["close"])/prevc-1>=0.09:
                if not half_done: half=float(r["close"]); half_done=True
                else: xp=float(r["close"]); xday=d; break
        if xp is not None: break
        # ③a：10:30 前量>=前日 70%，且跌幅>7%
        b1030=b[b.hhmm<=1030]
        if len(b1030):
            cum=float(b1030["volume"].sum()); px=float(b1030.iloc[-1]["close"])
            if prevv>0 and cum>=0.7*prevv and px/prevc-1<-0.07:
                xp=px; xday=d; break
        # ③b：14:30 累计量>=前日全日量，且跌幅>7%
        b1430=b[b.hhmm<=1430]
        if len(b1430):
            cum=float(b1430["volume"].sum()); px=float(b1430.iloc[-1]["close"])
            if prevv>0 and cum>=prevv and px/prevc-1<-0.07:
                xp=px; xday=d; break
        # ①止损：14:30 价格 < entry*0.87
        if len(b1430):
            px=float(b1430.iloc[-1]["close"])
            if px<entry*0.87:
                xp=px; xday=d; break
        # ②b：收盘口径（日线近似）：盈利>30% 且连续两日涨幅<=3% → 14:45 卖一半/全卖
        b1445=b[b.hhmm<=1445]
        if len(b1445):
            px1445=float(b1445.iloc[-1]["close"]); prof=px1445/entry-1
            if prof>0.30 and float(b["close"].iloc[-1])/prevc-1<=0.03:
                prev_rows2=df[df.date<d]
                if len(prev_rows2):
                    pd_day=prev_rows2.iloc[-1]["date"]; pdp=float(prev_rows2[prev_rows2.date==pd_day].iloc[-1]["close"])
                    if pdp/prevc-1<=0.03:
                        if not half_done: half=px1445; half_done=True
                        else: xp=px1445; xday=d; break
    if xp is None:
        last=df[df.date>=eday].iloc[-1]
        xp=float(last["close"]); xday=str(last["date"])
    r=(0.5*half+0.5*xp)/entry-1 if half is not None else xp/entry-1
    return dict(ret=float(r), entry=float(entry), entry_qfq=float(entry/factor), xp=float(xp), xday=xday, half=half is not None)

def sim_daily(P, tr):
    """日线近似：同一套规则（用于对比）"""
    j=tr["j"]; ed=tr["ed"]; entry=tr["entry"]; nd=P["nd"]
    C=P["C"]; L=P["L"]; V=P["V"]; ret=P["C"]/np.roll(P["C"],1,axis=0)-1
    half=None; half_done=False; xp=None; xd=None
    for d in range(ed+1, nd):
        if not np.isfinite(C[d,j]): continue
        prof=C[d,j]/entry-1; pct=ret[d,j]; pvv=V[d-1,j]
        if np.isfinite(L[d,j]) and L[d,j]<=entry*0.87: xp=entry*0.87; xd=d; break
        if np.isfinite(pct) and pct<-0.07 and np.isfinite(V[d,j]) and np.isfinite(pvv) and pvv>0 and V[d,j]>=0.7*pvv: xp=C[d,j]; xd=d; break
        if prof>0.30:
            sh=(np.isfinite(pct) and pct>=0.09) or (d-1>=ed+1 and np.isfinite(ret[d-1,j]) and ret[d-1,j]<=0.03 and pct<=0.03)
            if sh:
                if not half_done: half=C[d,j]; half_done=True
                else: xp=C[d,j]; xd=d; break
    if xp is None: xd=nd-1; xp=C[xd,j]
    r=(0.5*half+0.5*xp)/entry-1 if half is not None else xp/entry-1
    return dict(ret=float(r), entry=float(entry), xp=float(xp), xday=str(P["dates"][xd]), half=half is not None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--all", action="store_true", help="全量：所有 v3 交易（不分年份/不抽样）")
    ap.add_argument("--seed", type=int, default=7)
    args=ap.parse_args()
    t0=time.time()
    P=build(); trades=make_trades(P); print(f"v3 trades: {len(trades)}",flush=True)
    # 取 2021 年以后、有 5min 符号的样本
    tr=[x for x in trades if bs_sym(x["code"])]
    if args.all:
        sample=tr
        print(f"FULL: {len(sample)} 笔（全部 v3 交易）",flush=True)
    else:
        tr=[x for x in tr if P["dates"][x["ed"]]>="2021-01-01"]
        random.seed(args.seed); random.shuffle(tr); sample=tr[:args.n]
        print(f"sample: {len(sample)} 笔（2021 年后）",flush=True)
    import baostock as bs
    lg=bs.login()
    out=[]
    for k,x in enumerate(sample):
        j=x["j"]; ed=x["ed"]; top=x["top"]
        d0=P["dates"][ed-1]; d1=min(P["dates"][ed+80], P["dates"][P["nd"]-1])
        df=get5(x["code"], d0, d1)
        if df is None or len(df)==0: continue
        qfq_prev=float(P["C"][ed-1,j])
        res=sim_minute(P,x,df,qfq_prev)
        if res is None: continue
        rd=sim_daily(P,x)
        out.append(dict(code=x["code"], ed=P["dates"][ed], top=top,
                        entry_min=res["entry"], entry_daily=rd["entry"], entry_diff=(res["entry_qfq"]/rd["entry"]-1)*100,
                        ret_min=res["ret"], ret_daily=rd["ret"], ret_diff=(res["ret"]-rd["ret"])*100,
                        half_min=res["half"], half_daily=rd["half"]))
        if (k+1)%20==0: print(f"  {k+1}/{len(sample)} done [{time.time()-t0:.0f}s]",flush=True)
    bs.logout()
    r=pd.DataFrame(out)
    print(f"\n完成 {len(r)} 笔（{time.time()-t0:.0f}s）")
    if len(r):
        print(f"5min 口径：平均 {r.ret_min.mean()*100:.2f}% 中位 {r.ret_min.median()*100:.2f}% 胜率 {(r.ret_min>0).mean()*100:.1f}% 半仓 {(r.half_min).mean()*100:.1f}%")
        print(f"日线口径（同批）：平均 {r.ret_daily.mean()*100:.2f}% 中位 {r.ret_daily.median()*100:.2f}% 胜率 {(r.ret_daily>0).mean()*100:.1f}% 半仓 {(r.half_daily).mean()*100:.1f}%")
        print(f"入场价差（5min/日线-1）：平均 {r.entry_diff.mean():.2f}% 中位 {r.entry_diff.median():.2f}%")
        print(f"收益差（5min-日线）：平均 {r.ret_diff.mean():.2f}pp 中位 {r.ret_diff.median():.2f}pp")
    fn = "min5_calib_trades_all.csv" if args.all else "min5_calib_trades.csv"
    r.to_csv(os.path.join(LEARN,"analysis",fn),index=False,encoding="utf-8-sig")
    print(f"输出：analysis/{fn}")

if __name__=="__main__":
    main()
