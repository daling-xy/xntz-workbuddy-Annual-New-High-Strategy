# -*- coding: utf-8 -*-
"""structure_first.py —— 结构优先选股（缠论笔中枢），2023-2026，买卖点与 v3 相同
条件：全A & 当日成交额>=2亿 & 清洗 & 非涨停
      & 存在笔中枢（包含处理→分型→笔→3笔重叠）
      & 价格在中枢内 或 刚离开中枢（0<距ZG<=1倍中枢宽度）
      & 中枢上沿 ZG >= 0.85×250日最高收盘
买入：t+1/t+2 任一交易日 high>=1.07×30日最高收盘 且 open<=1.15×箱顶 → 成交价=max(open,1.07×箱顶)
输出：analysis/structure_first_trades_2023.csv
"""
import os, sys, time, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
from chan_state import chan_state
LEARN=r"D:\about\learn"

def main():
    t0=time.time(); P=M.build()
    dates=np.array(P["dates"]); nd=P["nd"]
    C=P["C"]; O=P["O"]; H=P["H"]; L=P["L"]; A=P["A"]
    ret=C/np.roll(C,1,axis=0)-1
    yhigh=pd.DataFrame(C).rolling(250).max().values
    hi30=pd.DataFrame(C).rolling(30).max().values
    valid=P["valid"]; clean=P["clean"]
    i0=int(np.searchsorted(dates,"2023-01-01")); i1=min(nd-3, int(np.searchsorted(dates,"2026-10-22")))
    trades=[]; cand=0
    for t in range(i0, i1+1):
        m=valid[t]&clean[t]&(A[t]>=2e8)&((ret[t]*100)<9.8)&(C[t]>=0.70*yhigh[t])
        idx=np.where(m)[0]
        for j in idx:
            st=chan_state(H[:,j],L[:,j],C[:,j],t)
            if not st.get("zone"): continue
            zg=st["zg"]; zd=st["zd"]; w=st["width"]; price=float(C[t,j])
            if not (zg>=0.85*yhigh[t,j]): continue
            if price<zd: continue
            if price>zg:
                nd_=(price/zg-1)/w if (w and w>0) else 999
                if nd_>1.0: continue
            cand+=1
            top=float(hi30[t,j])
            if not np.isfinite(top) or top<=0: continue
            trig=top*1.07; cap=top*1.15; entry=None; ed=None
            for d in (t+1,t+2):
                if d>=nd: break
                if np.isfinite(H[d,j]) and H[d,j]>=trig and np.isfinite(O[d,j]) and O[d,j]<=cap:
                    entry=max(float(O[d,j]),trig); ed=d; break
            if entry is None: continue
            trades.append(dict(code=P["codes"][j], t=dates[t], ed=dates[ed], top=top, entry=entry,
                               zg=zg, zd=zd, width=w, pos=st["pos"], nd=st.get("norm_dist")))
        if (t-i0)%100==0:
            print(f"  {t-i0}/{i1-i0+1} days [{time.time()-t0:.0f}s] candidates={cand} trades={len(trades)}",flush=True)
    d=pd.DataFrame(trades)
    out=os.path.join(LEARN,"analysis","structure_first_trades_2023.csv")
    d.to_csv(out,index=False,encoding="utf-8-sig")
    print(f"done [{time.time()-t0:.0f}s] candidates={cand} trades={len(trades)}")
    if len(d):
        d["y"]=pd.to_datetime(d["ed"]).dt.year
        print("trades by year:", d.y.value_counts().sort_index().to_dict())
        print("unique stocks:", d.code.nunique())
    print("输出:",out)

if __name__=="__main__":
    main()
