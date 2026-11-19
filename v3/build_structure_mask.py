# -*- coding: utf-8 -*-
"""build_structure_mask.py —— 生成 2021-2026 的“缠论结构候选”掩码（笔中枢）
条件：成交额>=2亿 & 清洗 & 非涨停 & C>=0.75×年高
      & 笔中枢存在 & 价格在中枢内或刚离开(0<距ZG<=1倍中枢宽度) & 中枢上沿 ZG>=0.85×年高
输出：D:/about/learn/analysis/structure_mask_2021.npz
"""
import os, sys, time, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
from chan_state import chan_state

def main():
    t0=time.time(); P=M.build()
    C=P["C"]; H=P["H"]; L=P["L"]; A=P["A"]; nd,nc=C.shape
    ret=C/np.roll(C,1,axis=0)-1
    yhigh=pd.DataFrame(C).rolling(250).max().values
    mask=np.zeros((nd,nc),bool)
    s=int(np.searchsorted(np.array(P["dates"]),"2021-01-04"))
    for t in range(s,nd):
        cheap=P["valid"][t]&P["clean"][t]&(A[t]>=2e8)&((ret[t]*100)<9.8)&(C[t]>=0.75*yhigh[t])
        for j in np.where(cheap)[0]:
            st=chan_state(H[:,j],L[:,j],C[:,j],t)
            if not st.get("zone"): continue
            zg=st["zg"]; zd=st["zd"]; w=st["width"]; price=float(C[t,j])
            if not np.isfinite(zg) or zg<0.85*yhigh[t,j]: continue
            if price<zd: continue
            if price>zg:
                if not (w and w>0): continue
                if (price/zg-1)/w>1.0: continue
            mask[t,j]=True
        if (t-s)%100==0: print(f"  {t-s}/{nd-s} [{time.time()-t0:.0f}s] 累计 {int(mask.sum())}",flush=True)
    np.savez_compressed(r"D:/about/learn/analysis/structure_mask_2021.npz", mask=mask)
    print(f"done [{time.time()-t0:.0f}s] mask={int(mask.sum())} days={nd-s}")

if __name__=="__main__": main()
