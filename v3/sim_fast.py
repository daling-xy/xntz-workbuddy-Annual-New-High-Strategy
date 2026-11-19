# -*- coding: utf-8 -*-
import numpy as np, pandas as pd

def sim_minute_fast(P, tr, df, qfq_close_prevd):
    """与 min5_calib.sim_minute 同样的规则，但用 numpy 预分组，避免逐日 pandas 过滤。"""
    j=tr["j"]; ed=tr["ed"]; top=tr["top"]; dts=P["dates"]; nd=P["nd"]
    dates=df["date"].to_numpy(); hhmm=df["hhmm"].to_numpy()
    O=df["open"].to_numpy(); H=df["high"].to_numpy(); L=df["low"].to_numpy(); Cc=df["close"].to_numpy(); Vv=df["volume"].to_numpy()
    order=np.lexsort((hhmm,dates)); dates=dates[order]; hhmm=hhmm[order]; O=O[order]; H=H[order]; L=L[order]; Cc=Cc[order]; Vv=Vv[order]
    ud=np.unique(dates); starts=np.searchsorted(dates,ud); ends=np.r_[starts[1:],len(dates)]
    prev_idx=np.searchsorted(ud, np.array([dts[ed-1]]))  # 前一日在 df 内的位置
    # 找 ed 前一日：df 中 < dts[ed] 的最后一天
    pos_ed=int(np.searchsorted(ud, np.array([dts[ed]]))[0])
    if pos_ed<=0: return None
    pday_i=pos_ed-1
    pclose=float(Cc[ends[pday_i]-1]); pvol=float(Vv[starts[pday_i]:ends[pday_i]].sum())
    factor=pclose/qfq_close_prevd if qfq_close_prevd>0 else np.nan
    if not np.isfinite(factor): return None
    trig=1.07*top*factor; cap=1.15*top*factor
    entry=None; eday_i=None; ent_hhmm=None
    for k in (0,1):
        di=pos_ed+k
        if di>=len(ud): break
        s,e=starts[di],ends[di]
        hi=H[s:e]; lo=L[s:e]; op=O[s:e]
        ok=(hi>=trig)&(lo<=cap)
        if ok.any():
            i=int(np.argmax(ok)); entry=float(min(max(op[i],trig),cap)); eday_i=di; ent_hhmm=int(hhmm[s+i]); break
    if entry is None: return None
    half=None; half_px=None; half_day=None; half_done=False; xp=None; xday=None
    for di in range(eday_i,len(ud)):
        if di==0: 
            prevc=pclose; prevv=pvol
        else:
            prevc=float(Cc[ends[di-1]-1]); prevv=float(Vv[starts[di-1]:ends[di-1]].sum())
        s,e=starts[di],ends[di]
        h=hhmm[s:e]; c=Cc[s:e]; v=Vv[s:e]
        # ②a
        for i in range(len(h)):
            if di==eday_i and h[i]<=ent_hhmm: continue
            prof=float(c[i])/entry-1
            if prof>0.30 and float(c[i])/prevc-1>=0.09:
                if not half_done: half=float(c[i]); half_px=half; half_day=str(ud[di]); half_done=True
                else: xp=float(c[i]); xday=str(ud[di]); break
        if xp is not None: break
        m=h<=1030
        if m.any():
            cum=float(v[m].sum()); px=float(c[m][-1])
            if prevv>0 and cum>=0.7*prevv and px/prevc-1<-0.07: xp=px; xday=str(ud[di]); break
        m=h<=1430
        if m.any():
            cum=float(v[m].sum()); px=float(c[m][-1])
            if prevv>0 and cum>=prevv and px/prevc-1<-0.07: xp=px; xday=str(ud[di]); break
            if px<entry*0.87: xp=px; xday=str(ud[di]); break
        m=h<=1445
        if m.any():
            px1445=float(c[m][-1]); prof=px1445/entry-1
            if prof>0.30 and float(c[-1])/prevc-1<=0.03:
                if di>0:
                    pdp=float(Cc[ends[di-2]-1]); pdc=float(Cc[ends[di-1]-1])
                    if pdp/pdc-1<=0.03:
                        if not half_done: half=px1445; half_px=half; half_day=str(ud[di]); half_done=True
                        else: xp=px1445; xday=str(ud[di]); break
    if xp is None:
        xp=float(Cc[-1]); xday=str(ud[-1])
    r=(0.5*half+0.5*xp)/entry-1 if half is not None else xp/entry-1
    return dict(ret=float(r), entry=float(entry), entry_qfq=float(entry/factor), xp=float(xp), xday=xday,
                half=half is not None, half_day=half_day, half_px_qfq=(float(half_px/factor) if half_px else None),
                exit_qfq=float(xp/factor), factor=float(factor))
