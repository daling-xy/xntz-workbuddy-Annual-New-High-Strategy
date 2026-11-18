# -*- coding: utf-8 -*-
"""min5_variants.py —— 在 5min 口径下校准 v3 各变体（V1/V2/V3/V4）
V4 交易太多（8,554 笔），抽样 200 笔；其余全量。复用 data/min_calib 断点缓存。
"""
import os, random, time, sys
import numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"

def run_variant(P, v, cap=None):
    tr=[x for x in M.make_trades(P, v) if M.bs_sym(x["code"])]
    n_all=len(tr)
    if cap and len(tr)>cap:
        random.seed(7); random.shuffle(tr); tr=tr[:cap]
    out=[]
    for x in tr:
        ed=x["ed"]; j=x["j"]; d0=P["dates"][ed-1]; d1=P["dates"][min(ed+80, P["nd"]-1)]
        df=M.get5(x["code"], d0, d1)
        if df is None or len(df)==0: continue
        rm=M.sim_minute(P,x,df,float(P["C"][ed-1,j]))
        if rm is None: continue
        rd=M.sim_daily(P,x)
        out.append(dict(variant=v, code=x["code"], ed=P["dates"][ed], top=x["top"],
                        entry_diff=(rm["entry_qfq"]/rd["entry"]-1)*100,
                        ret_min=rm["ret"], ret_daily=rd["ret"], ret_diff=(rm["ret"]-rd["ret"])*100,
                        half_min=rm["half"], half_daily=rd["half"]))
    d=pd.DataFrame(out)
    print(f"\n=== {v}: 交易 {n_all} 笔，校准 {len(d)} 笔 ===", flush=True)
    if len(d):
        print(f"  5min: 平均 {d.ret_min.mean()*100:6.2f}% 中位 {d.ret_min.median()*100:6.2f}% 胜率 {(d.ret_min>0).mean()*100:5.1f}% 半仓 {d.half_min.mean()*100:5.1f}%")
        print(f"  日线: 平均 {d.ret_daily.mean()*100:6.2f}% 中位 {d.ret_daily.median()*100:6.2f}% 胜率 {(d.ret_daily>0).mean()*100:5.1f}% 半仓 {d.half_daily.mean()*100:5.1f}%")
        print(f"  入场价差 {d.entry_diff.mean():.3f}% | 收益差 {d.ret_diff.mean():.2f}pp 中位 {d.ret_diff.median():.2f}pp")
    return d

def main():
    t0=time.time()
    P=M.build()
    import baostock as bs
    bs.login()
    allr=[]
    for v,cap in [("V1",None),("V2",None),("V3",None),("V4",200)]:
        d=run_variant(P,v,cap); allr.append(d)
        if len(d): d.to_csv(os.path.join(LEARN,"analysis",f"min5_variant_{v}.csv"),index=False,encoding="utf-8-sig")
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    bs.logout()
    R=pd.concat(allr,ignore_index=True)
    R.to_csv(os.path.join(LEARN,"analysis","min5_variants_all.csv"),index=False,encoding="utf-8-sig")
    print("\n=== 汇总（5min 口径）===")
    for v in ["V1","V2","V3","V4"]:
        d=R[R.variant==v]
        if len(d)==0: continue
        print(f"{v}: n={len(d):4d} 5min平均 {d.ret_min.mean()*100:6.2f}% 中位 {d.ret_min.median()*100:6.2f}% 胜率 {(d.ret_min>0).mean()*100:5.1f}% 半仓 {d.half_min.mean()*100:5.1f}%")
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__":
    main()
