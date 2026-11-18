# -*- coding: utf-8 -*-
"""chan_state.py —— 按原著口径重写：中枢序列 + 第三类买点 + 离开中枢力度
原著：教你炒股票18（中枢=前三个完成次级别走势重叠）、20（延伸/新中枢/三类买卖点）、33（延伸≤5段）。
工程近似：包含处理→分型→笔→以笔近似次级别走势；中枢用前三个连续笔重叠，后续笔与区间有重叠算延伸，离开后产生新中枢。
用法：python v3/chan_state.py
"""
import os, sys, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
from chan_center import merge_bars, fractals, build_bi

def chan_state(h,l,c,t,window=250):
    lo=max(0,t-window+1)
    mb=merge_bars(h[lo:t+1], l[lo:t+1]); fr=fractals(mb); bi=build_bi(mb,fr)
    out=dict(n_bi=len(bi))
    if len(bi)<3: return out
    zones=[]; i=0
    def rng(s): return (min(s["start"],s["end"]), max(s["start"],s["end"]))
    while i+2 < len(bi):
        segs=bi[i:i+3]
        hs=[rng(s)[1] for s in segs]; ls=[rng(s)[0] for s in segs]
        zg=min(hs); zd=max(ls)
        if zd>zg: i+=1; continue
        j=i+3
        while j<len(bi):
            low,high=rng(bi[j])
            if low<=zg and high>=zd: j+=1
            else: break
        zones.append(dict(zg=zg,zd=zd,start=i,end=j-1)); i=j
    if not zones: return out
    z=zones[-1]; price=float(c[t]); width=(z["zg"]-z["zd"])/z["zd"] if z["zd"]>0 else None
    out.update(zone=True, zg=z["zg"], zd=z["zd"], width=width, n_zone=len(zones),
               pos=("above" if price>z["zg"] else ("below" if price<z["zd"] else "inside")))
    # 第三类买点：离开中枢（某笔高点>ZG），下一笔回抽低点>ZG，且未被拉回中枢/跌破ZD
    tb=False; broken=False; j=z["end"]+1
    while j < len(bi)-1:
        low,high=rng(bi[j])
        if high>z["zg"]:
            nlow=rng(bi[j+1])[0]
            if nlow>z["zg"]: tb=True; break
            if nlow<z["zd"]: broken=True; break
            break   # 回抽进入中枢 → 三类买点不成立
        if low<z["zd"]: broken=True; break
        j+=1
    out["third_buy"]=bool(tb and price>z["zg"] and not broken)
    out["zone_broken"]=broken
    out["dev_zg"]=price/z["zg"]-1
    out["norm_dist"]=(out["dev_zg"]/width) if (width and width>0) else None
    return out

def main():
    P=M.build()
    reps=pd.read_csv(r"D:\about\learn\analysis\min5_calib_trades_all.csv",dtype={"code":str})
    reps["ed"]=pd.to_datetime(reps["ed"]); reps["m"]=reps.ed.dt.to_period("M"); reps["y"]=reps.ed.dt.year
    tr={(x["code"], P["dates"][x["ed"]]):(x["j"],x["t"]) for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])}
    rows=[]
    for _,r in reps.iterrows():
        key=(r["code"], r["ed"].strftime("%Y-%m-%d"))
        if key not in tr: continue
        j,t=tr[key]; s=chan_state(P["H"][:,j],P["L"][:,j],P["C"][:,j],t)
        rows.append(dict(code=r["code"], ed=r["ed"], ret=r["ret_min"], m=r["m"], y=r["y"],
                         zone=s.get("zone",False), width=s.get("width"), pos=s.get("pos"),
                         third=s.get("third_buy",False), broken=s.get("zone_broken",False),
                         dev_zg=s.get("dev_zg"), nd=s.get("norm_dist"), n_zone=s.get("n_zone")))
    d=pd.DataFrame(rows); print("n",len(d),"有中枢",int(d.zone.sum()),"三类买点",int(d.third.sum()))
    print("宽度分位:", d.width.dropna().quantile([0.1,0.5,0.9]).round(4).to_dict())
    rng=np.random.default_rng(42); allm=d.groupby("m").ret.mean()
    def stat(lab,m):
        s=d[m]
        if len(s)==0: print(f"{lab:30s} n=0"); return
        print(f"{lab:30s} n={len(s):4d} 平均 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 大亏 {(s.ret<=-0.20).mean()*100:4.1f}%")
    def boot(m,label,half=None):
        x=d[m] if half is None else d[m&(d.y>=half[0])&(d.y<=half[1])]
        base=d[(d.y>=half[0])&(d.y<=half[1])] if half else d
        am=x.groupby("m").ret.mean(); bm=base.groupby("m").ret.mean()
        df=pd.concat([am,bm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
        if len(diff)<5: print(f"  {label:22s} 月数不足"); return
        bs=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)]
        lo,hi=np.percentile(bs,[5,95]); print(f"  {label:22s} {diff.mean()*100:+6.2f}pp CI90 [{lo*100:+.2f},{hi*100:+.2f}] 月数{len(diff)}")
    print("\n=== 中枢状态分组（5min 收益）===")
    stat("全部", d.ret.notna()); stat("无中枢", ~d.zone)
    stat("三类买点状态", d.third)
    stat("价格在ZG上方", d.pos=="above"); stat("价格在中枢内", d.pos=="inside"); stat("价格在ZD下方", d.pos=="below")
    for lo,hi,lab in [(-9,0,"距ZG<=0(内/下)"),(0,0.5,"距ZG 0~0.5倍宽"),(0.5,1,"0.5~1倍宽"),(1,2,"1~2倍宽"),(2,99,">2倍宽")]:
        stat(lab, d.nd.notna()&(d.nd>lo)&(d.nd<=hi))
    print("\n=== 月度块 bootstrap（vs 全部）===")
    for lab,m in [("三类买点",d.third),("ZG上方",d.pos=="above"),("中枢内",d.pos=="inside"),
                  ("0<距ZG<=1倍宽",d.nd.notna()&(d.nd>0)&(d.nd<=1)),("距ZG>1倍宽",d.nd>1)]:
        boot(m,lab)
    print("\n=== 双向样本外 ===")
    for half in [(2019,2022),(2023,2026)]:
        print(f"-- {half[0]}-{half[1]} --")
        boot(d.third,"三类买点",half); boot(d.nd.notna()&(d.nd>0)&(d.nd<=1),"0<距ZG<=1倍宽",half)
    d.to_csv(r"D:\about\learn\analysis\chan_state_features.csv",index=False,encoding="utf-8-sig")
    print("\n输出：analysis/chan_state_features.csv")
if __name__=="__main__":
    main()
