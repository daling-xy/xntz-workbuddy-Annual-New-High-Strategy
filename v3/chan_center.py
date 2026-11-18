# -*- coding: utf-8 -*-
"""chan_center.py —— 按缠论原著口径复刻“笔中枢”（工程近似）
原著依据（缠中说禅博客）：
  中枢定义（教你炒股票18）：某级别走势类型中，被至少三个连续次级别走势类型所重叠的部分；
  区间 = (max(a2,b2,c2), min(a1,b1,c1))，A/B/C 为前三个完成的次级别走势类型。
  延伸/新中枢（第20课）、延伸段数上限（第33课：一般不超过5段，6段则构成更大级别中枢）。
本实现：K线包含处理 → 分型 → 笔（顶底交替、间隔≥4根合并K线）→ 以“笔”近似次级别走势 → 最近3笔重叠为中枢。
用法：python v3/chan_center.py --check 600519 2026-10-22
      python v3/chan_center.py --eval
"""
import argparse, os, sys, numpy as np, pandas as pd
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M

def merge_bars(H,L):
    """K线包含处理：返回合并K线列表 [{h,l,i}]（i=原索引）"""
    out=[]
    for k in range(len(H)):
        out.append(dict(h=float(H[k]), l=float(L[k]), i=k))
        while len(out)>=2:
            a,b=out[-2],out[-1]
            incl=(a["h"]>=b["h"] and a["l"]<=b["l"]) or (a["h"]<=b["h"] and a["l"]>=b["l"])
            if not incl: break
            up = (out[-3]["h"]<a["h"]) if len(out)>=3 else (b["h"]>=a["h"])
            if up:
                nh=max(a["h"],b["h"]); nl=max(a["l"],b["l"])
            else:
                nh=min(a["h"],b["h"]); nl=min(a["l"],b["l"])
            out[-2]=dict(h=nh,l=nl,i=b["i"]); out.pop()
    return out

def fractals(mb):
    """顶/底分型：返回 [(idx_in_mb, 'top'/'bottom', price)]"""
    fr=[]
    for i in range(1,len(mb)-1):
        a,b,c=mb[i-1],mb[i],mb[i+1]
        if b["h"]>a["h"] and b["h"]>c["h"] and b["l"]>a["l"] and b["l"]>c["l"]:
            fr.append((i,"top",b["h"]))
        elif b["l"]<a["l"] and b["l"]<c["l"] and b["h"]<a["h"] and b["h"]<c["h"]:
            fr.append((i,"bottom",b["l"]))
    return fr

def build_bi(mb, fr):
    """笔：顶底交替；顶底分型中心间隔≥4根合并K线；同类型取更极端者"""
    bi=[]; last=None
    for idx,typ,price in fr:
        if last is None:
            last=(idx,typ,price); continue
        if typ==last[1]:
            if (typ=="top" and price>last[2]) or (typ=="bottom" and price<last[2]):
                last=(idx,typ,price)
            continue
        if idx-last[0]>=4:
            bi.append(dict(a=last[0],b=idx,type=last[1],start=last[2],end=price))
            last=(idx,typ,price)
        else:
            continue
    return bi

def zone_from_bi(bi, n=3):
    """最近 n 笔重叠中枢：ZG=min(各笔高), ZD=max(各笔低)"""
    if len(bi)<n: return None
    segs=bi[-n:]
    highs=[max(s["start"],s["end"]) for s in segs]
    lows=[min(s["start"],s["end"]) for s in segs]
    zg=min(highs); zd=max(lows)
    if zd>zg: return dict(zg=zg,zd=zd,overlap=False,width=None,legs=n)
    return dict(zg=zg,zd=zd,overlap=True,width=(zg-zd)/zd if zd>0 else None,legs=n)

def features(h,l,c,t,window=250):
    lo=max(0,t-window+1)
    mb=merge_bars(h[lo:t+1], l[lo:t+1])
    fr=fractals(mb)
    bi=build_bi(mb,fr)
    out=dict(n_bi=len(bi), n_mb=len(mb), zone=None)
    if len(bi)<3: return out
    z=zone_from_bi(bi,3)
    out["zone"]=z
    price=float(c[t])
    if z and z["overlap"]:
        out["zg"]=z["zg"]; out["zd"]=z["zd"]; out["width"]=z["width"]
        out["dev_zg"]=price/z["zg"]-1 if z["zg"]>0 else None
        out["dev_zd"]=price/z["zd"]-1 if z["zd"]>0 else None
        out["pos"]="above" if price>z["zg"] else ("below" if price<z["zd"] else "inside")
    # 中枢个数（简化：统计连续笔序列中出现的不重叠中枢数）
    zones=[]; k=0
    while k+2<len(bi):
        segs=bi[k:k+3]
        highs=[max(s["start"],s["end"]) for s in segs]; lows=[min(s["start"],s["end"]) for s in segs]
        zg=min(highs); zd=max(lows)
        if zd<=zg:
            zones.append((zg,zd)); k+=3
        else:
            k+=1
    out["n_zone"]=len(zones)
    # 第三类买点近似：最近中枢之后，向上离开（某笔高>ZG）且回抽笔低点>ZG
    out["third_buy"]=False
    # 第三类买点近似：某3笔构成中枢后，后续某笔向上离开（高>ZG），再下一笔回抽低点>ZG
    for i in range(0, len(bi)-4):
        segs=bi[i:i+3]
        highs=[max(s["start"],s["end"]) for s in segs]; lows=[min(s["start"],s["end"]) for s in segs]
        zg2=min(highs); zd2=max(lows)
        if zd2>zg2: continue
        for j in range(i+3, len(bi)-1):
            if max(bi[j]["start"],bi[j]["end"])>zg2 and min(bi[j+1]["start"],bi[j+1]["end"])>zg2:
                out["third_buy"]=True; break
        if out["third_buy"]: break
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check",nargs="*"); ap.add_argument("--eval",action="store_true")
    args=ap.parse_args()
    P=M.build()
    if args.check:
        code=args.check[0] if len(args.check)>0 else "600519"
        j=P["codes"].index(code)
        t=P["nd"]-1
        f=features(P["H"][:,j],P["L"][:,j],P["C"][:,j],t)
        print(code, P["dates"][t], f)
        return
    import pandas as pd
    reps=pd.read_csv(r"D:\about\learn\analysis\min5_calib_trades_all.csv",dtype={"code":str})
    reps["ed"]=pd.to_datetime(reps["ed"]); reps["m"]=reps.ed.dt.to_period("M")
    tr={(x["code"], P["dates"][x["ed"]]):(x["j"],x["t"]) for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])}
    rows=[]
    for _,r in reps.iterrows():
        key=(r["code"], r["ed"].strftime("%Y-%m-%d"))
        if key not in tr: continue
        j,t=tr[key]
        f=features(P["H"][:,j],P["L"][:,j],P["C"][:,j],t)
        rows.append(dict(ret=r["ret_min"], half=r["half_min"], m=r["m"], n_bi=f["n_bi"], n_zone=f.get("n_zone",0),
                         width=f.get("width"), pos=f.get("pos"), third=f.get("third_buy",False)))
    d=pd.DataFrame(rows)
    print("n",len(d),"有中枢(宽度可算):",d.width.notna().sum())
    rng=np.random.default_rng(42); allm=d.groupby("m").ret.mean()
    def stat(lab,m):
        s=d[m]
        if len(s)==0: print(f"{lab:26s} n=0"); return
        print(f"{lab:26s} n={len(s):4d} 平均 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 大亏 {(s.ret<=-0.20).mean()*100:4.1f}%")
    def boot(m,label):
        s=d[m]; am=s.groupby("m").ret.mean()
        df=pd.concat([am,allm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
        if len(diff)<5: print(f"  {label}: 月数不足"); return
        bs=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)]
        lo,hi=np.percentile(bs,[5,95]); print(f"  {label:24s} {diff.mean()*100:+6.2f}pp CI90 [{lo*100:+.2f},{hi*100:+.2f}] 月数{len(diff)}")
    print("\n=== 笔中枢分组（5min 收益）===")
    stat("全部", d.ret.notna()); stat("有中枢", d.width.notna()); stat("无中枢(笔<3)", d.width.isna() & d.pos.isna())
    for th in (0.10,0.15,0.20,0.25):
        stat(f"中枢宽度<={int(th*100)}%", d.width.notna()&(d.width<=th))
    stat("宽度>25%", d.width.notna()&(d.width>0.25))
    stat("价格在ZG上方", d.pos=="above"); stat("价格在ZD下方", d.pos=="below"); stat("价格在中枢内", d.pos=="inside")
    stat("第三类买点结构", d.third==True)
    print("\n=== 月度块 bootstrap（vs 全部）===")
    for lab,m in [("有中枢", d.width.notna()), ("宽度<=15%", d.width.notna()&(d.width<=0.15)), ("宽度<=20%", d.width.notna()&(d.width<=0.20)),
                  ("ZG上方", d.pos=="above"), ("ZD下方", d.pos=="below"), ("第三类买点", d.third==True), ("无中枢", d.width.isna()&d.pos.isna())]:
        boot(m,lab)
    d.to_csv(r"D:\about\learn\analysis\chan_features.csv",index=False,encoding="utf-8-sig")
    print("\n输出：analysis/chan_features.csv")
if __name__=="__main__":
    main()