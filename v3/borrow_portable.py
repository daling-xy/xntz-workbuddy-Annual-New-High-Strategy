# -*- coding: utf-8 -*-
"""borrow_portable.py —— 按便携版真实口径（LOOKBACK_DAYS=30）复刻形态层+中枢，检验 v3 可借鉴性
形态层：zigzag(prominence=1.5%×均价的 find_peaks)、振幅CV/均值/最大比、连涨连跌、30日涨跌幅区间、
        近10日跌幅、近5日跌幅、高低点违规、回弹、离10日高价、连涨跌停；中枢：最近3段重叠 ZG/ZD。
样本：v3(V1) 1,205 笔 5min 校准交易；特征用触发前一日(t)的最近30根/250根日线。
"""
import os, sys, numpy as np, pandas as pd
from scipy.signal import find_peaks
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
LEARN=r"D:\about\learn"
P=M.build()
trades=[x for x in M.make_trades(P,"V1") if M.bs_sym(x["code"])]
meta={(x["code"], P["dates"][x["ed"]]):(x["j"], x["t"]) for x in trades}

def analyze(h,l,c,code):
    # 连涨连跌
    ch=np.diff(c); up=dn=mu=md=0
    for d in ch:
        if d>0: up+=1; dn=0; mu=max(mu,up)
        elif d<0: dn+=1; up=0; md=max(md,dn)
        else: up=dn=0
    if mu>6 or md>6: return "reject","连涨连跌",None
    tot=c[-1]/c[0]-1
    if tot<-0.06 or tot>0.35: return "reject","30日涨跌幅",None
    ch10=c[-1]/c[-10]-1
    if ch10<-0.07: return "reject","近10日跌幅",None
    ch5=c[-1]/c[-6]-1
    if ch5<-0.12: return "reject","近5日跌幅",None
    close_hi10=max(h[-10:]); dev=(close_hi10-c[-1])/close_hi10 if close_hi10>0 else 0
    if dev>0.13: return "reject","离10日高价",None
    # 涨跌停
    lim=0.20 if code.startswith(("300","301","688")) else 0.10
    pct=np.diff(c)/c[:-1]
    lu=ld=0
    for p in pct:
        lu=lu+1 if p>=lim-0.01 else 0
        ld=ld+1 if p<=-lim+0.01 else 0
        if lu>=3 or ld>=3: return "reject","连涨跌停",None
    # swings
    prom=0.015*float(np.mean(c))
    pk,_=find_peaks(h,prominence=prom,distance=2); tr,_=find_peaks(-l,prominence=prom,distance=2)
    # 极值/尖锐度剔除
    def rej(idx,is_peak):
        w=5
        if idx<w or idx>=len(c)-w: return False
        if is_peak:
            before=h[idx-w:idx]; after=h[idx:idx+w]
            rise=(h[idx]-before.min())/before.min() if before.min()>0 else 0
            fall=(h[idx]-after.min())/h[idx] if h[idx]>0 else 0
            if rise>0.12 and fall>0.12: return True
        else:
            before=l[idx-w:idx]; after=l[idx:idx+w]
            fall=(before.max()-l[idx])/before.max() if before.max()>0 else 0
            rise=(after.max()-l[idx])/l[idx] if l[idx]>0 else 0
            if fall>0.12 and rise>0.12: return True
        # sharpness
        if idx<2 or idx>=len(c)-2: return False
        if is_peak: left=abs(h[idx]-h[idx-2])/2; right=abs(h[idx+2]-h[idx])/2
        else: left=abs(l[idx]-l[idx-2])/2; right=abs(l[idx+2]-l[idx])/2
        return (abs(left-right)/c[idx] if c[idx]>0 else 0)>5.0
    vp=[i for i in pk if not rej(int(i),True)]; vt=[i for i in tr if not rej(int(i),False)]
    if len(vp)<2 or len(vt)<2: return "Q","无转折(Q)",None
    pts=[(int(i),"peak",float(h[i])) for i in vp]+[(int(i),"trough",float(l[i])) for i in vt]
    pts.sort(); seq=[]; last=None
    for p in pts:
        if last!=p[1]: seq.append(p); last=p[1]
    if len(seq)<4 or len(seq)%2!=0: return "Q","转折序列",None
    if seq[0][1]!="peak": seq=seq[1:]
    if seq and seq[-1][1]!="trough": seq=seq[:-1]
    if len(seq)<4: return "Q","转折序列",None
    peaks=[p for p in seq if p[1]=="peak"]; troughs=[p for p in seq if p[1]=="trough"]
    amps=[]
    for i in range(0,len(seq)-1,2):
        pp=seq[i][2]; tp=seq[i+1][2]
        if pp>0: amps.append((pp-tp)/pp)
    if len(amps)<2: return "Q","振幅段",None
    mean_a=float(np.mean(amps)); std_a=float(np.std(amps)); cv=std_a/mean_a if mean_a>0 else 999
    if cv>0.47: return "reject","振幅CV",None
    if not (0.04<=mean_a<=0.35): return "reject","平均振幅",None
    if max(amps) > mean_a*1.8: return "reject","最大/平均振幅",None
    srt=sorted(amps,reverse=True)
    if len(srt)>1 and srt[0] > srt[1]*1.5: return "reject","最大/次大振幅",None
    low_v=sum(1 for i in range(1,len(troughs)) if troughs[i][2]/troughs[i-1][2] < 0.97)
    if low_v>2: return "reject","低点下降",None
    high_v=0
    for i in range(1,len(peaks)):
        r=peaks[i][2]/peaks[i-1][2]
        if r<0.97 or (r-1)>0.06: high_v+=1
    tot_rise=(peaks[-1][2]-peaks[0][2])/peaks[0][2] if peaks[0][2]>0 else 0
    if high_v>2 or tot_rise>0.15: return "reject","高点违规",None
    reb_v=0
    for i in range(len(troughs)-1):
        if i<len(peaks) and i+1<len(peaks):
            if peaks[i+1][2] < peaks[i][2]*0.93: reb_v+=1
    if reb_v>2: return "reject","回弹不足",None
    # 中枢
    pts4=seq[-4:]
    highs=[]; lows=[]
    for a,b in zip(pts4[:-1],pts4[1:]):
        highs.append(max(a[2],b[2])); lows.append(min(a[2],b[2]))
    zg=min(highs); zd=max(lows); overlap=zd<=zg
    width=(zg-zd)/zd if (zd>0 and overlap) else None
    return "A","通过",dict(zg=zg, zd=zd, overlap=bool(overlap), width=width,
                dev_zg=(c[-1]/zg-1) if zg>0 else None, dev_zd=(c[-1]/zd-1) if zd>0 else None)

cal=pd.read_csv(os.path.join(LEARN,"analysis","min5_calib_trades_all.csv"),dtype={"code":str})
rows=[]
for _,r in cal.iterrows():
    key=(r["code"], pd.Timestamp(r["ed"]).strftime("%Y-%m-%d"))
    if key not in meta: continue
    j,t=meta[key]
    lo=max(0,t-29)
    h=P["H"][lo:t+1,j]; l=P["L"][lo:t+1,j]; c=P["C"][lo:t+1,j]
    if len(c)<25: continue
    st,why,info=analyze(h,l,c,r["code"])
    row=dict(code=r["code"], ret=r["ret_min"], half=r["half_min"], status=st, why=why)
    if info: row.update(info)
    rows.append(row)
d=pd.DataFrame(rows)
print("分组:", d.status.value_counts().to_dict())
print("A/Q 拒绝原因:", d[d.status!="A"].why.value_counts().to_dict())
d.to_csv(os.path.join(LEARN,"analysis","borrow_portable_features.csv"),index=False,encoding="utf-8-sig")

def stat(lab,m):
    s=d[m] if not isinstance(m,bool) else d
    if len(s)==0: print(f"{lab:32s} n=0"); return
    print(f"{lab:32s} n={len(s):4d} 平均 {s.ret.mean()*100:6.2f}% 中位 {s.ret.median()*100:6.2f}% 胜率 {(s.ret>0).mean()*100:5.1f}% 大亏 {(s.ret<=-0.20).mean()*100:4.1f}% 半仓 {s.half.mean()*100:5.1f}%")

print("\n=== v3 交易上叠加便携版形态层/中枢（5min 收益）===")
stat("基准(全部v3)", True)
stat("A级形态(通过全部形态闸)", d.status=="A")
stat("Q级(无转折/转折不足放行)", d.status=="Q")
stat("A+Q(便携版形态层通过)", d.status.isin(["A","Q"]))
stat("有效中枢(30bar口径,全部)", d.overlap==True)
stat("中枢偏离内(全部)", (d.dev_zg.notna())&(d.dev_zg<=0.08)&(d.dev_zd.notna())&(d.dev_zd>=-0.08))
stat("A & 有效中枢", (d.status=="A")&(d.overlap==True))
stat("A & 偏离内", (d.status=="A")&(d.dev_zg.notna())&(d.dev_zg<=0.08)&(d.dev_zd.notna())&(d.dev_zd>=-0.08))
stat("A & 有效中枢 & 偏离内", (d.status=="A")&(d.overlap==True)&(d.dev_zg.notna())&(d.dev_zg<=0.08)&(d.dev_zd.notna())&(d.dev_zd>=-0.08))
stat("拒绝样本(非A/Q)", d.status=="reject")
