# -*- coding: utf-8 -*-
"""q_box_eval.py —— 在 v3 的 5min 交易上检验：剔A留Q / 箱宽 / 中枢偏离，含月度块 bootstrap"""
import pandas as pd, numpy as np, os
d=pd.read_csv(r"D:\about\learn\analysis\borrow_portable_features.csv",dtype={"code":str})
d["ed"]=pd.to_datetime(d["ed"]); d["m"]=d.ed.dt.to_period("M")
# w30：从 kline_long2 重算
import sys
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build()
tr={ (x["code"], P["dates"][x["ed"]]):(x["j"],x["t"]) for x in M.make_trades(P,"V1") if M.bs_sym(x["code"]) }
w30=[]
for _,r in d.iterrows():
    j,t=tr[(r["code"], r["ed"].strftime("%Y-%m-%d"))]
    lo=max(0,t-29)
    hi=float(np.nanmax(P["C"][lo:t+1,j])); lo2=float(np.nanmin(P["C"][lo:t+1,j]))
    w30.append((hi/lo2-1) if lo2>0 else np.nan)
d["w30"]=w30
rng=np.random.default_rng(42)
def stats(m):
    s=d[m]; return len(s), s.ret.mean()*100, s.ret.median()*100, (s.ret>0).mean()*100, (s.ret<=-0.20).mean()*100
def boot(m,label):
    a=d[m]; allm=d.groupby("m").ret.mean()
    am=a.groupby("m").ret.mean()
    df=pd.concat([am,allm],axis=1).dropna(); diff=(df.iloc[:,0]-df.iloc[:,1]).values
    n=len(diff)
    if n<5: print(f"{label:30s} 月数不足"); return
    bs=[rng.choice(diff,size=n,replace=True).mean() for _ in range(2000)]
    lo,hi=np.percentile(bs,[5,95])
    print(f"{label:30s} {diff.mean()*100:+6.2f}pp CI90 [{lo*100:+.2f},{hi*100:+.2f}] 月数{n}")
print("=== 分组统计（5min 收益）===")
def row(lab,m):
    n,mean,med,win,big=stats(m)
    print(f"{lab:32s} n={n:4d} 平均 {mean:6.2f}% 中位 {med:6.2f}% 胜率 {win:5.1f}% 大亏 {big:4.1f}%")
row("全部v3", d.status.notna())
row("A only", d.status=="A")
row("Q only（剔A和拒绝）", d.status=="Q")
row("Q+拒绝（剔A）", d.status.isin(["Q","reject"]))
row("A+Q（剔拒绝）", d.status.isin(["A","Q"]))
row("箱宽<=15%", d.w30<=0.15)
row("箱宽<=20%", d.w30<=0.20)
row("箱宽<=25%", d.w30<=0.25)
row("箱宽>25%", d.w30>0.25)
row("Q & 箱宽<=15%", (d.status=="Q")&(d.w30<=0.15))
row("Q+拒绝 & 箱宽<=15%", (d.status.isin(["Q","reject"]))&(d.w30<=0.15))
row("A中偏离内", (d.status=="A")&(d.dev_zg.notna())&(d.dev_zg<=0.08)&(d.dev_zd.notna())&(d.dev_zd>=-0.08))
print("\n=== 相对“全部v3”的月度块 bootstrap ===")
boot(d.status=="Q","Q only")
boot(d.status.isin(["Q","reject"]),"Q+拒绝(剔A)")
boot(d.status.isin(["A","Q"]),"A+Q(剔拒绝)")
boot(d.w30<=0.15,"箱宽<=15%")
boot(d.w30<=0.20,"箱宽<=20%")
boot(d.w30>0.25,"箱宽>25%")
boot((d.status=="Q")&(d.w30<=0.15),"Q & 箱宽<=15%")
boot((d.status=="A")&(d.dev_zg<=0.08)&(d.dev_zd>=-0.08),"A & 偏离内")
