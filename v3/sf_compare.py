# -*- coding: utf-8 -*-
"""sf_compare.py —— 结构优先 vs v3：日线近似口径对比（2023-2026）"""
import numpy as np, pandas as pd, os, sys
BASE=r"C:\Users\xntz\Desktop\test\yearly_high_strategy"
sys.path.insert(0, os.path.join(BASE,"v3"))
import min5_calib as M
P=M.build(); dates=np.array(P["dates"]); jmap={c:i for i,c in enumerate(P["codes"])}
sf=pd.read_csv(r"D:\about\learn\analysis\structure_first_trades_2023.csv",dtype={"code":str})
sf["t"]=pd.to_datetime(sf["t"]); sf["ed"]=pd.to_datetime(sf["ed"])
def daily_ret(df):
    out=[]
    for _,r in df.iterrows():
        j=jmap.get(r["code"])
        if j is None: continue
        t=int(np.searchsorted(dates, r["t"].strftime("%Y-%m-%d")))
        ed=int(np.searchsorted(dates, r["ed"].strftime("%Y-%m-%d")))
        tr=dict(j=j,t=t,ed=ed,top=float(r["top"]),entry=float(r["entry"]))
        v=M.sim_daily(P,tr)
        out.append(v["ret"] if v else np.nan)
    return np.array(out)
sf["ret_d"]=daily_ret(sf)
v3=pd.read_csv(r"D:\about\learn\analysis\min5_calib_trades_all.csv",dtype={"code":str})
v3["ed"]=pd.to_datetime(v3["ed"]); v3=v3[v3.ed.dt.year>=2023].copy()
print("=== 2023-2026 交易数 ===")
print("structure-first:",len(sf)," v3(已校准):",len(v3))
print("结构优先 年分布:", sf.ed.dt.year.value_counts().sort_index().to_dict())
print("v3 年分布:", v3.ed.dt.year.value_counts().sort_index().to_dict())
# 同日/同月组合
def mon_stats(d,col,label):
    x=d.dropna(subset=[col]).copy(); x["m"]=x.ed.dt.to_period("M")
    dm=x.groupby("ed")[col].mean(); dm.index=pd.to_datetime(dm.index)
    mm=dm.groupby(dm.index.to_period("M")).mean()
    print(f"{label}: 交易 {len(x)} 笔 | 月度(日等权) 均值 {mm.mean()*100:.2f}% 月胜率 {(mm>0).mean()*100:.1f}% 月数 {len(mm)} | 单笔均值 {x[col].mean()*100:.2f}% 中位 {x[col].median()*100:.2f}% 胜率 {(x[col]>0).mean()*100:.1f}% 大亏 {(x[col]<=-0.20).mean()*100:.1f}%")
    return mm
mm_sf=mon_stats(sf,"ret_d","结构优先(日线近似)")
mm_v3=mon_stats(v3,"ret_daily","v3(日线近似,同批)")
# 月度差
diff=pd.concat([mm_sf,mm_v3],axis=1).dropna(); diff.columns=["sf","v3"]
if len(diff):
    print(f"\n月度差(SF-v3): 平均 {((diff.sf-diff.v3).mean()*100):+.2f}pp 胜率 {(diff.sf>diff.v3).mean()*100:.1f}% 月数 {len(diff)}")
# 分年
for y in [2023,2024,2025,2026]:
    a=sf[sf.ed.dt.year==y].ret_d; b=v3[v3.ed.dt.year==y].ret_daily
    print(f"{y}: SF n={len(a):4d} 均值 {a.mean()*100:6.2f}% | v3 n={len(b):3d} 均值 {b.mean()*100:6.2f}%")
# 重叠
sfk=set(zip(sf.code, sf.ed.dt.strftime("%Y-%m-%d"))); v3k=set(zip(v3.code, v3.ed.dt.strftime("%Y-%m-%d")))
print("\n重叠交易(code+触发日):", len(sfk & v3k))
sf.to_csv(r"D:\about\learn\analysis\structure_first_trades_2023_ret.csv",index=False,encoding="utf-8-sig")
