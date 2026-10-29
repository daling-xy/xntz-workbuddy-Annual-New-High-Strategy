# -*- coding: utf-8 -*-
"""
成长版参数搜索（2026-10-27）

目标：在「能选出成长票」与「alpha 仍然显著」之间找可行配置。
方法：网格搜索 + 训练/验证期分段（防 data snooping）
"""
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from yh_rev_fine import load_events, load_fin, attach_fin
from yh_blockboot_all import make_dmap, block_boot, daily

HI = 30.0
SPLIT = "2023-01-01"

ev = load_events()
fin = load_fin()
ev = attach_fin(ev, fin)
dmap = make_dmap(ev)
ev["_d"] = pd.to_datetime(ev["date"])


def stat(g: pd.DataFrame):
    """返回 (n, days, growth_share, beta, alpha_pt, ci_lo, ci_hi, nb, winrate)"""
    if len(g) < 200:
        return None
    pt, lo, hi, nb = block_boot(daily(g), dmap)
    return dict(
        n=len(g), days=int(g["date"].nunique()),
        hi=float((g["YSTZ"] >= HI).mean() * 100),
        beta=float(g["beta"].median()),
        pt=pt, lo=lo, hi_=hi, nb=nb,
        win=float((g["ab_fwd60"] > 0).mean() * 100),
        per_day=len(g) / max(g["date"].nunique(), 1),
    )


GRID = dict(
    prox=[0.95, 0.98],
    width=[0.15, 0.20, 0.25, 0.30],
    amt=[2e8, 4e8],
    tlo=[0.0, 1.5],
    thi=[6.0, 10.0, 15.0, 100.0],
)
keys = list(GRID)
combos = list(itertools.product(*[GRID[k] for k in keys]))
print("网格配置数：%d" % len(combos))

rows = []
for combo in combos:
    c = dict(zip(keys, combo))
    g = ev[(ev["prox_now"] >= c["prox"]) & (ev["width30"] <= c["width"]) &
           (ev["amt30"] >= c["amt"]) & (ev["turn30"] >= c["tlo"]) &
           (ev["turn30"] <= c["thi"])]
    s = stat(g)
    if s is None:
        continue
    tr = stat(g[g["_d"] < SPLIT])
    te = stat(g[g["_d"] >= SPLIT])
    rows.append(dict(c, **s,
                     tr_pt=tr["pt"] if tr else np.nan,
                     tr_lo=tr["lo"] if tr else np.nan,
                     te_pt=te["pt"] if te else np.nan,
                     te_lo=te["lo"] if te else np.nan))

df = pd.DataFrame(rows)
print("有效配置（事件>=200）：%d" % len(df))


def show(sub, title, n=14):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)
    print("  %-5s %-6s %-5s %-5s %-6s %7s %6s %7s %9s %8s %9s %8s %8s" % (
        "prox", "箱宽", "额(亿)", "换手", "日均票", "事件", "区块", "高成长",
        "全样本α", "CI下界", "判定", "训练α", "验证α"))
    sub = sub.sort_values("hi", ascending=False).head(n)
    for _, r in sub.iterrows():
        verdict = "显著" if r["lo"] > 0 else ("显著负" if r["hi_"] < 0 else "含零")
        print("  %-5.2f %-6.0f%% %-5.0f %-5s %-6.1f %7d %6d %6.1f%% %+8.2f%% %+8.2f%% %8s %+7.2f%% %+7.2f%%" % (
            r["prox"], r["width"] * 100, r["amt"] / 1e8,
            "%.1f-%.0f" % (r["tlo"], min(r["thi"], 99)), r["per_day"],
            r["n"], r["nb"], r["hi"], r["pt"] * 100, r["lo"] * 100, verdict,
            r["tr_pt"] * 100, r["te_pt"] * 100))


print()
print("=" * 118)
print("【基准】当前 v2 配置（prox>=0.95, 箱宽<=15%, 额>=4亿, 换手 0~6%）")
print("=" * 118)
b = df[(df["prox"] == 0.95) & (df["width"] == 0.15) & (df["amt"] == 4e8) &
       (df["tlo"] == 0.0) & (df["thi"] == 6.0)]
if len(b):
    r = b.iloc[0]
    print("  事件 %d  日均票 %.2f  高成长占比 %.1f%%  β %.2f  α %+.2f%% CI[%+.2f,%+.2f] 区块%d" % (
        r["n"], r["per_day"], r["hi"], r["beta"], r["pt"] * 100, r["lo"] * 100,
        r["hi_"] * 100, r["nb"]))

# 1) 全样本 alpha 显著 且 高成长占比最高的配置
sig = df[df["lo"] > 0]
show(sig, "① 全样本 α 显著（CI下界>0）的配置，按高成长占比排序")

# 2) 显著 + 训练/验证两段都为正（抗过拟合）
rob = sig[(sig["tr_pt"] > 0) & (sig["te_pt"] > 0)]
show(rob, "② 叠加稳健性：全样本显著 + 训练期(<%s)与验证期(>=%s)点估计同为正" % (SPLIT, SPLIT))

# 3) 只看箱宽放宽后能否在保住显著性的同时提高成长占比
w15 = df[(df["prox"] == 0.95) & (df["width"] == 0.15) & (df["amt"] == 4e8)]
show(w15, "③ 固定 prox=0.95 / 额>=4亿 / 箱宽<=15%，只动换手窗口（看换手能救回多少成长）")

print()
print("=" * 118)
print("④ 箱宽的边际代价（prox=0.95, 额>=4亿, 换手0~6% 固定）")
print("=" * 118)
print("  %-8s %8s %8s %9s %10s %10s" % ("箱宽", "事件", "日均票", "高成长占比", "α", "CI"))
for w in [0.15, 0.20, 0.25, 0.30]:
    r = df[(df["prox"] == 0.95) & (df["width"] == w) & (df["amt"] == 4e8) &
           (df["tlo"] == 0.0) & (df["thi"] == 6.0)]
    if len(r):
        r = r.iloc[0]
        v = "显著" if r["lo"] > 0 else ("显著负" if r["hi_"] < 0 else "含零")
        print("  <=%-6.0f%% %8d %8.1f %8.1f%% %+9.2f%% [%+5.2f,%+5.2f] %s" % (
            w * 100, r["n"], r["per_day"], r["hi"], r["pt"] * 100,
            r["lo"] * 100, r["hi_"] * 100, v))

print()
print("=" * 118)
print("⑤ 放宽后新增的那批票，单独看有没有 alpha（箱宽 15~25% 边际带 + 高成长子集）")
print("=" * 118)
base = ev[(ev["prox_now"] >= 0.95) & (ev["amt30"] >= 4e8) &
          (ev["turn30"] >= 0.0) & (ev["turn30"] <= 6.0)]
slices = [
    ("箱内 <=15%", base[base["width30"] <= 0.15]),
    ("边际带 15~25%", base[(base["width30"] > 0.15) & (base["width30"] <= 0.25)]),
    ("边际带 15~25% ∩ 高成长>=30", base[(base["width30"] > 0.15) & (base["width30"] <= 0.25) & (base["YSTZ"] >= 30)]),
    ("边际带 15~25% ∩ prox>=0.98", base[(base["width30"] > 0.15) & (base["width30"] <= 0.25) & (base["prox_now"] >= 0.98)]),
    ("全市场 ∩ 高成长>=30（不加结构门）", ev[ev["YSTZ"] >= 30]),
    ("箱内<=15% ∩ 高成长>=30", base[(base["width30"] <= 0.15) & (base["YSTZ"] >= 30)]),
]
print("  %-34s %8s %9s %10s %16s %8s" % ("切片", "事件", "日均票", "β", "α [90%CI]", "判定"))
for lab, g in slices:
    if len(g) < 100:
        print("  %-34s %8d 样本不足" % (lab, len(g)))
        continue
    pt, lo, hi, nb = block_boot(daily(g), dmap)
    v = "显著" if lo > 0 else ("显著负" if hi < 0 else "含零")
    print("  %-34s %8d %9.1f %10.2f %+7.2f%% [%+5.2f,%+5.2f] %8s" % (
        lab, len(g), len(g) / max(g["date"].nunique(), 1), g["beta"].median(),
        pt * 100, lo * 100, hi * 100, v))

df.to_csv("output/growth_grid.csv", index=False, encoding="utf-8-sig")
print()
print("网格明细已存：output/growth_grid.csv")
