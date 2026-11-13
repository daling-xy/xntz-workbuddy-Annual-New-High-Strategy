# -*- coding: utf-8 -*-
"""2026-11-12：把百亚股份/温氏股份纳入主池的最小改动方案扫描。

用户诉求：只要一张主池表；百亚股份(003006)、温氏股份(300498) 必须入选；
不刻意压缩池子数量。

做法：直接读 11-12 收盘的全市场打分明细（已含全部形态因子），复现 v2.6 主门，
再逐一模拟放宽/补充通道，统计新增票与名单，不改动任何正式参数。
"""
from __future__ import annotations

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "output", "v2", "全市场打分明细_v2.csv")
OUT = os.path.join(BASE, "output", "merge_pcb_20261112.txt")

BLACK = ["银行Ⅱ", "基础建设", "房屋建设Ⅱ", "专业工程", "工程咨询服务Ⅱ",
         "铁路公路", "物流", "航空机场", "通信服务", "银行", "房屋建设",
         "工程咨询服务"]

d = pd.read_csv(SRC, encoding="utf-8-sig", dtype={"code": str})
d["code"] = d["code"].str.zfill(6)
num = lambda c: pd.to_numeric(d[c], errors="coerce")

prox, w10, w20, w30 = num("prox_now"), num("width10"), num("width20"), num("width30")
ct, hi10, hi30 = num("contraction"), num("hi10"), num("hi30")
r10, bd, amt = num("ret10"), num("base_days"), num("amt30")
pcb = num("pcb_hit").fillna(0)
plat_w, pcb_days = num("pcb_plat_w"), num("pcb_days")
ind = d["industry"].astype(str)
ok_ind = ~ind.isin(BLACK) & ind.notna()
base_ok = (prox >= 0.88) & (amt >= 2e8) & ok_ind & (r10 <= 0.08) & (hi10 >= hi30 * 0.95) & (w30 <= 0.50)

lines = []
def put(s=""):
    lines.append(s)
    print(s)

put("=" * 100)
put("百亚股份 / 温氏股份 纳入主池 —— 方案扫描（基准日 2026-11-12 收盘，数据源：通达信指数口径 + 本地前复权日线缓存）")
put("=" * 100)
put(f"全市场样本 {len(d)} 只；基线主池（v2.6 pass）= {int(d['pass'].sum())} 只")
put("")
put("【卡点诊断】")
cols = ["name", "prox_now", "width10", "width20", "width30", "contraction", "hi10", "hi30",
        "ret10", "base_days", "amt30", "pcb_hit", "pcb_plat_w", "pcb_days", "score"]
for c in ["003006", "300498"]:
    r = d[d["code"] == c].iloc[0]
    put(f"\n  {c} {r['name']}：")
    put(f"    VCP: w10={r['width10']:.2%}(门10%) ct={r['contraction']:.3f}(门0.45) w30={r['width30']:.2%}(门50%)")
    put(f"    C  : w10/w20={r['width10']/r['width20']:.3f}(门0.70) w20/w30={r['width20']/r['width30']:.3f}(门0.90)")
    put(f"    H  : hi10/hi30={r['hi10']/r['hi30']:.3f}(门0.95)  F: base_days={int(r['base_days'])}(门35)")
    put(f"    R  : ret10={r['ret10']:.2%}(门8%)  额={r['amt30']/1e8:.2f}亿(门2亿)  prox_now={r['prox_now']:.3f}(门0.88)")
    put(f"    PCB: hit={int(r['pcb_hit'])} 平台宽={r['pcb_plat_w']:.2%} 平台日={int(r['pcb_days'])} 评分={r['score']:.1f}")

# ---------- 各方案 ----------
def vcp(w10_max, ct_max):
    return (w10 <= w10_max) & (ct <= ct_max)

def c_gate(w30k, fast=0.04):
    return ((w10 <= w20 * 0.70) & (w20 <= w30 * w30k)) | (w10 <= fast)

def f_gate(bmax):
    return bd <= bmax

base_v26 = d["pass"].astype(bool)

schemes = {}
# S1 全局放宽（主门整体动刀）
schemes["S1 全局放宽 w10≤12% ct≤0.70 c_w30≤1.00 F≤60"] = (
    base_ok & vcp(0.12, 0.70) & c_gate(1.00) & f_gate(60))
schemes["S2 全局放宽 w10≤12% ct≤0.70 c_w30≤1.00 F≤35"] = (
    base_ok & vcp(0.12, 0.70) & c_gate(1.00) & f_gate(35))
schemes["S3 全局放宽 w10≤12% ct≤0.70 C原样 F≤60"] = (
    base_ok & vcp(0.12, 0.70) & c_gate(0.90) & f_gate(60))

# S4/S5/S6：主门完全不动，只加「PCB 补充通道」（并列入口）
pcb_b = pcb == 1
schemes["S4 主门 + PCB通道(w10≤12% ct≤0.70 免C/F)"] = (
    base_v26 | (pcb_b & base_ok & vcp(0.12, 0.70) & c_gate(1.00)))
schemes["S5 主门 + PCB通道(另限平台宽≤16%)"] = (
    base_v26 | (pcb_b & base_ok & vcp(0.12, 0.70) & c_gate(1.00) & (plat_w <= 0.16)))
schemes["S6 主门 + PCB通道(平台宽≤16% 且平台日10~45)"] = (
    base_v26 | (pcb_b & base_ok & vcp(0.12, 0.70) & c_gate(1.00) &
                (plat_w <= 0.16) & (pcb_days >= 10) & (pcb_days <= 45)))

# S7：不靠 PCB 标签，纯形态的「末端略宽 + 横盘久」豁免
schemes["S7 主门 + 纯形态通道(w10≤12% ct≤0.70 免C/F)"] = (
    base_v26 | (base_ok & vcp(0.12, 0.70) & c_gate(1.00)))

put("\n" + "=" * 100)
put("【方案对比】")
put("=" * 100)
for name, m in schemes.items():
    got = d[m].sort_values("score", ascending=False)
    new = got[~got["code"].isin(d[base_v26]["code"])]
    hit1 = "百亚" in "".join(got["name"].tolist()) or "003006" in got["code"].tolist()
    hit2 = "温氏" in "".join(got["name"].tolist()) or "300498" in got["code"].tolist()
    put(f"\n{name}")
    put(f"  主池 {len(got)} 只（新增 {len(new)}）｜百亚 {'✓' if hit1 else '✗'} 温氏 {'✓' if hit2 else '✗'}")
    put(f"  新增名单：" + "、".join(f"{r['name']}({r['code']},{r['score']:.1f})" for _, r in new.iterrows()))

put("\n" + "=" * 100)
put("【S1 全局放宽后完整池子（按评分）】")
put("=" * 100)
s1 = d[schemes["S1 全局放宽 w10≤12% ct≤0.70 c_w30≤1.00 F≤60"]].sort_values("score", ascending=False)
put(f"  共 {len(s1)} 只")
for _, r in s1.iterrows():
    put(f"  {r['score']:5.1f}  {r['code']} {r['name']:<8} {str(r['industry']):<10} "
        f"prox {r['prox_now']:.3f} w10 {r['width10']:.1%} ct {r['contraction']:.2f} "
        f"bd {int(r['base_days']):>2} pcb {int(r['pcb_hit'])} amt {r['amt30']/1e8:.1f}亿")

put("\n" + "=" * 100)
put("【S6 主门+PCB通道后完整池子（按评分）】")
put("=" * 100)
s6 = d[schemes["S6 主门 + PCB通道(平台宽≤16% 且平台日10~45)"]].sort_values("score", ascending=False)
put(f"  共 {len(s6)} 只")
for _, r in s6.iterrows():
    put(f"  {r['score']:5.1f}  {r['code']} {r['name']:<8} {str(r['industry']):<10} "
        f"prox {r['prox_now']:.3f} w10 {r['width10']:.1%} ct {r['contraction']:.2f} "
        f"bd {int(r['base_days']):>2} pcb {int(r['pcb_hit'])} 平台宽 {r['pcb_plat_w']:.1%} amt {r['amt30']/1e8:.1f}亿")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n已输出: {OUT}")
