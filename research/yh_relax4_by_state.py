# -*- coding: utf-8 -*-
"""
四门放宽实证 · 市场状态条件化复跑（2026-11-10 第二轮）
================================================================
第一轮（research/yh_relax4.py）回答的是「全样本 / DOWN 期放宽是否更好」→ 四门全否。
本轮回答用户的新问题：「市场转好后，是不是该放宽？」
即检验【状态依赖】：放宽的收益/风险是否随市场状态改变，
尤其在「深跌后宽度快速修复」(REBOUND) 段是否由负转正。

方法沿用第一轮两条纪律：
  ① 边际组检验在【不含被测门】的样本上做（防幸存者偏差）
  ② 整池用配对区块 bootstrap（60交易日区块 / 4000次 / 90%CI）
新增：所有结果按状态分层输出，并给出逐年方向一致性。

状态定义仅用于历史分层研究，不构成对当前市场的判断。
数据来源：通达信指数日线（宽度）＋腾讯前复权日线事件表。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yh_blockboot_all import make_dmap, paired_block_boot, sig  # noqa: E402
from yh_conserv_diag import load_events, strict, BAD  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE_DIR, "output", "relax4_by_state_20261110.txt")
_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


BASE_GATES = dict(prox=0.95, width=0.15, amt=4e8, turn=6.0, ret60=0.19)
MIN_N = 8


def gate(d, drop=None, over=None):
    g = dict(BASE_GATES)
    if over:
        g.update(over)
    m = pd.Series(True, index=d.index)
    if drop != "prox":
        m &= d["prox_now"] >= g["prox"]
    if drop != "width":
        m &= d["width30"] <= g["width"]
    if drop != "amt":
        m &= d["amt30"] >= g["amt"]
    if drop != "turn":
        m &= d["turn30"] <= g["turn"]
    if drop != "ret60":
        m &= pd.to_numeric(d["ret60"], errors="coerce") <= g["ret60"]
    return d[m]


HYP = [
    # (标题, 被测门, 边际组A掩码, 整池放宽参数, 简述)
    ("H1 距年高 0.95→0.90", "prox",
     lambda x: (x["prox_now"] >= 0.90) & (x["prox_now"] < 0.95),
     lambda x: (x["prox_now"] >= 0.95),
     dict(prox=0.90), None, "0.90~0.95 边际"),
    ("H2 箱宽 15%→20%", "width",
     lambda x: (x["width30"] > 0.15) & (x["width30"] <= 0.20),
     lambda x: (x["width30"] <= 0.15),
     dict(width=0.20), None, "15%~20% 边际"),
    ("H3 换手 6%→7%", "turn",
     lambda x: (x["turn30"] > 6.0) & (x["turn30"] <= 7.0),
     lambda x: (x["turn30"] <= 6.0),
     dict(turn=7.0), None, "6%~7% 边际"),
    ("H4 删除 ret60 门", "ret60",
     lambda x: pd.to_numeric(x["ret60"], errors="coerce") > 0.19,
     lambda x: pd.to_numeric(x["ret60"], errors="coerce") <= 0.19,
     dict(ret60=np.inf), "ret60", "ret60>19% 边际"),
]


def state_report(ev, dmap, title, drop, maskA, maskB, over, over_drop, labA):
    P("\n" + "=" * 112)
    P(title)
    P("=" * 112)
    d = gate(ev, drop=drop)                    # 不含被测门
    base_pool = gate(ev)                       # 现池
    relax_pool = gate(ev, drop=over_drop, over=over)
    P("  %-9s %8s %8s | %9s %9s %9s %8s | %8s %8s %8s | %s" % (
        "状态", "nA", "nB", "A超额", "B超额", "A−B", "大亏率差", "现池日均", "放宽日均", "配对差", "判定"))
    P("  " + "-" * 108)
    summary = {}
    for s in ["REBOUND", "UP", "CHOP", "DOWN", "全样本"]:
        sub = ev if s == "全样本" else ev[ev["state"] == s]
        if len(sub) < 200:
            P("  %-9s 样本不足" % s)
            continue
        ds = gate(sub, drop=drop)
        a, b = ds[maskA(ds)], ds[maskB(ds)]
        if len(a) < 30 or len(b) < 30:
            P("  %-9s 边际组样本不足 (nA=%d nB=%d)" % (s, len(a), len(b)))
            continue
        ga, gb = a["ab_fwd60"].mean(), b["ab_fwd60"].mean()
        bad_a = (a["ab_fwd60"] < BAD).mean()
        bad_b = (b["ab_fwd60"] < BAD).mean()
        pb = gate(sub)
        pr = gate(sub, drop=over_drop, over=over)
        pt, lo, hi, nb = paired_block_boot(pr, pb, dmap)
        per_b = pb.groupby("date").size().mean() if len(pb) else 0
        per_r = pr.groupby("date").size().mean() if len(pr) else 0
        verdict = "—"
        if not pd.isna(pt):
            verdict = sig(pt, lo, hi)
        P("  %-9s %8d %8d | %+8.2f%% %+8.2f%% %+8.2fpp %+7.1fpp | %8.2f %8.2f %+7.2fpp | %s" % (
            s, len(a), len(b), ga * 100, gb * 100, (ga - gb) * 100,
            (bad_a - bad_b) * 100, per_b, per_r,
            (pt * 100) if not pd.isna(pt) else np.nan, verdict))
        summary[s] = (ga - gb, bad_a - bad_b, pt, lo, hi)

    # 逐年方向一致性（各状态内，有效年两组均≥8事件）
    P("\n  逐年方向一致性（A−B 与全样本同向 / 有效年；风险=A不更危险）：")
    for s in ["REBOUND", "UP", "CHOP", "DOWN", "全样本"]:
        sub = ev if s == "全样本" else ev[ev["state"] == s]
        ds = gate(sub, drop=drop)
        if len(ds) < 200:
            continue
        glob = ds[maskA(ds)]["ab_fwd60"].mean() - ds[maskB(ds)]["ab_fwd60"].mean()
        ok = tot = okb = 0
        for y, g in ds.groupby(ds["date"].dt.year):
            aa, bb = g[maskA(g)], g[maskB(g)]
            if len(aa) < MIN_N or len(bb) < MIN_N:
                continue
            tot += 1
            ok += 1 if ((aa["ab_fwd60"].mean() - bb["ab_fwd60"].mean()) > 0) == (glob > 0) else 0
            okb += 1 if (aa["ab_fwd60"] < BAD).mean() <= (bb["ab_fwd60"] < BAD).mean() else 0
        if tot:
            P("    %-9s 收益 %d/%d 年   风险 %d/%d 年" % (s, ok, tot, okb, tot))
    return summary


def main():
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 112)
    P("四门放宽 · 状态条件化复跑 · 2026-11-10 · 数据来源：通达信指数 / 腾讯前复权日线")
    P("=" * 112)
    P("事件表 %s 行 / %d 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["date"].nunique(), ev["date"].min().date(), ev["date"].max().date()))
    P("状态分布：" + str(ev["state"].value_counts().to_dict()))
    P("基准门：prox≥0.95 / 箱宽≤15% / 额≥4亿 / 换手≤6% / ret60≤19%")
    P("指标：ab_fwd60（β调整60日超额）；大亏 = ab_fwd60<−20%；配对区块 bootstrap 90%CI")
    P("REBOUND = 宽度20日绝对变化>+5pp 且 宽度<40%（深跌后快速修复，历史分层用）")
    all_res = {}
    for title, drop, mA, mB, over, odrop, lab in HYP:
        all_res[title] = state_report(ev, dmap, title + "  【边际组：" + lab + "】",
                                       drop, mA, mB, over, odrop, lab)
    P("\n" + "=" * 112)
    P("汇总：REBOUND 段 vs 其余状态的整池配对点估计（放宽池 − 现池）")
    P("=" * 112)
    P("  %-24s %12s %12s %12s %12s" % ("假设", "REBOUND", "UP", "CHOP", "DOWN"))
    P("  " + "-" * 76)
    for k, v in all_res.items():
        vals = []
        for s in ["REBOUND", "UP", "CHOP", "DOWN"]:
            vals.append("%+10.2fpp" % (v[s][2] * 100) if s in v and not pd.isna(v[s][2]) else "         —")
        P("  %-24s %s" % (k, " ".join(vals)))
    P("\n耗时 %.1fs" % (time.time() - t0))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入 " + OUT)


if __name__ == "__main__":
    main()
