# -*- coding: utf-8 -*-
"""
VCP 形态捕捉器实证（2026-11-12 用户指令）
====================================================================
用户指令（原话要点）：
  ① 不要限制换手率，这个门槛直接删掉（turn30 ≤ 6% 删除）
  ② ret60（60日涨幅 ≤19%）也删掉
  ③ 德尔平台段 prox 0.89~0.96「其实已经处在合理区间」→ prox 门放宽
  ④ 「先看怎么捕捉这个形态」→ 形态为核心，多尺度收缩（w10 + ct）

映射成候选捕捉器（相对现池 v2.1 的变化）：
  保留：事件表自带基本面清洗（ST/退市、行业黑名单、TTM 营收≥4亿、TTM 盈利）
        + amt30 ≥ 2亿（流动性底线，用户未提，沿用 v2.1）
  删除：turn30 门、ret60 门、w30 ≤ 15% 门
  新增：VCP 多尺度形态门  w10 ≤ X% 且 ct = w10/w30 ≤ Y（末端收缩）
        + w30 ≤ CAP 极端波动兜底（德尔平台段 w30 中位 44.2%，需 ≥50%）
  放宽：prox ∈ {0.90, 0.88, 0.85}（用户认可 0.89~0.96 → 重点看 0.88/0.85）

⚠️ 定位提示：prox ≥ 0.85 意味着「距年高 15% 内」，这不再是「创新高前夜」，
   而是 Minervini 原教旨 VCP（上升趋势中的收缩中继平台）。策略性质改变，
   本脚本只量化这个改变的代价与收益，是否采纳由用户裁决。

方法（沿用既有纪律）：
  ① 网格扫描：prox × w10 × ct × w30cap，看四口径与池子规模
  ② 三个候选方案的边际组逐年检验（A=方案新放入 / B=现池）+ 整池配对 bootstrap
  ③ vol_dry（量比干涸）附加项的增量检验
  ④ 德尔股份 300473 平台段逐日重建：各候选条件下的实际覆盖天数
本脚本【只做研究，不改任何参数】。
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_relax4 import gate, load_events, report_relax, P, _lines  # noqa: E402
from yh_blockboot_all import make_dmap  # noqa: E402
from yh_factors import features_at  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "vcp_capture_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8                 # v2.1 成交额门（流动性底线保留）
BAD = -0.20               # 大亏定义


# ---------------------------------------------------------------- VCP 门
def gate_vcp(d: pd.DataFrame, prox: float, w10: float, ct: float,
             wcap: float | None, dry: float | None = None,
             amt: float = AMT) -> pd.DataFrame:
    """VCP 捕捉器：无 turn / ret60 / w30≤15% 门；形态由 w10+ct 刻画。"""
    m = (d["prox_now"] >= prox) & (d["amt30"] >= amt)
    m &= (d["width10"] <= w10) & (d["contraction"] <= ct)
    if wcap is not None:
        m &= d["width30"] <= wcap
    if dry is not None:
        m &= d["vol_dry"] <= dry
    return d[m]


def four_way(tag: str, d: pd.DataFrame) -> None:
    P("    %-22s 事件 %6d  日均 %5.2f 只 | ab %+6.2f%%  ex %+6.2f%%  fwd %+6.2f%%  大亏 %4.1f%%  β %.2f" % (
        tag, len(d), d.groupby("date").size().mean(),
        d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        d["fwd60"].mean() * 100, (d["ab_fwd60"] < BAD).mean() * 100,
        d["beta"].median()))


def year_four(d_uni: pd.DataFrame, maskA, pool_base: pd.DataFrame, label: str) -> None:
    """边际组逐年：A=方案新放入（不在现池），B=现池。"""
    in_base = d_uni.index.isin(pool_base.index)
    a = d_uni[maskA(d_uni) & ~in_base].copy()
    b = d_uni[in_base].copy()
    a["y"], b["y"] = a["date"].dt.year, b["date"].dt.year
    gdiff = a["ab_fwd60"].mean() - b["ab_fwd60"].mean()
    P("\n    【%s】逐年四口径（A=新放入 n=%d / B=现池 n=%d）" % (label, len(a), len(b)))
    P("    %-6s %5s %5s %10s %10s %10s %10s %10s %10s" % (
        "年", "nA", "nB", "A-B(ab)", "A-B(ex)", "A-B(fwd)", "A大亏", "B大亏", "风险差"))
    ok_ab = ok_ex = ok_bad = tot = 0
    for y in sorted(set(a["y"]) | set(b["y"])):
        x, z = a[a["y"] == y], b[b["y"] == y]
        if len(x) < 8 or len(z) < 8:
            continue
        dab = (x["ab_fwd60"].mean() - z["ab_fwd60"].mean()) * 100
        dex = (x["ex_fwd60"].mean() - z["ex_fwd60"].mean()) * 100
        dfw = (x["fwd60"].mean() - z["fwd60"].mean()) * 100
        ba, bb = (x["ab_fwd60"] < BAD).mean() * 100, (z["ab_fwd60"] < BAD).mean() * 100
        tot += 1
        ok_ab += 1 if (dab > 0) == (gdiff > 0) else 0
        ok_ex += 1 if dex > 0 else 0
        ok_bad += 1 if ba <= bb else 0
        P("    %-6d %5d %5d %+9.2fpp %+9.2fpp %+9.2fpp %8.1f%% %8.1f%% %+9.1fpp" % (
            y, len(x), len(z), dab, dex, dfw, ba, bb, ba - bb))
    P("    及格线 6/8：ab方向 %d/%d | ex为正 %d/%d | A不更危险 %d/%d" % (
        ok_ab, tot, ok_ex, tot, ok_bad, tot))


# ---------------------------------------------------------------- 个案
def case_study() -> None:
    P("\n" + "=" * 104)
    P("④ 个案验证：德尔股份 300473 平台段（2026-10-12~10-23）能否被捕捉")
    P("=" * 104)
    k = pickle.load(open(os.path.join(CACHE, "kline_320.pkl"), "rb"))
    b = k.get("300473")
    if not b:
        P("  缓存中无 300473，跳过")
        return
    prm = {"high_days": 250, "box_days": 30, "tight_days": 10}
    rows = []
    for i in range(250, len(b)):
        f = features_at(b, prm, idx=i)
        if not f:
            continue
        rows.append(dict(
            date=f["date"], close=f["close"], prox=1 + f["dd_from_high"],
            w30=f["width30"], w10=f["width10"], ct=f["contraction"],
            amt=f["amt30"] / 1e8, dry=f["vol_dry"]))
    df = pd.DataFrame(rows)
    plat = df[(df["date"] >= "2026-10-12") & (df["date"] <= "2026-10-23")].copy()
    P("  平台段 %d 天逐日命中情况（amt≥2亿 恒成立，turn/ret60 已删除）：" % len(plat))
    P("  %-11s %7s %7s %7s %6s %6s | %4s %4s %4s %4s" % (
        "date", "close", "prox", "w10", "ct", "dry", "p95", "p88", "p85", "B"))
    for _, r in plat.iterrows():
        p95 = "√" if r["prox"] >= 0.95 else "·"
        p88 = "√" if r["prox"] >= 0.88 else "·"
        p85 = "√" if r["prox"] >= 0.85 else "·"
        vb = "√" if (r["prox"] >= 0.88 and r["w10"] <= 0.10
                     and r["ct"] <= 0.45 and r["w30"] <= 0.50) else "·"
        P("  %-11s %7.2f %7.3f %6.1f%% %6.2f %6.2f | %4s %4s %4s %4s" % (
            r["date"], r["close"], r["prox"], r["w10"] * 100, r["ct"], r["dry"],
            p95, p88, p85, vb))
    vb_hit = ((plat["prox"] >= 0.88) & (plat["w10"] <= 0.10)
              & (plat["ct"] <= 0.45) & (plat["w30"] <= 0.50))
    P("  候选 VCP-B（prox≥0.88 & w10≤10%% & ct≤0.45 & w30≤50%%）平台段命中 %d/%d 天"
      % (int(vb_hit.sum()), len(plat)))
    P("  （prox=0.95 命中 0 天；prox 放到 0.88/0.85 后由 w10+ct 决定是否命中）")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 104)
    P("VCP 形态捕捉器实证 · 2026-11-12 · 删换手率门 / 删 ret60 门 / prox 放宽 / 多尺度形态门")
    P("=" * 104)
    P("事件表 events_fixed.pkl：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min().date(), ev["date"].max().date()))
    P("对照基准 = 现池 v2.1：prox≥0.95 / w30≤15% / amt30≥2亿 / turn30≤6% / ret60≤19%")
    P("候选捕捉器：prox≥P / w10≤W / ct≤C / w30≤CAP / amt30≥2亿（无 turn、无 ret60）")
    P("四口径：ab_fwd60(β中性alpha) / ex_fwd60(横截面超额) / fwd60(原始) / 大亏率(ab<-20%)")
    P("本脚本只做研究，不改任何参数。")

    pool_base = gate(ev, over={"amt": AMT})           # 现池 v2.1

    # ---- ① 网格扫描 ----
    P("\n" + "=" * 104)
    P("① 网格扫描（w30 兜底固定 50%；18 组合）")
    P("=" * 104)
    P("  %-30s %6s %7s %9s %9s %9s %7s %6s" % (
        "条件", "n", "日均", "ab", "ex", "fwd", "大亏", "β"))
    grid_rows = []
    for px in [0.90, 0.88, 0.85]:
        for w10 in [0.08, 0.10, 0.12]:
            for ct in [0.35, 0.45]:
                s = gate_vcp(ev, px, w10, ct, 0.50)
                if len(s) < 100:
                    continue
                grid_rows.append((px, w10, ct, s))
                P("  %-30s %6d %7.2f %+8.2f%% %+8.2f%% %+8.2f%% %6.1f%% %6.2f" % (
                    "p%.2f w10≤%.0f%% ct≤%.2f" % (px, w10 * 100, ct),
                    len(s), s.groupby("date").size().mean(),
                    s["ab_fwd60"].mean() * 100, s["ex_fwd60"].mean() * 100,
                    s["fwd60"].mean() * 100, (s["ab_fwd60"] < BAD).mean() * 100,
                    s["beta"].median()))
    P("  %-30s %6d %7.2f %+8.2f%% %+8.2f%% %+8.2f%% %6.1f%% %6.2f" % (
        "【现池 v2.1】", len(pool_base), pool_base.groupby("date").size().mean(),
        pool_base["ab_fwd60"].mean() * 100, pool_base["ex_fwd60"].mean() * 100,
        pool_base["fwd60"].mean() * 100, (pool_base["ab_fwd60"] < BAD).mean() * 100,
        pool_base["beta"].median()))

    # ---- ② 三个候选方案 ----
    P("\n" + "=" * 104)
    P("② 候选方案整池对比（w30 兜底 50%）")
    P("=" * 104)
    plans = [
        ("VCP-A 严格 p0.90 w10≤8% ct≤0.35", 0.90, 0.08, 0.35),
        ("VCP-B 均衡 p0.88 w10≤10% ct≤0.45", 0.88, 0.10, 0.45),
        ("VCP-C 宽松 p0.85 w10≤12% ct≤0.45", 0.85, 0.12, 0.45),
    ]
    pools = {}
    P("\n  【整池四口径均值】")
    four_way("现池 v2.1", pool_base)
    for tag, px, w10, ct in plans:
        pl = gate_vcp(ev, px, w10, ct, 0.50)
        pools[tag] = pl
        four_way(tag.split(" ")[0], pl)
    # vol_dry 附加
    pl_dry = gate_vcp(ev, 0.88, 0.10, 0.45, 0.50, dry=1.0)
    pools["VCP-B+量比干涸 dry≤1.0"] = pl_dry
    four_way("VCP-B+dry≤1.0", pl_dry)

    # ---- ③ 边际组逐年 + bootstrap ----
    P("\n" + "=" * 104)
    P("③ 边际组逐年 + 整池配对 bootstrap（边际检验样本 = 方案池 ∪ 现池）")
    P("=" * 104)
    for tag, pl in list(pools.items()):
        d_uni = pd.concat([pl, pool_base]).drop_duplicates()
        in_pl = d_uni.index.isin(pl.index)
        year_four(d_uni, lambda x, _m=in_pl: _m, pool_base, tag)
        P("\n  " + "-" * 100)
        report_relax(
            tag, d_uni,
            lambda x, _pl=pl: pd.Series(x.index.isin(_pl.index), index=x.index)
            & ~pd.Series(x.index.isin(pool_base.index), index=x.index),
            lambda x: pd.Series(x.index.isin(pool_base.index), index=x.index),
            "方案新放入", "现池",
            pl, pool_base, dmap)

    case_study()

    P("\n" + "=" * 104)
    P("⑤ 结论要点（自动汇总，最终裁决见速查卡/日志）")
    P("=" * 104)
    P("  1) 删 turn + 删 ret60 + prox 放宽 + 多尺度形态门，本质是换成 VCP 中继平台策略，")
    P("     与「年度新高前夜」定位不同：β、大亏率、池子规模的变化见 ①②。")
    P("  2) 德尔平台段的覆盖依赖 prox 放到 ≤0.88（其平台段 prox 最低 0.874，0.88 档会漏 1~2 天）。")
    P("  3) 是否采纳由用户裁决；若采纳，建议作为独立池与 v2.1 分开记账（参数不动）。")
    P("  耗时 %.1f 秒" % (time.time() - t0))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print("已写出", OUT, flush=True)


if __name__ == "__main__":
    main()
