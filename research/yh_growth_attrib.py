# -*- coding: utf-8 -*-
"""
成长股去哪了 —— 严格归因（2026-10-27）

回答三个问题：
  A. 口径澄清：成交额 amt30≥4亿 与 营收 revenue_ttm≥4亿 是不是一回事
  B. 换手窗口：v2 的 0~6% 与 v1 的 1.5~10% 到底谁在稀释成长占比（ceteris paribus）
  C. 箱宽：这道门砍掉的成长票，边际质量如何（能不能放宽）
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from yh_rev_fine import load_events, load_fin, attach_fin
from yh_blockboot_all import make_dmap, block_boot, daily

HI = 30.0  # 高成长：营收同比 >= 30%

ev = load_events()
fin = load_fin()
ev = attach_fin(ev, fin)
dmap = make_dmap(ev)


def prof(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return dict(n=0, hi=np.nan, beta=np.nan, turn=np.nan, ystz=np.nan)
    return dict(
        n=len(g),
        hi=float((g["YSTZ"] >= HI).mean() * 100),
        beta=float(g["beta"].median()),
        turn=float(g["turn30"].median()),
        ystz=float(g["YSTZ"].median()),
    )


def row(label: str, r: dict) -> None:
    print("  %-34s %8d %9.1f%% %8.2f %9.2f%% %9.1f%%" % (
        label, r["n"], r["hi"], r["beta"], r["turn"], r["ystz"]))


HDR = "  %-34s %8s %9s %8s %9s %9s" % ("配置", "事件数", "高成长占比", "beta", "换手中位", "增速中位")

print("=" * 100)
print("A. 口径澄清：两个「4亿」是不是同一个东西？")
print("=" * 100)
print("  参数1  PARAMS['amt30_min']  = 4e8  → 30日【日均成交额】>= 4 亿元/天 （行情量能门槛）")
print("  参数2  PARAMS['revenue_min'] = 4e8  → 最近12个月【营业总收入 TTM】>= 4 亿元 （财报规模门槛）")
print("  两者量纲不同（元/天 vs 元/年）、列不同（amt30 vs revenue_ttm）、作用不同。")
print("  数值都是 4e8 纯属巧合 —— 后者由『原2亿半年累计 ≈ 全年4亿』标定而来。")
print()

print("=" * 100)
print("B. 换手窗口归因（ceteris paribus：prox>=0.95、箱宽<=15%、成交额>=4亿 全部固定）")
print("=" * 100)
print(HDR)

BASE = ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) & (ev["amt30"] >= 4e8)]

win = {
    "v2 窗口  0 ~ 6%":        (0.0, 6.0),
    "v1 窗口  1.5 ~ 10%":     (1.5, 10.0),
    "交集     1.5 ~ 6%":      (1.5, 6.0),
    "仅v2多出  0 ~ 1.5%":     (0.0, 1.5),
    "仅v1多出  6 ~ 10%":      (6.0, 10.0),
    "无约束    0 ~ 100%":     (0.0, 100.0),
}
res = {}
for lab, (lo, hi) in win.items():
    g = BASE[(BASE["turn30"] >= lo) & (BASE["turn30"] <= hi)]
    res[lab] = prof(g)
    row(lab, res[lab])

print()
print("  【关键分解】以交集 1.5~6%% 为共同基准，看两端各自的稀释/浓缩作用：")
common = res["交集     1.5 ~ 6%"]
lowadd = res["仅v2多出  0 ~ 1.5%"]
hicut = res["仅v1多出  6 ~ 10%"]
print("    共同部分 1.5~6%%    : 高成长 %.1f%%  β %.2f  事件 %d" % (common["hi"], common["beta"], common["n"]))
print("    低端 0~1.5%%（v2独有）: 高成长 %.1f%%  β %.2f  事件 %d  → 拉低还是拉高？" % (
    lowadd["hi"], lowadd["beta"], lowadd["n"]))
print("    高端 6~10%%（v1独有）: 高成长 %.1f%%  β %.2f  事件 %d  → v2 砍掉的是不是成长票？" % (
    hicut["hi"], hicut["beta"], hicut["n"]))
print()
n_low, n_com, n_hi = lowadd["n"], common["n"], hicut["n"]
v2_hi = (n_low * lowadd["hi"] + n_com * common["hi"]) / max(n_low + n_com, 1)
v1_hi = (n_hi * hicut["hi"] + n_com * common["hi"]) / max(n_hi + n_com, 1)
print("    算术还原：v2窗口(0~6%%) 高成长 ≈ %.1f%% ；v1窗口(1.5~10%%) 高成长 ≈ %.1f%%" % (v2_hi, v1_hi))
print("    差额 %.1fpp 中：低端加入贡献 %.1fpp，高端砍掉贡献 %.1fpp" % (
    v1_hi - v2_hi, v2_hi - common["hi"], v1_hi - common["hi"]))

print()
print("=" * 100)
print("C. 箱宽归因（ceteris paribus：prox>=0.95、成交额>=4亿、换手 0~6% 固定，只动箱宽）")
print("=" * 100)
print("  %-16s %8s %11s %8s %10s %14s %10s" % (
    "箱宽上限", "事件数", "高成长占比", "beta", "增速中位", "60日超额", "90%CI"))
WBASE = ev[(ev["prox_now"] >= 0.95) & (ev["amt30"] >= 4e8) &
           (ev["turn30"] >= 0.0) & (ev["turn30"] <= 6.0)]
for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    g = WBASE[WBASE["width30"] <= w]
    if len(g) < 30:
        continue
    pt, lo, hi, nb = block_boot(daily(g), dmap)
    v = "显著" if lo > 0 else ("显著负" if hi < 0 else "含零")
    print("  %-16s %8d %10.1f%% %8.2f %9.1f%% %+13.2f%% [%+5.2f,%+5.2f] %s" % (
        "<=%.0f%%" % (w * 100), len(g), (g["YSTZ"] >= HI).mean() * 100,
        g["beta"].median(), g["YSTZ"].median(), pt * 100, lo * 100, hi * 100, v))

print()
print("  【边际带检验】15%~25% 这一圈「因箱宽被挡在外面」的票，质量到底如何？")
inb = WBASE[WBASE["width30"] <= 0.15]
band = WBASE[(WBASE["width30"] > 0.15) & (WBASE["width30"] <= 0.25)]
print("    %-22s %8d %10.1f%% %8.2f" % ("箱内 <=15%", len(inb),
                                        (inb["YSTZ"] >= HI).mean() * 100, inb["beta"].median()))
print("    %-22s %8d %10.1f%% %8.2f" % ("边际带 15%~25%", len(band),
                                        (band["YSTZ"] >= HI).mean() * 100, band["beta"].median()))
for lab, g in [("箱内<=15%", inb), ("边际带15~25%", band)]:
    pt, lo, hi, nb = block_boot(daily(g), dmap)
    v = "显著" if lo > 0 else ("显著负" if hi < 0 else "含零")
    print("      %-14s 60日超额 %+6.2f%%  CI[%+5.2f,%+5.2f]  区块%d  %s" % (
        lab, pt * 100, lo * 100, hi * 100, nb, v))

print()
print("  【箱内成长票的 alpha】在 <=15%% 箱内，高成长子集单独表现如何？")
for lab, g in [("高成长 YSTZ>=30", inb[inb["YSTZ"] >= 30]),
               ("中成长 10~30", inb[(inb["YSTZ"] >= 10) & (inb["YSTZ"] < 30)]),
               ("低成长 0~10", inb[(inb["YSTZ"] >= 0) & (inb["YSTZ"] < 10)]),
               ("负增长 <0", inb[inb["YSTZ"] < 0])]:
    if len(g) < 30:
        print("      %-18s n=%d 样本不足" % (lab, len(g)))
        continue
    pt, lo, hi, nb = block_boot(daily(g), dmap)
    v = "显著" if lo > 0 else ("显著负" if hi < 0 else "含零")
    print("      %-18s n=%-5d 超额 %+6.2f%%  CI[%+5.2f,%+5.2f]  胜率%.1f%%  %s" % (
        lab, len(g), pt * 100, lo * 100, hi * 100,
        (g["ab_fwd60"] > 0).mean() * 100, v))

print()
print("  【成交额门槛归因】（prox>=0.95、箱宽<=15%、换手0~6% 固定）")
print("  %-16s %8s %11s %8s %10s" % ("成交额下限", "事件数", "高成长占比", "beta", "增速中位"))
AB = ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) &
        (ev["turn30"] >= 0.0) & (ev["turn30"] <= 6.0)]
for a in [1e8, 2e8, 4e8, 8e8]:
    g = AB[AB["amt30"] >= a]
    print("  %-16s %8d %10.1f%% %8.2f %9.1f%%" % (
        ">=%.0f亿" % (a / 1e8), len(g), (g["YSTZ"] >= HI).mean() * 100,
        g["beta"].median(), g["YSTZ"].median()))

print()
print("=" * 100)
print("D. 当下（2026-10-27 收盘）全市场还有没有成长票可选？")
print("=" * 100)
try:
    from yh_panel import build_panel
    p = build_panel(asof_offset=0, days=320)
    p = p.dropna(subset=["prox_now", "width30", "amt30", "turn30"])
    print("  面板股票数 %d" % len(p))
    u = p[(p["prox_now"] >= 0.95) & (p["amt30"] >= 4e8) & (p["turn30"] <= 6.0)]
    print("  过【距年高+成交额+换手】三道门（不含箱宽）：%d 只" % len(u))
    for w in [0.15, 0.20, 0.25, 0.30]:
        s = u[u["width30"] <= w]
        hg = s[s["rev_yoy"] >= 30]
        print("    箱宽<=%-4.0f%%  %4d 只   其中营收增速>=30%%: %d 只  >=20%%: %d 只" % (
            w * 100, len(s), len(hg), (s["rev_yoy"] >= 20).sum()))
    s15 = u[u["width30"] <= 0.15]
    print()
    print("  箱宽<=15%% 的 %d 只，营收增速分布：" % len(s15))
    q = s15["rev_yoy"]
    print("    中位 %.1f%%  四分位 %.1f%% / %.1f%%  最大 %.1f%%  最小 %.1f%%" % (
        q.median(), q.quantile(.25), q.quantile(.75), q.max(), q.min()))
    print("    >=30%% 的 %d 只：%s" % (
        (q >= 30).sum(),
        "、".join("%s(%s,%.0f%%)" % (r["name"], r.name, r["rev_yoy"])
                  for _, r in s15[q >= 30].iterrows()) or "无"))
except Exception as e:
    print("  [跳过] 面板加载失败：%s" % e)
