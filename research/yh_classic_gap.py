# -*- coding: utf-8 -*-
"""
经典「年度新高 / 52周新高」策略 vs 我们现行 v2 —— 背离点实证检验
================================================================
文献基线（检索所得，脚本内附出处）：
  1. George & Hwang (2004, JF)    ：52周高比值 = 现价/52周高，越高越好；锚定效应
  2. 国盛金工《创新高股票中的Alpha》：前期涨跌幅低 / 前期换手率低 → 反应不足 → 后续超额高
  3. Minervini VCP / O'Neil CANSLIM：需趋势确认（站上50/150/200日线）+ 突破放量确认
  4. Darvas 箱体理论              ：箱体上沿突破 + 放量 = 买点；跌破箱底止损
  5. Alpha Suite 52w-High 实现    ：距高5%以内 + 站上50/200日线 + 20日均量>60日均量10%
                                    + "不在下跌趋势中追新高"

检验口径：结构门内（prox_now>=0.95, 箱宽<=15%, 额>=4亿, 换手0~6%）
          配对区块 bootstrap（区块=60交易日，90%CI）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_blockboot_all import make_dmap, block_boot, daily   # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_full():
    p = os.path.join(CACHE, "events_fixed.pkl")
    ev = pd.read_pickle(p)
    print("事件表: %s 行, %s 只股票" % (len(ev), ev["code"].nunique()))
    print("可用列: %s" % ", ".join(sorted(ev.columns.tolist())))
    if "beta" in ev.columns and "mkt_fwd60" in ev.columns:
        lo, hi = ev["beta"].quantile([0.005, 0.995])
        ev["beta"] = ev["beta"].clip(lo, hi)
        ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    else:
        ev["ab_fwd60"] = ev["ex_fwd60"]
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]
    return ev


def gate(ev):
    return ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) &
              (ev["amt30"] >= 4e8) & (ev["turn30"] <= 6.0) &
              ev["ab_fwd60"].notna()]


def band_test(d, dmap, col, bins, label, col_name):
    """按某因子分档，报告各档 ab_fwd60 的区块 bootstrap 点估计与 90%CI"""
    print("\n" + "-" * 100)
    print("%s（分档变量：%s）" % (label, col_name))
    print("-" * 100)
    print("  %-22s %7s %10s %20s %10s %9s" % ("分档", "事件数", "点估计", "90%CI", "胜率", "大亏率"))
    for lab, m in bins:
        s = d[m(d)]
        if len(s) < 30:
            print("  %-22s %7d %10s %20s %10s %9s" % (lab, len(s), "样本不足", "—", "—", "—"))
            continue
        pt, lo, hi, nb = block_boot(daily(s, "ab_fwd60"), dmap)
        win = float((s["ab_fwd60"] > 0).mean())
        bad = float((s["ab_fwd60"] < -0.20).mean())
        v = "显著" if (lo > 0 or hi < 0) else "含零"
        print("  %-22s %7d %9.2f%% [%+6.2f,%+6.2f]%-6s %8.1f%% %8.1f%%" % (
            lab, len(s), pt * 100, lo * 100, hi * 100, v, win * 100, bad * 100))
    return


def pair_test(d, dmap, maskA, maskB, labA, labB):
    """A 组 vs B 组，配对（同交易日做差）+ 区块 bootstrap"""
    a = d[maskA(d)].groupby("date")["ab_fwd60"].mean()
    b = d[maskB(d)].groupby("date")["ab_fwd60"].mean()
    idx = a.index.intersection(b.index)
    if len(idx) < 20:
        print("  %s vs %s：配对日不足（%d）" % (labA, labB, len(idx)))
        return
    diff = (a.loc[idx] - b.loc[idx]).rename("ab_fwd60")
    pt, lo, hi, nb = block_boot(diff, dmap)
    v = "显著" if (lo > 0 or hi < 0) else "含零"
    print("  %-26s vs %-24s 差%+6.2fpp [%+6.2f,%+6.2f] %s  配对日%d 区块%d" % (
        labA, labB, pt * 100, lo * 100, hi * 100, v, len(idx), nb))


def main():
    ev = load_full()
    d = gate(ev)
    dmap = make_dmap(ev)
    print("\n结构门内事件: %d 条, %d 只, %d 个交易日" % (
        len(d), d["code"].nunique(), d["date"].nunique()))

    # ==================================================================
    print("\n" + "=" * 100)
    print("D1 【最核心】已突破 vs 箱内蓄势 —— 用户观察到的分组，谁的后续超额更高？")
    print("=" * 100)
    band_test(d, dmap, None, [
        ("创年新高 prox>=0.999", lambda x: x["prox_now"] >= 0.999),
        ("贴年高 0.99~0.999", lambda x: (x["prox_now"] >= 0.99) & (x["prox_now"] < 0.999)),
        ("箱内 0.98~0.99", lambda x: (x["prox_now"] >= 0.98) & (x["prox_now"] < 0.99)),
        ("箱内 0.97~0.98", lambda x: (x["prox_now"] >= 0.97) & (x["prox_now"] < 0.98)),
        ("箱内 0.95~0.97", lambda x: (x["prox_now"] >= 0.95) & (x["prox_now"] < 0.97)),
    ], "按距年高分档", "prox_now")

    print("\n  配对检验：")
    pair_test(d, dmap, lambda x: x["prox_now"] >= 0.999,
              lambda x: x["prox_now"] < 0.999, "创年新高(prox>=0.999)", "未创新高(<0.999)")
    pair_test(d, dmap, lambda x: x["prox_now"] >= 0.98,
              lambda x: x["prox_now"] < 0.98, "prox>=0.98", "prox<0.98")

    # ==================================================================
    print("\n" + "=" * 100)
    print("D2 前期涨跌幅（国盛金工核心因子：涨幅低=反应不足=后续超额高）")
    print("=" * 100)
    for c, name in [("ret60", "近60日涨幅"), ("ret250", "近250日涨幅")]:
        if c not in d.columns:
            print("  （无 %s 列，跳过）" % c)
            continue
        q = d[c].quantile([0.25, 0.5, 0.75])
        band_test(d, dmap, None, [
            ("最低25%% (<=%.0f%%)" % (q[0.25] * 100), lambda x, c=c, v=q[0.25]: x[c] <= v),
            ("25~50%%", lambda x, c=c, a=q[0.25], b=q[0.5]: (x[c] > a) & (x[c] <= b)),
            ("50~75%%", lambda x, c=c, a=q[0.5], b=q[0.75]: (x[c] > a) & (x[c] <= b)),
            ("最高25%% (>%.0f%%)" % (q[0.75] * 100), lambda x, c=c, v=q[0.75]: x[c] > v),
        ], name, c)
        print("  配对检验（最低25%% − 最高25%%）：")
        pair_test(d, dmap, lambda x, c=c, v=q[0.25]: x[c] <= v,
                  lambda x, c=c, v=q[0.75]: x[c] > v, name + "最低25%", name + "最高25%")

    # ==================================================================
    print("\n" + "=" * 100)
    print("D3 量能确认（Darvas/Minervini 要求突破放量；国盛称放量有正超额但未采用）")
    print("=" * 100)
    if "vol_dry" in d.columns:
        q = d["vol_dry"].quantile([0.25, 0.5, 0.75])
        band_test(d, dmap, None, [
            ("显著缩量 (<=%.2f)" % q[0.25], lambda x, v=q[0.25]: x["vol_dry"] <= v),
            ("偏缩量", lambda x, a=q[0.25], b=q[0.5]: (x["vol_dry"] > a) & (x["vol_dry"] <= b)),
            ("偏放量", lambda x, a=q[0.5], b=q[0.75]: (x["vol_dry"] > a) & (x["vol_dry"] <= b)),
            ("显著放量 (>%.2f)" % q[0.75], lambda x, v=q[0.75]: x["vol_dry"] > v),
        ], "近10日额/近30日额（vol_dry）", "vol_dry")
        print("  配对检验（显著放量 − 显著缩量）：")
        pair_test(d, dmap, lambda x, v=q[0.75]: x["vol_dry"] > v,
                  lambda x, v=q[0.25]: x["vol_dry"] <= v, "显著放量", "显著缩量")

    # ==================================================================
    print("\n" + "=" * 100)
    print("D4 趋势/均线确认（Minervini Trend Template、CANSLIM 的 M、Alpha Suite）")
    print("=" * 100)
    if all(c in d.columns for c in ["close", "ma60", "ma120", "ma20"]):
        d = d.copy()
        d["above60"] = (d["close"] > d["ma60"]).astype(float)
        d["above120"] = (d["close"] > d["ma120"]).astype(float)
        d["stack"] = d["above60"] + d["above120"] + (d["close"] > d["ma20"]).astype(float)
        band_test(d, dmap, None, [
            ("站上全部3条均线(3)", lambda x: x["stack"] >= 3),
            ("站上2条", lambda x: x["stack"] == 2),
            ("站上1条", lambda x: x["stack"] == 1),
            ("全部跌破(0)", lambda x: x["stack"] <= 0),
        ], "均线站上条数（20/60/120）", "stack")
        print("  配对检验（站上全部 − 未全站上）：")
        pair_test(d, dmap, lambda x: x["stack"] >= 3, lambda x: x["stack"] < 3,
                  "站上全部3条", "未全站上")

    # ==================================================================
    print("\n" + "=" * 100)
    print("D5 箱内位置（Darvas：应等突破箱顶，而非箱底埋伏）")
    print("=" * 100)
    if "pos_in_box" in d.columns:
        band_test(d, dmap, None, [
            ("箱顶 0.8~1.0", lambda x: x["pos_in_box"] >= 0.8),
            ("箱中 0.4~0.8", lambda x: (x["pos_in_box"] >= 0.4) & (x["pos_in_box"] < 0.8)),
            ("箱底 0.0~0.4", lambda x: x["pos_in_box"] < 0.4),
        ], "30日箱体内位置", "pos_in_box")
        print("  配对检验（箱顶 − 箱底）：")
        pair_test(d, dmap, lambda x: x["pos_in_box"] >= 0.8,
                  lambda x: x["pos_in_box"] < 0.4, "箱顶0.8~1.0", "箱底0~0.4")

    # ==================================================================
    print("\n" + "=" * 100)
    print("D6 盈利增速 vs 营收增速（经典用 EPS/盈利增速，我们此前测的是营收增速）")
    print("=" * 100)
    from yh_rev_fine import load_fin, attach_fin   # noqa: E402
    fin = load_fin()
    df2 = attach_fin(d, fin)
    for c, name in [("SJLTZ", "净利同比"), ("YSTZ", "营收同比")]:
        if c not in df2.columns:
            print("  （无 %s 列，跳过）" % c)
            continue
        s = df2[df2[c].notna()]
        band_test(s, dmap, None, [
            ("负增长 <0", lambda x, c=c: x[c] < 0),
            ("0~30%", lambda x, c=c: (x[c] >= 0) & (x[c] < 30)),
            ("30~100%", lambda x, c=c: (x[c] >= 30) & (x[c] < 100)),
            (">=100%", lambda x, c=c: x[c] >= 100),
        ], name, c)
        print("  配对检验（>=30%% − <0）：")
        pair_test(s, dmap, lambda x, c=c: x[c] >= 30, lambda x, c=c: x[c] < 0,
                  name + ">=30%", name + "<0")


if __name__ == "__main__":
    main()
