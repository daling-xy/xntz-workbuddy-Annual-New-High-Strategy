# -*- coding: utf-8 -*-
"""
三项补充诊断：D3 量能（新定义） / D6 净利同比 SJLTZ / RS 相对强度
================================================================
背景：
  D3（量能确认）：classic_gap 用的 vol_dry = 近10日额/近30日额，与箱宽/prox 高度共线。
    本脚本换用 Minervini 语义的【事件日量比】 vr = 当日成交额 / 近20日日均成交额，
    并在「真突破」子样本（prox_now>=0.999）上单独检验"突破日放量"假设。
  D6（盈利增速）：新财报缓存 SJLTZ 填充率 99.5%，可测 CANSLIM 的 C（当季盈利增速）。
    用【法定披露截止日】做无前视 merge（复用 yh_rev_fine._avail_date 逻辑）。
  RS（相对强度）：O'Neil RS = 过去12个月价格表现的全市场百分位。
    rs_pct = 当日全市场 ret250 的百分位排名（不涉及指数，与 IBD RS rating 同义）。

检验口径：当前 v2 完整结构门（prox_now>=0.95, 箱宽<=15%, 额>=4亿, 换手0~6%,
          ret60<=19%），配对区块 bootstrap（区块=60交易日，90%CI）。
"""
from __future__ import annotations

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_blockboot_all import make_dmap, block_boot, daily  # noqa: E402
from yh_rev_fine import _avail_date  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "d3_d6_rs.txt")

_lines = []


def P(s=""):
    print(s, flush=True)
    _lines.append(str(s))


def roll_mean(x, w):
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    s = np.maximum(0, idx - w + 1)
    return (csum[idx + 1] - csum[s]) / (idx - s + 1)


# ---------------------------------------------------------------- 事件表
def load_full():
    ev = pd.read_pickle(os.path.join(CACHE, "events_fixed.pkl"))
    if "beta" in ev.columns and "mkt_fwd60" in ev.columns:
        lo, hi = ev["beta"].quantile([0.005, 0.995])
        ev["beta"] = ev["beta"].clip(lo, hi)
        ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    else:
        ev["ab_fwd60"] = ev["ex_fwd60"]
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]
    ev["date"] = pd.to_datetime(ev["date"])
    return ev


def gate_v2(ev):
    """当前 v2 完整结构门（含 ret60<=19%）。"""
    return ev[(ev["prox_now"] >= 0.95) & (ev["width30"] <= 0.15) &
              (ev["amt30"] >= 4e8) & (ev["turn30"] <= 6.0) &
              (pd.to_numeric(ev["ret60"], errors="coerce") <= 0.19) &
              ev["ab_fwd60"].notna()].copy()


# ---------------------------------------------------------------- kline 特征长表
def build_kline_feat():
    """全市场长表：code, date, vr(当日额/近20日均额), ret250。
    只保留 ret250 有效的位置（>=250根bar），供 RS 当日横截面百分位使用。"""
    p = os.path.join(CACHE, "kline_feat.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    kmap = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    codes, dts, vrs, r250s = [], [], [], []
    for code, bars in kmap.items():
        n = len(bars)
        if n < 270:
            continue
        d = [b["date"] for b in bars]
        c = np.array([b["close"] for b in bars], dtype=float)
        amt = np.array([b.get("amount", 0.0) for b in bars], dtype=float)
        ma20a = roll_mean(amt, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = amt / np.where(ma20a > 0, ma20a, np.nan)
            ret250 = c / np.roll(c, 250) - 1.0
        ret250[:250] = np.nan
        m = ~np.isnan(ret250)
        codes.extend([code] * int(m.sum()))
        dts.extend([dd for dd, ok in zip(d, m) if ok])
        vrs.append(vr[m])
        r250s.append(ret250[m])
    df = pd.DataFrame({
        "code": codes,
        "date": pd.to_datetime(dts),
        "vr": np.concatenate(vrs),
        "ret250": np.concatenate(r250s),
    })
    # RS：当日全市场 ret250 百分位（IBD RS rating 同义，0~100）
    df["rs_pct"] = df.groupby("date")["ret250"].rank(pct=True) * 100.0
    df.to_pickle(p)
    return df


# ---------------------------------------------------------------- SJLTZ 无前视贴表
def attach_sjltz(d: pd.DataFrame) -> pd.DataFrame:
    fh = pd.read_pickle(os.path.join(CACHE, "fin_history.pkl"))
    f = fh[["SECURITY_CODE", "REPORTDATE", "SJLTZ"]].copy()
    f = f.rename(columns={"SECURITY_CODE": "code"})
    f["SJLTZ"] = pd.to_numeric(f["SJLTZ"], errors="coerce")
    f["date"] = _avail_date(f["REPORTDATE"])
    f = f.dropna(subset=["date"]).sort_values("date")
    f = f.groupby(["code", "date"], as_index=False)["SJLTZ"].last()
    d = d.sort_values("date")
    f = f.sort_values("date")          # merge_asof 要求右表按 on 键全局有序
    out = pd.merge_asof(d, f[["code", "date", "SJLTZ"]],
                        on="date", by="code", direction="backward")
    return out


# ---------------------------------------------------------------- 检验工具
def band_test(d, dmap, bins, label):
    P("\n" + "-" * 100)
    P("%s" % label)
    P("-" * 100)
    P("  %-24s %7s %10s %22s %10s %9s" % ("分档", "事件数", "点估计", "90%CI", "胜率", "大亏率"))
    for lab, m in bins:
        s = d[m(d)]
        if len(s) < 30:
            P("  %-24s %7d %10s %22s %10s %9s" % (lab, len(s), "样本不足", "—", "—", "—"))
            continue
        pt, lo, hi, nb = block_boot(daily(s, "ab_fwd60"), dmap)
        win = float((s["ab_fwd60"] > 0).mean())
        bad = float((s["ab_fwd60"] < -0.20).mean())
        v = "显著" if (lo > 0 or hi < 0) else "含零"
        P("  %-24s %7d %9.2f%% [%+6.2f,%+6.2f]%-6s %8.1f%% %8.1f%%  区块%d" % (
            lab, len(s), pt * 100, lo * 100, hi * 100, v, win * 100, bad * 100, nb))


def pair_test(d, dmap, maskA, maskB, labA, labB):
    a = d[maskA(d)].groupby("date")["ab_fwd60"].mean()
    b = d[maskB(d)].groupby("date")["ab_fwd60"].mean()
    idx = a.index.intersection(b.index)
    if len(idx) < 20:
        P("  %s vs %s：配对日不足（%d）" % (labA, labB, len(idx)))
        return
    diff = (a.loc[idx] - b.loc[idx]).rename("ab_fwd60")
    pt, lo, hi, nb = block_boot(diff, dmap)
    v = "显著" if (lo > 0 or hi < 0) else "含零"
    P("  %-28s vs %-26s 差%+6.2fpp [%+6.2f,%+6.2f] %-6s 配对日%d 区块%d" % (
        labA, labB, pt * 100, lo * 100, hi * 100, v, len(idx), nb))


def spear(d, cols):
    P("\n  Spearman 相关（共线性检查，门内事件）:")
    sub = d[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman")
    for a in cols:
        P("  " + "  ".join("%s=%.2f" % (b, corr.loc[a, b]) for b in cols if b != a))


# ---------------------------------------------------------------- main
def main():
    ev = load_full()
    d = gate_v2(ev)
    dmap = make_dmap(ev)
    P("=" * 100)
    P("三项补充诊断 · 当前 v2 完整结构门内（含 ret60<=19%）")
    P("事件 %d 条，%d 只，%d 个交易日" % (len(d), d["code"].nunique(), d["date"].nunique()))
    P("=" * 100)

    # ---------- kline 特征 ----------
    P("\n[1] 构建全市场 kline 特征（vr 当日量比 / rs_pct）...")
    feat = build_kline_feat()
    P("    长表 %d 行，覆盖 %s ~ %s" % (
        len(feat), feat["date"].min().date(), feat["date"].max().date()))
    d = pd.merge(d, feat[["code", "date", "vr", "rs_pct"]], on=["code", "date"], how="left")

    # ================================================================== D3
    P("\n" + "=" * 100)
    P("D3 量能确认（新定义：vr = 当日成交额 / 近20日日均成交额，Minervini 语义）")
    P("=" * 100)
    P("  事件表 vr 覆盖率：%.1f%%" % (100 * d["vr"].notna().mean()))
    band_test(d, dmap, [
        ("缩量 vr<0.8", lambda x: x["vr"] < 0.8),
        ("正常 0.8~1.5", lambda x: (x["vr"] >= 0.8) & (x["vr"] < 1.5)),
        ("放量 1.5~3", lambda x: (x["vr"] >= 1.5) & (x["vr"] < 3)),
        ("显著放量 vr>=3", lambda x: x["vr"] >= 3),
    ], "全部门内事件")
    P("\n  配对检验：")
    pair_test(d, dmap, lambda x: x["vr"] >= 1.5, lambda x: x["vr"] < 0.8,
              "放量 vr>=1.5", "缩量 vr<0.8")
    pair_test(d, dmap, lambda x: (x["vr"] >= 1.4) & (x["vr"] < 2.0),
              lambda x: x["vr"] < 1.0, "适度放量 1.4~2.0(Minervini)", "缩量 vr<1.0")
    spear(d, ["vr", "width30", "prox_now", "turn30", "ret60"])

    # 真突破子样本（Minervini 的"突破日放量"原语义）
    dn = d[d["prox_now"] >= 0.999].copy()
    P("\n  【真突破子样本】prox_now>=0.999，n=%d：" % len(dn))
    band_test(dn, dmap, [
        ("缩量 vr<0.8", lambda x: x["vr"] < 0.8),
        ("正常 0.8~1.5", lambda x: (x["vr"] >= 0.8) & (x["vr"] < 1.5)),
        ("放量 1.5~3", lambda x: (x["vr"] >= 1.5) & (x["vr"] < 3)),
        ("显著放量 vr>=3", lambda x: x["vr"] >= 3),
    ], "真突破事件")
    pair_test(dn, dmap, lambda x: x["vr"] >= 1.5, lambda x: x["vr"] < 0.8,
              "突破日放量>=1.5", "突破日缩量<0.8")

    # ================================================================== RS
    P("\n" + "=" * 100)
    P("RS 相对强度（rs_pct = 当日全市场 ret250 百分位，O'Neil RS rating 同义）")
    P("=" * 100)
    P("  事件表 rs_pct 覆盖率：%.1f%%" % (100 * d["rs_pct"].notna().mean()))
    band_test(d, dmap, [
        ("弱 RS<50", lambda x: x["rs_pct"] < 50),
        ("中 50~70", lambda x: (x["rs_pct"] >= 50) & (x["rs_pct"] < 70)),
        ("强 70~85", lambda x: (x["rs_pct"] >= 70) & (x["rs_pct"] < 85)),
        ("领袖 RS>=85", lambda x: x["rs_pct"] >= 85),
    ], "全部门内事件")
    P("\n  配对检验：")
    pair_test(d, dmap, lambda x: x["rs_pct"] >= 85, lambda x: x["rs_pct"] < 50,
              "RS>=85(领头羊)", "RS<50")
    pair_test(d, dmap, lambda x: x["rs_pct"] >= 70, lambda x: x["rs_pct"] < 70,
              "RS>=70", "RS<70")
    spear(d, ["rs_pct", "ret60", "ret250", "prox_now", "vr"])

    # 交叉格：长期强 + 短期不热（CANSLIM L 与 ret60 门的交叉验证）
    P("\n  【交叉格】RS>=85 且 ret60<=中位数 vs 其余（『长期强短期冷』是否更优）：")
    med = d["ret60"].median()
    pair_test(d, dmap,
              lambda x: (x["rs_pct"] >= 85) & (x["ret60"] <= med),
              lambda x: ~((x["rs_pct"] >= 85) & (x["ret60"] <= med)),
              "RS>=85且低涨幅", "其余")

    # ================================================================== D6
    P("\n" + "=" * 100)
    P("D6 净利同比 SJLTZ（CANSLIM 的 C：当季盈利增速）")
    P("=" * 100)
    d2 = attach_sjltz(d)
    cov = d2["SJLTZ"].notna().mean()
    P("  事件表 SJLTZ 覆盖率：%.1f%%（受 fin_history 报告期覆盖限制）" % (100 * cov))
    P("  覆盖的日期范围：%s ~ %s" % (
        d2.loc[d2["SJLTZ"].notna(), "date"].min().date(),
        d2.loc[d2["SJLTZ"].notna(), "date"].max().date()))
    s6 = d2[d2["SJLTZ"].notna()].copy()
    band_test(s6, dmap, [
        ("负增长 <0", lambda x: x["SJLTZ"] < 0),
        ("0~30%", lambda x: (x["SJLTZ"] >= 0) & (x["SJLTZ"] < 30)),
        ("30~100%", lambda x: (x["SJLTZ"] >= 30) & (x["SJLTZ"] < 100)),
        (">=100%", lambda x: x["SJLTZ"] >= 100),
    ], "门内事件（SJLTZ 覆盖段）")
    P("\n  配对检验：")
    pair_test(s6, dmap, lambda x: x["SJLTZ"] >= 25, lambda x: x["SJLTZ"] < 0,
              "净利>=25%(CANSLIM C)", "净利<0")
    pair_test(s6, dmap, lambda x: x["SJLTZ"] >= 25, lambda x: x["SJLTZ"] < 25,
              "净利>=25%", "净利<25%")
    # 盈利增速 vs 营收增速（同为 2025 后样本的对照）
    from yh_rev_fine import attach_fin  # noqa: E402
    fh = pd.read_pickle(os.path.join(CACHE, "fin_history.pkl"))
    s6 = attach_fin(s6, fh)
    if "rev_yoy" in s6.columns:
        P("\n  【同段对照】营收同比 YSTZ 分档（同 SJLTZ 覆盖段）：")
        band_test(s6, dmap, [
            ("负增长 <0", lambda x: x["rev_yoy"] < 0),
            ("0~30%", lambda x: (x["rev_yoy"] >= 0) & (x["rev_yoy"] < 30)),
            (">=30%", lambda x: x["rev_yoy"] >= 30),
        ], "营收同比（同段）")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    P("\n已输出到: %s" % OUT)


if __name__ == "__main__":
    main()
