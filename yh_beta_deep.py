# -*- coding: utf-8 -*-
"""
深挖四个问题：
  A. Var / Cov / 相关系数 ρ / 标准差 σ 四者是什么关系，在真实市场里各是多少
  B. β 为什么用 250 天：不同窗口下 β 稳不稳
  C. 一个「事件」到底长什么样（回答：事件 = 一只票 × 一个日期，不是"簇"）
  D. 事件表 vs 筛选条件：策略的门槛是事后套上去的 mask
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_beta import build_market_ret  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def beta_of(cl, m, i, w):
    """用 i 之前 w 天算 beta，返回 (beta, rho, sigma_stock, sigma_mkt, cov, var)"""
    if i - w + 1 < 1:
        return (np.nan,) * 6
    r = np.empty(len(cl)); r[0] = np.nan
    r[1:] = cl[1:] / np.maximum(cl[:-1], 1e-9) - 1.0
    j = i - w + 1
    rr, mm = r[j:i+1], np.asarray(m)[j:i+1]
    ok = np.isfinite(rr) & np.isfinite(mm)
    rr, mm = rr[ok], mm[ok]
    if len(rr) < w * 0.8:
        return (np.nan,) * 6
    n = len(rr)
    dr, dm = rr - rr.mean(), mm - mm.mean()
    cov = float((dr * dm).sum()) / (n - 1)
    var = float((dm ** 2).sum()) / (n - 1)
    vr = float((dr ** 2).sum()) / (n - 1)
    sr, sm = np.sqrt(vr), np.sqrt(var)
    rho = cov / (sr * sm) if sr > 0 and sm > 0 else np.nan
    return cov / var if var > 1e-12 else np.nan, rho, sr, sm, cov, var


def main():
    mkt = build_market_ret()
    mv = mkt.values
    mpos = {d: k for k, d in enumerate(mkt.index)}
    with open(os.path.join(CACHE, "kline_long2.pkl"), "rb") as f:
        kmap = pickle.load(f)

    print("=" * 100)
    print("A. Var / Cov / 相关系数 ρ / 标准差 σ 的关系（在真实市场里各是多少）")
    print("=" * 100)
    print("""
  方差   Var(X) = Σ(x−x̄)²/(n−1)      单位 = 百分比的平方 → 不直观，所以开方
  标准差 σ(X)   = √Var(X)             单位 = 百分比       → 【用这个说"波动多大"】
  协方差 Cov(X,Y)= Σ(x−x̄)(y−ȳ)/(n−1)  单位 = 百分比的平方 → 也不直观
  相关系数 ρ    = Cov(X,Y) ÷ (σ_X·σ_Y) 无单位，锁死在 [−1, +1]

  四者的桥：  β = Cov(r,m) ÷ Var(m) = ρ × (σ_股票 ÷ σ_市场)
              ↑ 协方差不是相关系数！相关系数是协方差"除以两个标准差"标准化后的版本
""")
    # 市场整体波动
    mfin = mv[np.isfinite(mv)]
    # 注：原始序列有 2 个数据瑕疵日（样本股极少导致等权收益爆掉），用 1% 截尾稳健估计
    mrob = mfin[(mfin > -0.20) & (mfin < 0.20)]
    print(f"  【市场基准 = 等权全A日收益】共 {len(mfin)} 个交易日（剔除 2 个瑕疵日后 {len(mrob)} 天）")
    print(f"    日收益标准差 σ_m = {mrob.std(ddof=1)*100:.3f}%")
    print(f"    年化波动        = {mrob.std(ddof=1)*np.sqrt(250)*100:.1f}%")
    print(f"    Var(m)          = {mrob.var(ddof=1)*1e4:.2f}（万分之一单位）")
    print()

    for code in ["600030", "601108", "600519", "300209"]:
        bars = kmap.get(code)
        if not bars:
            continue
        dates = [b["date"] for b in bars]
        if "2026-10-21" not in dates:
            continue
        i = dates.index("2026-10-21")
        cl = np.array([float(b["close"]) for b in bars])
        m = np.array([mv[mpos[d]] if d in mpos else np.nan for d in dates])
        b, rho, sr, sm, cov, var = beta_of(cl, m, i, 250)
        name = bars[i].get("name", "")
        print(f"  {code} {name:<6} σ_股票={sr*100:5.2f}%  σ_市场={sm*100:5.2f}%  "
              f"ρ={rho:5.2f}  Cov={cov*1e4:7.3f}  Var={var*1e4:7.3f}  "
              f"→ β = ρ×(σ股/σ市) = {rho:.2f}×{sr/sm:.2f} = {b:.3f}")
    print()

    print("=" * 100)
    print("B. β 为什么用 250 天？—— 不同窗口下 β 稳不稳")
    print("=" * 100)
    print(f"  {'股票':<16}{'窗口':>8}{'β':>9}{'ρ':>8}{'σ股/σ市':>10}")
    for code in ["600030", "600519"]:
        bars = kmap.get(code)
        dates = [b["date"] for b in bars]
        i = dates.index("2026-10-21")
        cl = np.array([float(b["close"]) for b in bars])
        m = np.array([mv[mpos[d]] if d in mpos else np.nan for d in dates])
        name = bars[i].get("name", "")
        for w in [60, 120, 250, 500]:
            b, rho, sr, sm, cov, var = beta_of(cl, m, i, w)
            if np.isnan(b):
                continue
            print(f"  {code+' '+name:<16}{w:>6}日{b:>9.3f}{rho:>8.2f}{sr/sm:>10.2f}")
        print()

    print("  同一只票、不同【时点】用 250 天算出来的 β（看它自己漂不漂）：")
    print(f"  {'股票':<16}{'时点':>12}{'β(250日)':>12}")
    for code in ["600030", "600519"]:
        bars = kmap.get(code)
        dates = [b["date"] for b in bars]
        cl = np.array([float(b["close"]) for b in bars])
        m = np.array([mv[mpos[d]] if d in mpos else np.nan for d in dates])
        name = bars[-1].get("name", "")
        for asof in ["2024-06-20", "2025-06-20", "2026-06-22", "2026-10-21"]:
            cand = [d for d in dates if d <= asof]
            if len(cand) < 260:
                continue
            i = dates.index(cand[-1])
            b, *_ = beta_of(cl, m, i, 250)
            if not np.isnan(b):
                print(f"  {code+' '+name:<16}{cand[-1]:>12}{b:>12.3f}")
        print()

    print("=" * 100)
    print("C. 一个「事件」长什么样   —— 事件 = 一只票 × 一个日期，不是 bootstrap 里的「簇」")
    print("=" * 100)
    ev = pd.read_pickle(os.path.join(CACHE, "events_long_beta.pkl"))
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    cols = ["code", "date", "close", "prox_now", "width30", "amt30", "turn30",
            "regime", "beta", "fwd60", "mkt_fwd60", "ex_fwd60", "ab_fwd60"]
    print("  随机抽 5 个事件（每行 = 一只票在某一天的完整快照）：")
    print(ev[cols].dropna(subset=["beta"]).sample(5, random_state=7).to_string(index=False))
    print()
    print(f"  事件总数 = {len(ev)}   涉及股票数 = {ev['code'].nunique()}   "
          f"覆盖交易日 = {ev['date'].nunique()}")
    print(f"  平均每个交易日有多少个事件 = {len(ev)/ev['date'].nunique():.0f}")
    print()
    print("  逐行验证「去均值超额」和「β调整超额」是怎么来的（拿上面其中一个事件算）：")
    # 挑一个 β 远离 0、市场收益也正常的事件，避免取到上市首日（窗口不足）
    pool = ev[(ev["beta"].abs() > 0.3) & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]
    one = pool[cols].sample(1, random_state=3).iloc[0]
    print(f"    股票 {one['code']}")
    print(f"    日期 {one['date']}  β={one['beta']:.3f}")
    print(f"    fwd60（这票未来60天）      = {one['fwd60']*100:+.2f}%")
    print(f"    mkt_fwd60（市场未来60天）   = {one['mkt_fwd60']*100:+.2f}%")
    print(f"    β × 市场                  = {one['beta']*one['mkt_fwd60']*100:+.2f}%")
    print(f"    α = fwd60 − β×mkt_fwd60   = {one['fwd60']*100:.2f}% − "
          f"{one['beta']*one['mkt_fwd60']*100:.2f}% = {one['ab_fwd60']*100:+.2f}%")
    print()

    print("=" * 100)
    print("D. 事件表 vs 筛选条件 —— 策略的门槛是事后套上去的 mask")
    print("=" * 100)
    dn = ev[(ev["regime"] == "DOWN") & ev["beta"].notna()]
    print(f"  下跌市全部事件（原料库，不设任何门槛）        : {len(dn):>8}")
    for tag, mask in [
        ("+ 当前价/年高>=0.95", dn["prox_now"] >= 0.95),
        ("+ 箱宽<=15%", (dn["prox_now"] >= 0.95) & (dn["width30"] <= 0.15)),
        ("+ 额>=4亿", (dn["prox_now"] >= 0.95) & (dn["width30"] <= 0.15) & (dn["amt30"] >= 4e8)),
        ("+ 换手<=6%【完整策略】", (dn["prox_now"] >= 0.95) & (dn["width30"] <= 0.15)
                                & (dn["amt30"] >= 4e8) & (dn["turn30"] <= 6.0)),
    ]:
        d = dn[mask]
        print(f"    {tag:<28}: {len(d):>8}   β调整超额 {d['ab_fwd60'].mean()*100:+.2f}%")
    print()
    print("  关键：事件表本身【不含任何策略门槛】，它是全市场×全历史的原料库；")
    print("        策略的门槛是事后在事件表上套 mask，套一次筛出一批事件，再算这批的 α 均值。")


if __name__ == "__main__":
    main()
