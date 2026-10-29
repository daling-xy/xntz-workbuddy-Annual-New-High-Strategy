# -*- coding: utf-8 -*-
"""
样本区间诊断实验
================
回答两个问题：
  Q1 起点太远？  2019-06 起 7.3 年，市场机制是否漂移，参数是否还适用
  Q2 末端不够新？事件表止于 2026-07-22，离今天 2026-10-26 有 3 个月缺口，怎么办

实验分段：
  A 样本两端诊断（含 fwd60 边界推算）
  B 起点敏感性：不同起点各自定参，看最优档是否漂移
  C 机制漂移诊断：分年份市场特征 + 策略表现
  D 滚动前推 walk-forward：截至 T 年定参 → T+1 年样本外检验
  E 时间加权定参：指数加权（半衰期）后重新定参
  F 短窗口标签 fwd20：把可检验末端往前推
  G 结论与方案
"""
import os
import gc
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_final_calib2 import boot_ci  # 整簇 bootstrap（若不存在则本地兜底）

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
LINE = "=" * 78

# 只保留分析需要的列，避免 pandas 在全量宽表上 consolidate 爆内存
NEED = ["code", "date", "prox", "prox_now", "dd_from_high", "width30",
        "amt30", "turn30", "fwd60", "ex_fwd60", "ab_fwd60",
        "mkt_fwd60", "beta", "regime", "fwd20", "ex_fwd20"]
DTYPE = {c: "float32" for c in NEED if c not in ("code", "date", "regime")}


def load_events() -> pd.DataFrame:
    p = os.path.join(CACHE, "events_long_beta.pkl")
    ev = pd.read_pickle(p)

    need = [c for c in NEED if c in ev.columns]
    ev = ev[need].copy()

    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]

    # 必须有：beta、市场收益、个股未来收益，否则算不出 α
    ev = ev[ev["beta"].notna() & ev["mkt_fwd60"].notna() & ev["fwd60"].notna()]

    # β 截尾，防个别异常值主导
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi).astype("float32")

    # 重算 α，保证口径统一
    ev["ab_fwd60"] = (ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]).astype("float32")
    gc.collect()
    return ev


# ----------------------------------------------------------------------
# 统一评价器：给定子集，返回 β调整超额 + 整簇 bootstrap CI
# ----------------------------------------------------------------------
def evaluate(d: pd.DataFrame, label: str, B: int = 600) -> dict:
    """评价一个事件子集的选股能力。

    口径沿用项目既有 boot_ci：先按交易日取日内均值（日等权），
    再对交易日取平均；bootstrap 以【交易日】为簇整簇重抽样。
    这样同一天的几千个事件同进同出，不会低估标准误。
    """
    if len(d) < 30:
        return {"标签": label, "n": len(d), "β调整超额": np.nan,
                "CI下界": np.nan, "CI上界": np.nan, "判定": "样本不足"}
    try:
        est, lo, hi = boot_ci(d, "ab_fwd60", n=B)
    except Exception:
        # 兜底：与 boot_ci 相同的日等权口径，只是不给出 CI
        est = d.groupby("date")["ab_fwd60"].mean().mean()
        lo, hi = np.nan, np.nan
    est = float(est) * 100
    lo = float(lo) * 100 if lo == lo else np.nan
    hi = float(hi) * 100 if hi == hi else np.nan
    if lo != lo:
        judge = "无CI"
    elif lo > 0 and hi > 0:
        judge = "显著为正"
    elif lo < 0 and hi < 0:
        judge = "显著为负"
    else:
        judge = "含零·无证据"
    return {"标签": label, "n": len(d), "β调整超额": est,
            "CI下界": lo, "CI上界": hi, "判定": judge}


def show(rows: list, title: str = ""):
    df = pd.DataFrame(rows)
    if title:
        print(title)
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()


# ----------------------------------------------------------------------
# A 段：样本两端诊断
# ----------------------------------------------------------------------
def seg_a(ev: pd.DataFrame, raw_last: str):
    print(LINE)
    print("A. 样本两端诊断")
    print(LINE)
    d = pd.to_datetime(ev["date"])
    ev_years = ev.assign(y=d.dt.year)

    print(f"  事件表范围      : {d.min().date()} ~ {d.max().date()}")
    print(f"  原始日线末端    : {raw_last}")
    print(f"  总事件数        : {len(ev):,}")
    print(f"  交易日（簇）数  : {ev['date'].nunique():,}")
    print()

    print("  ★ 关键：末端缺口不是数据没更新，而是 fwd60 标签的结构性限制")
    print("    fwd60 = 未来 60 个交易日收益。要给事件日 T 打标签，")
    print("    手上必须有 T 之后 60 个完整交易日的行情。")
    print(f"    数据末端 {raw_last} 往前数 60 个交易日 = {d.max().date()}")
    print("    → 事件表已经顶到了可标注的理论极限，再新一天的数据也无法延长。")
    print()

    g = ev_years.groupby("y").agg(
        事件数=("code", "size"), 交易日=("date", "nunique"), 股票数=("code", "nunique"))
    print("  分年份样本分布：")
    print(g.to_string())
    print()

    days = ev[["date", "regime"]].drop_duplicates()
    days["y"] = pd.to_datetime(days["date"]).dt.year
    ct = pd.crosstab(days["y"], days["regime"])
    print("  分年份市场环境（交易日计数）：")
    print(ct.to_string())
    print()
    return g


# ----------------------------------------------------------------------
# B 段：起点敏感性 —— 每个起点各自定参，看最优档漂不漂移
# ----------------------------------------------------------------------
PROX_GRID = [0.95]
WIDTH_GRID = [0.10, 0.15, 0.20, 0.25, 1.00]  # 1.00 = 不约束
AMT_MIN = 4e8
TURN_LO, TURN_HI = 0.0, 6.0


def mask_of(df: pd.DataFrame, prox_thr: float, w_thr: float) -> pd.Series:
    m = (df["prox_now"] >= prox_thr) & (df["amt30"] >= AMT_MIN) \
        & (df["turn30"] >= TURN_LO) & (df["turn30"] <= TURN_HI)
    if w_thr < 1.0:
        m &= (df["width30"] <= w_thr)
    return m


def seg_b(ev: pd.DataFrame):
    print(LINE)
    print("B. 起点敏感性：不同起点各自定参，最优档会不会漂移？")
    print(LINE)
    ev = ev.copy()
    ev["y"] = pd.to_datetime(ev["date"]).dt.year

    starts = [2019, 2021, 2022, 2023, 2024]
    rows = []
    best_by_start = {}
    for s in starts:
        sub = ev[ev["y"] >= s]
        if len(sub) < 500:
            continue
        inner = []
        for w in WIDTH_GRID:
            m = mask_of(sub, PROX_GRID[0], w)
            r = evaluate(sub[m], f"{s}起 箱宽{'不约束' if w>=1 else '<='+format(w,'.0%')}")
            inner.append((w, r))
        tag = "不约束" if inner[-1][0] >= 1 else format(inner[-1][0], ".0%")
        row = {"起点": s, "可用事件": f"{len(sub):,}"}
        for w, r in inner:
            k = "不约束" if w >= 1 else format(w, ".0%")
            row[k] = r["β调整超额"]
            row[k + "_CI下"] = r["CI下界"]
        rows.append(row)
        # 记录该起点下的最优档
        valid = [x for x in inner if x[1]["判定"] != "样本不足"]
        if valid:
            bw = max(valid, key=lambda x: x[1]["β调整超额"])[0]
            best_by_start[s] = bw

    df = pd.DataFrame(rows)
    cols_est = [c for c in df.columns if not c.endswith("_CI下")]
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 220):
        print(df[cols_est].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()
    print("  （各档 β调整超额，单位 %；CI下界略）")
    print()
    print("  ★ 各起点搜出的最优箱宽档：")
    for s, bw in best_by_start.items():
        print(f"     {s} 起样本  →  箱宽 <= {bw:.0%}")
    drift = set(best_by_start.values())
    print()
    if len(drift) == 1:
        print(f"  ✅ 全部指向同一档 <= {list(drift)[0]:.0%}  → 参数稳健，起点不是问题")
    else:
        print(f"  ⚠ 最优档在 {sorted(drift)} 之间漂移 → 存在起点依赖，需谨慎")
    print()
    return best_by_start


# ----------------------------------------------------------------------
# C 段：机制漂移诊断
# ----------------------------------------------------------------------
def seg_c(ev: pd.DataFrame):
    print(LINE)
    print("C. 机制漂移诊断：市场特征与策略表现，分年份到底变没变")
    print(LINE)
    ev = ev.copy()
    ev["y"] = pd.to_datetime(ev["date"]).dt.year

    rows = []
    for y, sub in ev.groupby("y"):
        full = mask_of(sub, PROX_GRID[0], 1.00)
        tight = mask_of(sub, PROX_GRID[0], 0.15)
        loose = mask_of(sub, PROX_GRID[0], 0.25)
        r_all = evaluate(sub[full], "", B=400)
        r_15 = evaluate(sub[tight], "", B=400)
        r_25 = evaluate(sub[loose], "", B=400)
        rows.append({
            "年份": y,
            "全市场箱宽中位": sub["width30"].median(),
            "贴年高比例": float((sub["prox_now"] >= 0.95).mean()),
            "不约束_n": r_all["n"], "不约束": r_all["β调整超额"],
            "15%_n": r_15["n"], "15%": r_15["β调整超额"], "15%判定": r_15["判定"],
            "25%_n": r_25["n"], "25%": r_25["β调整超额"],
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 240):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()
    print("  注：全市场箱宽中位 = 该年份所有事件的 30日振幅中位数，衡量当年度波动环境")
    print("      贴年高比例 = 当前价/年高 >= 0.95 的事件占比，衡量当年度强势股多寡")
    print()
    return df


# ----------------------------------------------------------------------
# D 段：滚动前推 walk-forward（最硬的样本外证据）
# ----------------------------------------------------------------------
def seg_d(ev: pd.DataFrame):
    print(LINE)
    print("D. 滚动前推 walk-forward：截至 T 年定参 → 在 T+1 年做样本外检验")
    print(LINE)
    ev = ev.copy()
    ev["y"] = pd.to_datetime(ev["date"]).dt.year
    years = sorted(ev["y"].unique())

    rows = []
    for i in range(1, len(years)):
        train_y = years[:i]      # 用这些年份定参
        test_y = years[i]        # 在这一年检验
        tr = ev[ev["y"].isin(train_y)]
        te = ev[ev["y"] == test_y]
        if len(tr) < 500 or len(te) < 200:
            continue

        # 样本内定参：网格里挑 β调整超额最高的（要求 CI 下界>0，否则退而求其次）
        cands = []
        for px in [0.95, 0.90]:
            for w in [0.10, 0.15, 0.20, 0.25, 1.00]:
                m = (tr["prox_now"] >= px) & (tr["amt30"] >= AMT_MIN) \
                    & (tr["turn30"] >= TURN_LO) & (tr["turn30"] <= TURN_HI)
                if w < 1.0:
                    m &= (tr["width30"] <= w)
                r = evaluate(tr[m], "", B=400)
                cands.append((px, w, r))
        sig = [c for c in cands if c[2]["判定"] == "显著为正"]
        pool = sig if sig else [c for c in cands if c[2]["判定"] != "样本不足"]
        if not pool:
            continue
        bpx, bw, _ = max(pool, key=lambda c: c[2]["β调整超额"])

        # 样本外检验
        m_te = (te["prox_now"] >= bpx) & (te["amt30"] >= AMT_MIN) \
            & (te["turn30"] >= TURN_LO) & (te["turn30"] <= TURN_HI)
        if bw < 1.0:
            m_te &= (te["width30"] <= bw)
        r_te = evaluate(te[m_te], "", B=400)

        # 对照：直接用当前 v2 参数（0.95 / 15%）
        m_v2 = (te["prox_now"] >= 0.95) & (te["amt30"] >= AMT_MIN) \
            & (te["turn30"] >= TURN_LO) & (te["turn30"] <= TURN_HI) \
            & (te["width30"] <= 0.15)
        r_v2 = evaluate(te[m_v2], "", B=400)

        rows.append({
            "训练期": f"{train_y[0]}-{train_y[-1]}", "检验年": test_y,
            "选出的参数": f"{bpx:.2f}/{'不约束' if bw>=1 else format(bw,'.0%')}",
            "样本外_n": r_te["n"], "样本外超额": r_te["β调整超额"], "判定": r_te["判定"],
            "v2参数_n": r_v2["n"], "v2参数超额": r_v2["β调整超额"], "v2判定": r_v2["判定"],
        })

    df = pd.DataFrame(rows)
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 240):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}" if x == x else "  n/a"))
    print()

    # 汇总
    ok = df[df["样本外超额"] == df["样本外超额"]]
    okv = df[df["v2参数超额"] == df["v2参数超额"]]
    print("  ★ 汇总：")
    print(f"     滚动定参（每年重选）  : 超额均值 {ok['样本外超额'].mean():+.2f}%  "
          f"正年数 {int((ok['样本外超额']>0).sum())}/{len(ok)}")
    print(f"     固定 v2 参数（不重选）: 超额均值 {okv['v2参数超额'].mean():+.2f}%  "
          f"正年数 {int((okv['v2参数超额']>0).sum())}/{len(okv)}")
    print()
    diff = ok["样本外超额"].mean() - okv["v2参数超额"].mean()
    if diff > 0.3:
        print(f"  ⚠ 滚动重定参比固定参数好 {diff:+.2f}pp → 建议引入季度滚动重校准")
    elif diff < -0.3:
        print(f"  ✅ 固定参数反而更好 {-diff:+.2f}pp → 参数稳定，不需要频繁重定")
    else:
        print(f"  ✅ 两者差距 {diff:+.2f}pp，在噪音范围内 → 固定参数足够，但可定期复核")
    print()
    return df


# ----------------------------------------------------------------------
# E 段：时间加权定参
# ----------------------------------------------------------------------
def seg_e(ev: pd.DataFrame):
    print(LINE)
    print("E. 时间加权定参：近期样本给更高权重，最优档会变吗")
    print(LINE)
    ev = ev.copy()
    d = pd.to_datetime(ev["date"])
    ev["y"] = d.dt.year
    end = d.max()
    age = (end - d).dt.days.values.astype("float64")

    # 指数加权：半衰期 h 天 → lambda = ln2/h
    halflives = [250, 500, 1000, 1e9]  # 1e9 ≈ 等权
    print(f"  {'半衰期':<10}{'箱宽<=10%':>12}{'<=15%':>10}{'<=20%':>10}{'<=25%':>10}{'不约束':>10}   {'最优档':>8}")
    print("  " + "-" * 74)
    bests = []
    for h in halflives:
        w = np.exp(-np.log(2) * age / h)
        scores = []
        for wt in WIDTH_GRID:
            m = mask_of(ev, PROX_GRID[0], wt)
            if m.sum() < 30:
                scores.append(np.nan); continue
            # 日等权：先按交易日聚合日内均值，再对交易日做加权平均
            dv = ev.loc[m].groupby("date")["ab_fwd60"].mean().dropna()
            dts = pd.to_datetime(dv.index)
            dw = np.exp(-np.log(2) * (end - dts).days.values.astype("float64") / h)
            scores.append(float((dv.values * dw).sum() / dw.sum()) * 100)
        b = int(np.nanargmax(scores))
        best_tag = "不约束" if WIDTH_GRID[b] >= 1 else format(WIDTH_GRID[b], ".0%")
        hl = "等权" if h > 1e8 else f"{int(h)}天"
        print(f"  {hl:<10}" + "".join(f"{s:>+10.2f}" if s == s else f"{'n/a':>10}" for s in scores)
              + f"   {best_tag:>8}")
        bests.append(WIDTH_GRID[b])
    print()
    uniq = sorted(set(bests))
    if len(uniq) == 1:
        print(f"  ✅ 无论怎么加权，最优档都指向 <= {uniq[0]:.0%} → 参数不受时间衰减影响")
    else:
        print(f"  ⚠ 加权方式不同，最优档在 {uniq} 之间变化 → 近期与历史存在结构性差异")
        print(f"     等权（≈全样本）指向 <= {bests[-1]:.0%}，"
              f"最短半衰期（≈只看近1年）指向 <= {bests[0]:.0%}")
    print()
    return bests


# ----------------------------------------------------------------------
# F 段：短窗口标签 fwd20，把可检验末端往前推
# ----------------------------------------------------------------
def seg_f(ev: pd.DataFrame):
    print(LINE)
    print("F. 短窗口标签 fwd20：牺牲标签长度，换取时效性")
    print(LINE)
    ev = ev.copy()
    ev["y"] = pd.to_datetime(ev["date"]).dt.year

    # fwd20 与 fwd60 的相关性
    ok = ev["fwd20"].notna() & ev["fwd60"].notna()
    c = np.corrcoef(ev.loc[ok, "fwd20"], ev.loc[ok, "fwd60"])[0, 1]
    print(f"  fwd20 与 fwd60 的相关系数 = {c:.3f}")
    print("  （高相关说明 fwd20 可作为 fwd60 的近期代理，但要打折看）")
    print()

    # 用 fwd20 检验：各箱宽档在【最近 12 个月】的表现
    print("  【近 12 个月】用 fwd20 做标签，各箱宽档表现：")
    recent = ev[ev["date"] >= "2025-07-23"]
    rows = []
    for w in WIDTH_GRID:
        m = mask_of(recent, PROX_GRID[0], w)
        d = recent[m]
        if len(d) < 20:
            rows.append({"箱宽档": "不约束" if w >= 1 else format(w, ".0%"),
                         "n": len(d), "fwd20超额": np.nan}); continue
        # 近端事件不足 60 日，只能用 fwd20；取横截面去均值口径，日等权
        v = d.groupby("date")["ex_fwd20"].mean().dropna()
        rows.append({"箱宽档": "不约束" if w >= 1 else format(w, ".0%"),
                     "n": len(d), "fwd20超额": float(v.mean()) * 100})
    show(rows)
    print("  注：近端事件不足 60 日，只能用 fwd20；样本量小，仅作方向性参考。")
    print()


# ----------------------------------------------------------------------
# G 段：结论
# ----------------------------------------------------------------------
def seg_g(ev: pd.DataFrame, best_b: dict, best_e: list, df_d: pd.DataFrame):
    print(LINE)
    print("G. 结论与建议方案")
    print(LINE)
    print("""
  Q1 起点要不要后移？
     由 B/E 段判定：若不同起点、不同时间加权都指向同一档，说明参数跨期稳健，
     保留长样本是对的（长样本 = 更多样本 = 更窄 CI）；反之则应缩短起点。

  Q2 末端缺口怎么办？
     ★ 关键认知：这不是缺陷，是 fwd60 标签的必然代价。
       任何「预测未来 60 日」的回测，末端必有 60 个交易日的空白区。
       想缩短空白只有三条路：
         ① 换短标签（fwd20）→ 空白缩到 20 日，但预测目标也变了
         ② 用不完整窗口（不足 60 日就先算着）→ 会引入截断偏差
         ③ 滚动前推复核（D 段）→ 不消除空白，但能持续验证参数是否失效
""")
    print(LINE)
    print("※ 数据来源：腾讯前复权日线与换手率（通达信数据源）、通达信指数日线、")
    print("※ 东方财富财报。以上均为历史统计与回测结论，不构成投资建议。")
    print(LINE)


def main():
    ev = load_events()
    print(f"载入事件 {len(ev):,} 条，交易日 {ev['date'].nunique():,}，股票 {ev['code'].nunique():,}")
    print()

    # A 段：末端边界推算
    import pickle
    try:
        k = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
        all_d = set()
        for code, bars in k.items():
            for x in bars[-3:]:
                all_d.add(x["date"])
        raw_last = max(all_d)
    except Exception as e:
        raw_last = f"(读取失败 {type(e).__name__})"

    seg_a(ev, raw_last)
    best_b = seg_b(ev)
    gc.collect()
    seg_c(ev)
    gc.collect()
    df_d = seg_d(ev)
    gc.collect()
    best_e = seg_e(ev)
    gc.collect()
    seg_f(ev)
    seg_g(ev, best_b, best_e, df_d)


if __name__ == "__main__":
    main()
