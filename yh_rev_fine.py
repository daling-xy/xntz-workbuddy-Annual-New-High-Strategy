# -*- coding: utf-8 -*-
"""
【营收同比门槛解剖】—— 回答「rev_yoy >= 30% 是怎么来的、它在当前参数组合里最优吗」

★ 关键痛点：项目里所有历史校准脚本（yh_calibrate / yh_final_calib* / yh_regime_calib*）
       都【从未】用过 rev_yoy —— 因为 events 事件表本身不含财务字段。
       30% 是作者先验（README 3.4「作者定下的不可动摇的底线」），没被搜索验证过。
       本脚本第一次把它放进网格。

★ 无前视偏差处理：财报按【UPDATE_DATE 披露日】merge_asof 到事件日，
       即每个事件只能用「当天已披露」的最新报告期营收同比。

本脚本做 5 段：
  A 拉全历史财报 → 构造 asof 表 → 无前视 join 到事件表
  B 单因子单调性：rev_yoy 分档 vs beta 调整超额（检验是否真单调）
  C 细扫 rev_yoy 门槛（其余参数固定为 v2：prox>=0.95 + 箱宽<=15% + 额>=4亿 + 换手<=6%）
  D 分周期（DOWN 是当前环境）+ 前后半样本
  E 配对 bootstrap：30% 相对相邻档（20/25/40/50）是否显著更优

数据来源：腾讯前复权日线、通达信指数日线、东方财富业绩报表(RPT_LICO_FN_CPD)。
所有结论均为历史统计，不构成投资建议。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import _session, UA

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BOOT = 2000
_rng = np.random.default_rng(20261027)
LINE = "=" * 122
SUB = "-" * 122

FIN_CACHE = os.path.join(CACHE, "fin_history.pkl")
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
COLS = "SECURITY_CODE,REPORTDATE,UPDATE_DATE,TOTAL_OPERATE_INCOME,YSTZ,SJLTZ"


# ---------------------------------------------------------------- 财报数据
def _report_dates() -> list:
    ds = []
    for y in range(2017, 2027):
        for md in ("03-31", "06-30", "09-30", "12-31"):
            ds.append(f"{y}-{md}")
    return ds


def _fetch_one(rd: str) -> list:
    """拉单个报告期的全部 A 股业绩报表。注意：不能带 SECURITY_TYPE_CODE 过滤，
       实测带上后历史报告期返回 0 条（该字段只对最新期有效）。"""
    rows, page = [], 1
    while page <= 30:
        params = {
            "reportName": "RPT_LICO_FN_CPD",
            "columns": COLS,
            "filter": f"(REPORTDATE='{rd}')",
            "pageSize": 500, "pageNumber": page,
            "source": "WEB", "client": "WEB",
            "sortColumns": "SECURITY_CODE", "sortTypes": "1",
        }
        try:
            r = _session().get(URL, params=params,
                               headers={**UA, "Referer": "https://data.eastmoney.com/"},
                               timeout=30)
            res = (r.json() or {}).get("result") or {}
        except Exception:
            break
        d = res.get("data") or []
        rows.extend(d)
        pages = res.get("pages") or 0
        if page >= pages or not d:
            break
        page += 1
    return rows


def load_fin(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(FIN_CACHE) and not refresh:
        return pd.read_pickle(FIN_CACHE)

    rds = _report_dates()
    print(f"  拉取 {len(rds)} 个报告期的业绩报表（并发 12）...", flush=True)
    t0 = time.time()
    all_rows: list = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, rows in enumerate(ex.map(_fetch_one, rds), 1):
            all_rows.extend(rows)
            if i % 5 == 0 or i == len(rds):
                print(f"    {i}/{len(rds)} 报告期  累计 {len(all_rows):,} 行  "
                      f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("财报拉取为空")

    # 只保留 A 股（沪60/68、深00/30）
    df["SECURITY_CODE"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    ok = df["SECURITY_CODE"].str[:2].isin(["60", "68", "00", "30"])
    df = df[ok].copy()

    df["REPORTDATE"] = pd.to_datetime(df["REPORTDATE"]).dt.strftime("%Y-%m-%d")
    df["UPDATE_DATE"] = pd.to_datetime(df["UPDATE_DATE"]).dt.strftime("%Y-%m-%d")
    for c in ("TOTAL_OPERATE_INCOME", "YSTZ", "SJLTZ"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["UPDATE_DATE", "YSTZ"])
    # 同一批延期 Subsequently 会有重复 UPDATE，取最后.Update 的最大者
    df = df.sort_values(["SECURITY_CODE", "REPORTDATE", "UPDATE_DATE"])
    df = df.drop_duplicates(["SECURITY_CODE", "REPORTDATE"], keep="last")
    df.to_pickle(FIN_CACHE)
    print(f"  完成：{len(df):,} 行 / {df['SECURITY_CODE'].nunique()} 股 / "
          f"{df['REPORTDATE'].nunique()} 报告期，耗时 {time.time()-t0:.0f}s", flush=True)
    return df


def _avail_date(rd_series: pd.Series) -> pd.Series:
    """报告期 -> 公众可得日，取【法定披露截止日】（保守，杜绝前视偏差）。
       A股法定时限：一季报 4/30、半年报 8/31、三季报 10/31、年报 次年 4/30。

    ⚠️ 为什么不能用东方财富的 UPDATE_DATE：
       实测 688018 的 2025-09-30 报告期 UPDATE_DATE = 2026-10-24 —— 它是
       「数据最后修今日」，不是首次披露日。用它做 asof join 会把当时的事件
       错配到很久以后的财报（实测 2026 年事件有 43,857 条错配到 2024-09-30）。
    """
    rd = pd.to_datetime(rd_series)
    out = []
    for d in rd:
        if d.month == 3:
            out.append(pd.Timestamp(d.year, 4, 30))
        elif d.month == 6:
            out.append(pd.Timestamp(d.year, 8, 31))
        elif d.month == 9:
            out.append(pd.Timestamp(d.year, 10, 31))
        else:
            out.append(pd.Timestamp(d.year + 1, 4, 30))
    return pd.Series(out, index=rd_series.index)


def attach_fin(ev: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """无前视偏差：按 SECURITY_CODE 分组，用【法定披露截止日】做 merge_asof。"""
    f = fin[["SECURITY_CODE", "REPORTDATE", "YSTZ",
             "TOTAL_OPERATE_INCOME"]].copy()
    f["date"] = _avail_date(f["REPORTDATE"])
    f = f.rename(columns={"SECURITY_CODE": "code"})
    # ⚠️ 可得日会撞车：年报(12-31)截止次年4/30 与 次年一季报(03-31)截止当年4/30 是同一天。
    #    撞车时应取【报告期更新】者（信息更新），此处先去重再 join。
    f = (f.sort_values(["code", "date", "REPORTDATE"])
           .drop_duplicates(["code", "date"], keep="last")
           .sort_values("date").reset_index(drop=True))

    e = ev.copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values("date").reset_index(drop=True)
    # 只有 date >= 该股首次披露日 才能取到值，天然的 merge_asof 语义
    out = pd.merge_asof(
        e.reset_index(drop=True), f, on="date", by="code",
        direction="backward", allow_exact_matches=True,
    )
    out = out.rename(columns={"REPORTDATE": "fin_report",
                              "TOTAL_OPERATE_INCOME": "fin_revenue"})
    # 披露与事件日间隔
    out["fin_lag_days"] = (out["date"] - pd.to_datetime(out["fin_report"])).dt.days
    return out


# ---------------------------------------------------------------- 统计工具
def boot_ci(sub: pd.DataFrame, key: str = "ab_fwd60", n: int = BOOT):
    v = sub.groupby("date")[key].mean().dropna().values
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    est = float(v.mean())
    idx = _rng.integers(0, len(v), size=(n, len(v)))
    bs = v[idx].mean(axis=1)
    return est, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def stat_row(tag: str, d: pd.DataFrame, w: int = 26):
    n = len(d)
    if n < 30:
        print(f"  {tag:<{w}} n={n:>6}  样本不足")
        return None
    est, lo, hi = boot_ci(d)
    win = float((d["ab_fwd60"] > 0).mean())
    big = float((d["fwd60"] < -0.20).mean())
    per = float(d.groupby("date").size().mean())
    sig = "显著为正" if (lo > 0) else ("显著为负" if hi < 0 else "含零·无效")
    print(f"  {tag:<{w}} n={n:>6}  日均{per:>5.2f}只  β中位{d['beta'].median():>5.2f}  "
          f"{est*100:+6.2f}%  CI[{lo*100:+5.2f},{hi*100:+5.2f}]  "
          f"胜率{win*100:4.1f}%  大亏{big*100:4.1f}%  {sig}")
    return est


def section(t: str) -> None:
    print()
    print(LINE)
    print(t)
    print(LINE)


def load_events() -> pd.DataFrame:
    p = os.path.join(CACHE, "events_fixed.pkl")
    if not os.path.exists(p):
        p = os.path.join(CACHE, "events_long_beta.pkl")
    ev = pd.read_pickle(p)
    lo, hi = ev["beta"].quantile([0.005, 0.995])
    ev["beta"] = ev["beta"].clip(lo, hi)
    ev["ab_fwd60"] = ev["fwd60"] - ev["beta"] * ev["mkt_fwd60"]
    ev["prox_now"] = 1.0 + ev["dd_from_high"]
    keep = [c for c in ["code", "date", "regime", "prox_now", "width30", "amt30",
                        "turn30", "beta", "fwd60", "ab_fwd60"] if c in ev.columns]
    ev = ev[keep]
    return ev[ev["beta"].notna() & ev["fwd60"].notna()].reset_index(drop=True)


def base_v2(d: pd.DataFrame) -> pd.DataFrame:
    """v2 的其余四项纪律，不含营收"""
    return d[(d["prox_now"] >= 0.95) & (d["width30"] <= 0.15) &
             (d["amt30"] >= 4e8) & (d["turn30"] <= 6.0)]


def main() -> None:
    ev = load_events()
    print(f"事件表: {len(ev):,} 行 / {ev['code'].nunique()} 股 / "
          f"{ev['date'].nunique()} 交易日")
    print(f"日期区间: {ev['date'].min()} ~ {ev['date'].max()}")

    if "--refresh" in sys.argv:
        load_fin(refresh=True)
    fin = load_fin()
    print(f"\n财报表: {len(fin):,} 行 / {fin['SECURITY_CODE'].nunique()} 股 / "
          f"{fin['REPORTDATE'].nunique()} 报告期")

    ev2 = attach_fin(ev, fin)
    hit = ev2["YSTZ"].notna().mean()
    print(f"无前视 join 完成：YSTZ 覆盖率 {hit*100:.1f}%  "
          f"（未覆盖多为当时尚无披露或事件日早于首份财报）")
    lag = ev2["fin_lag_days"].dropna()
    if len(lag):
        print(f"财报滞留天数：中位 {lag.median():.0f} 日  "
              f"p25={lag.quantile(.25):.0f}  p75={lag.quantile(.75):.0f}")

    W = ev2[ev2["YSTZ"].notna()]
    b = base_v2(W)
    print(f"\nv2 基准池 + 有营收数据: {len(b):,} 事件")

    # ---------------- B ----------------
    section("B. 单因子单调性：rev_yoy 分档 vs beta 调整超额")
    print("  口径：v2 四纪律已生效（prox>=0.95 + 箱宽<=15% + 额>=4亿 + 换手<=6%）")
    print()
    bins = [(-np.inf, 0), (0, 10), (10, 20), (20, 30), (30, 50),
            (50, 100), (100, np.inf)]
    labels = ["rev_yoy <= 0%", "0 ~ 10%", "10 ~ 20%", "20 ~ 30%",
              "30 ~ 50%", "50 ~ 100%", "> 100%"]
    for (lo_, hi_), lb in zip(bins, labels):
        d = b[(b["YSTZ"] > lo_) & (b["YSTZ"] <= hi_)]
        stat_row(lb, d)

    # ---------------- C ----------------
    section("C. rev_yoy 门槛细扫（其余四项固定为 v2）")
    grid = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
    res = {}
    for x in grid:
        res[x] = stat_row(f"rev_yoy >= {x}%", b[b["YSTZ"] >= x], w=18)
    print()
    vals = {k: v for k, v in res.items() if v is not None}
    ranked = sorted(vals.items(), key=lambda kv: -kv[1])
    print("  → 超额最高的 4 档:",
          ", ".join([f"{k}%({v*100:+.2f}%)" for k, v in ranked[:4]]))
    if 30 in [k for k, _ in ranked]:
        print(f"  → 30% 的排名: {[k for k, _ in ranked].index(30) + 1} / {len(ranked)}")

    # ---------------- D ----------------
    for rg in ["DOWN", "UP", "CHOP"]:
        section(f"D. 分周期重扫 —— [{rg}]")
        br = base_v2(W[W["regime"] == rg])
        print(f"  {rg} 基准池: {len(br):,} 事件")
        print()
        for x in [0, 10, 20, 25, 30, 40, 50, 80]:
            stat_row(f"rev_yoy >= {x}%", br[br["YSTZ"] >= x], w=18)

    section("D2. 前后半样本对照（分界 2022-12-27）")
    ds = sorted(ev2["date"].unique())
    mid = ds[len(ds) // 2]
    print(f"  分界日: {mid}")
    print()
    for tag, part in [("前半", W[W["date"] < mid]), ("后半", W[W["date"] >= mid])]:
        bp = base_v2(part)
        print(f"  【{tag}】基准池 {len(bp):,} 事件")
        for x in [0, 20, 30, 50]:
            stat_row(f"    >={x}%", bp[bp["YSTZ"] >= x], w=18)
        print()

    # ---------------- E ----------------
    section("E. 配对 bootstrap：30% 相对其它门槛")
    print("  注意：各档【样本不同】，必须配对（同一交易日）比较")
    print()

    def daily(d: pd.DataFrame) -> pd.Series:
        return d.groupby("date")["ab_fwd60"].mean().dropna()

    s30 = daily(b[b["YSTZ"] >= 30])
    for rival in [0, 10, 20, 25, 40, 50, 80]:
        sr = daily(b[b["YSTZ"] >= rival])
        common = s30.index.intersection(sr.index)
        if len(common) < 20:
            print(f"  30% vs >= {rival:>3}%: 共同交易日不足（{len(common)}）")
            continue
        diff = (s30.loc[common] - sr.loc[common]).values
        est = float(diff.mean())
        idx = _rng.integers(0, len(diff), size=(BOOT, len(diff)))
        bs = diff[idx].mean(axis=1)
        lo, hi = float(np.percentile(bs, 5)), float(np.percentile(bs, 95))
        verdict = ("30% 显著更优" if lo > 0 else
                   ("30% 显著更差" if hi < 0 else "无显著差异（打平）"))
        print(f"  30% vs >= {rival:>3}%: 差 {est*100:+6.2f}pp  "
              f"CI[{lo*100:+5.2f},{hi*100:+5.2f}]  → {verdict}  ({len(common)} 日)")

    print()
    print(LINE)
    print("若 30% 与其它档全部打平 → 说明营收增速【在已有 v2 四纪律之后】不再提供边际信息；")
    print("若 30% 显著更优 → 它是有效增强项；若显著更差 → 应下调或移除。")
    print(LINE)


if __name__ == "__main__":
    main()
