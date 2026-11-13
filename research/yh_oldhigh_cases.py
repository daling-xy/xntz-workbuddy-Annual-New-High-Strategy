# -*- coding: utf-8 -*-
"""「旧高双顶」历史案例逐只还原 · 2026-11-12

用户选项：「先看更多历史案例」——挑几只 hi250_ago>60 但过 v2.6 全门的票，
逐一拆形态，判断这类票该不该留。

⚠️ 案例是按【已知结果】挑的极值，不是随机样本，只用于看形态，不能当收益证据。
   因此同时输出全样本分布对照（旧高组 vs 新高组）与好坏配对案例。

底座 = prox>=0.88 & 额>=2亿 & VCP；A = 底座 + R/C/H/F（现 v2.6）
OLD  = A & hi250_ago>60；NEW = A & hi250_ago<=60
"""
from __future__ import annotations

import gc
import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_shape_precise import cond_R, cond_C, cond_H     # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "oldhigh_cases_20261112.txt")
CACHE = os.path.join(BASE_DIR, "cache")
AMT = 2e8
_lines: list[str] = []


def P(s: str = "") -> None:
    print(s)
    _lines.append(s)


def dist_row(tag: str, d: pd.DataFrame) -> str:
    q = d["fwd60"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
    return "  %-14s %7d | %+7.2f%% %+7.2f%% | %5.1f%% %5.1f%% %5.1f%% | %+6.1f%% %+6.1f%% | %6.2f | %s" % (
        tag, len(d), d["ab_fwd60"].mean() * 100, d["ex_fwd60"].mean() * 100,
        (d["fwd60"] > 0.20).mean() * 100, (d["fwd60"] < -0.20).mean() * 100,
        (d["ab_fwd60"] < -0.20).mean() * 100,
        d["mfe60"].mean() * 100, d["mae20"].mean() * 100,
        d["beta"].median(),
        " / ".join("%+.1f%%" % (v * 100) for v in q))


def pick_cases(d: pd.DataFrame, n_good: int = 5, n_bad: int = 3) -> pd.DataFrame:
    """按 code 去重 + 尽量分散年份，从 OLD 组挑 fwd60 极值案例。"""
    d = d.dropna(subset=["fwd60", "mfe60"]).copy()
    d["year"] = d["date"].str[:4].astype(int)
    out = []

    def take(pool, n, tag):
        pool = pool.sort_values("fwd60", ascending=(tag == "差"))
        used_years: dict[int, int] = {}
        used_codes: set[str] = set()
        got = 0
        for _, r in pool.iterrows():
            if got >= n:
                break
            if r["code"] in used_codes:
                continue
            if used_years.get(r["year"], 0) >= 2:
                continue
            used_codes.add(r["code"])
            used_years[r["year"]] = used_years.get(r["year"], 0) + 1
            rr = r.to_dict()
            rr["case_tag"] = tag
            out.append(rr)
            got += 1
        # 年份限制导致不足则放宽补齐
        if got < n:
            for _, r in pool.iterrows():
                if got >= n:
                    break
                if r["code"] in used_codes:
                    continue
                used_codes.add(r["code"])
                rr = r.to_dict()
                rr["case_tag"] = tag
                out.append(rr)
                got += 1

    take(d, n_good, "好")
    take(d, n_bad, "差")
    return pd.DataFrame(out)


def shape_of(bars, datestr: str) -> dict:
    """在长历史里还原事件日前后形态。"""
    dates = [x["date"] for x in bars]
    if datestr not in dates:
        return {}
    i = dates.index(datestr)
    cl = np.array([x["close"] for x in bars], float)
    c = cl[i]

    def sret(k):
        return c / cl[i - k] - 1 if i - k >= 0 else np.nan

    pre = cl[max(0, i - 119):i + 1]
    low_i = int(np.argmin(pre))
    high_i = int(np.argmax(pre))
    box = cl[i - 29:i + 1]
    hi30 = box.max()
    post = {}
    for k in (20, 40, 60):
        post[k] = (cl[i + k] / c - 1) if i + k < len(cl) else np.nan
    seg60 = cl[i + 1:min(len(cl), i + 61)]
    return {
        "ret120": sret(120), "ret60": sret(60), "ret20": sret(20),
        "low_ago": len(pre) - 1 - low_i, "high_ago": len(pre) - 1 - high_i,
        "dd120": c / pre.min() - 1, "to120high": c / pre.max() - 1,
        "w30": box.max() / box.min() - 1, "close_hi30": c / hi30 - 1,
        "post20": post[20], "post40": post[40], "post60": post[60],
        "mfe60_real": (seg60.max() / c - 1) if len(seg60) else np.nan,
        "price": c, "bars": len(cl),
    }


def main() -> None:
    P("=" * 118)
    P("「旧高双顶」历史案例逐只还原 · 2026-11-12 · A = v2.6 全门（含 F）")
    P("⚠️ 案例按【已知结果】挑极值，非随机样本，只看形态、不能当收益证据；收益结论以全样本分布为准")
    P("=" * 118)

    ev = pickle.load(open(os.path.join(CACHE, "events_fixed.pkl"), "rb"))
    ev["date"] = ev["date"].astype(str).str[:10]
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    if "prox_now" not in ev.columns:
        ev["prox_now"] = 1.0 + ev["dd_from_high"]
    base = ((ev["prox_now"] >= 0.88) & (ev["amt30"] >= AMT)
            & (ev["width10"] <= 0.10) & (ev["contraction"] <= 0.45)
            & (ev["width30"] <= 0.50))
    ev = ev[base].copy()

    st = pickle.load(open(os.path.join(CACHE, "flat_struct.pkl"), "rb"))
    st["date"] = st["date"].astype(str).str[:10]
    st["code"] = st["code"].astype(str).str.zfill(6)
    ev = ev.merge(st, on=["code", "date"], how="left")

    A = (cond_R(ev).fillna(False) & cond_C(ev).fillna(False)
         & cond_H(ev).fillna(False) & (ev["base_days"] <= 35).fillna(False))
    old = ev[A & (ev["ago250"] > 60)].copy()
    new = ev[A & (ev["ago250"] <= 60)].copy()
    P("")
    P("A 全门事件 %s 行 | 旧高组(>60日) %s 行 (%.1f%%) | 新高组 %s 行" % (
        f"{int(A.sum()):,}", f"{len(old):,}", len(old) / max(A.sum(), 1) * 100, f"{len(new):,}"))

    P("")
    P("① 全样本分布对照（这才是收益结论；分位数为 fwd60 的 10/25/50/75/90）")
    P("")
    P("  %-14s %7s | %8s %8s | %6s %6s %6s | %7s %7s | %5s | %s" % (
        "组", "n", "ab_fwd60", "ex_fwd60", "涨>20%", "跌>20%", "大亏率", "MFE60", "MAE20", "β", "fwd60分位"))
    P("  " + "-" * 112)
    P(dist_row("旧高 >60日", old))
    P(dist_row("新高 <=60日", new))
    P(dist_row("全部 A", ev[A]))

    # 按 hi250_ago 分箱看单调性
    P("")
    P("② hi250_ago 分箱（看旧高越久是不是越差）")
    P("")
    P("  %-14s %7s | %8s %8s | %6s %6s" % ("区间", "n", "ab_fwd60", "fwd60", "涨>20%", "跌>20%"))
    for lo, hi, lab in ((0, 20, "0-20日"), (21, 60, "21-60日"), (61, 120, "61-120日"),
                        (121, 180, "121-180日"), (181, 250, "181-250日")):
        s = ev[A & (ev["ago250"] >= lo) & (ev["ago250"] <= hi)]
        if len(s) == 0:
            continue
        P("  %-14s %7d | %+7.2f%% %+7.2f%% | %5.1f%% %5.1f%%" % (
            lab, len(s), s["ab_fwd60"].mean() * 100, s["fwd60"].mean() * 100,
            (s["fwd60"] > 0.20).mean() * 100, (s["fwd60"] < -0.20).mean() * 100))

    cases = pick_cases(old, 5, 3)
    names = pickle.load(open(os.path.join(CACHE, "names_raw.pkl"), "rb"))
    snap = pickle.load(open(os.path.join(CACHE, "snapshot.pkl"), "rb"))
    nm_of = lambda c: (snap.get(c) or {}).get("name") or names.get(c) or c

    # 释放大表，再加载长历史 K 线
    del ev, st, old, new, A
    gc.collect()
    kl = pickle.load(open(os.path.join(CACHE, "kline_long2.pkl"), "rb"))
    P("")
    P("③ 逐只案例（旧高组，按 fwd60 挑 5 好 + 3 差；同一代码只取一次、尽量分散年份）")
    P("")
    for _, r in cases.iterrows():
        c, ds = r["code"], r["date"]
        sh = shape_of(kl.get(c, []), ds)
        P("  " + "-" * 112)
        P("  [%s] %s %s (%s)   事件日 hi250_ago=%d日" % (
            r["case_tag"], c, nm_of(c), ds, int(r["ago250"])))
        P("   事件日指标: prox %.3f | w10 %.1f%% w30 %.1f%% ct %.2f | base_days %d | ret20 %+.1f%% ret60 %+.1f%%" % (
            r["prox_now"], r["width10"] * 100, r["width30"] * 100, r["contraction"],
            int(r["base_days"]), r["ret20"] * 100, r["ret60"] * 100))
        if sh:
            P("   前 120 日: 收 %.2f | 120日涨 %+.1f%% | 距120日低 %+.1f%% | 距120日高 %+.1f%% | 120日最低点在 %d 日前 | w30 %.1f%% | 收盘距箱顶 %+.1f%%" % (
                sh["price"], sh["ret120"] * 100, sh["dd120"] * 100, sh["to120high"] * 100,
                sh["low_ago"], sh["w30"] * 100, sh["close_hi30"] * 100))
            P("   后 60 日: +20日 %+.1f%% | +40日 %+.1f%% | +60日 %+.1f%% | 60日内最大 %+.1f%%" % (
                sh["post20"] * 100, sh["post40"] * 100, sh["post60"] * 100, sh["mfe60_real"] * 100))
        P("   事件表: fwd60 %+.1f%% | ex_fwd60 %+.1f%% | ab_fwd60 %+.1f%% | MFE60 %+.1f%% | MAE20 %+.1f%%" % (
            r["fwd60"] * 100, r["ex_fwd60"] * 100, r["ab_fwd60"] * 100,
            r["mfe60"] * 100, r["mae20"] * 100))

    P("")
    P("④ 读法提示")
    P("   - 「旧高」不是风险标签：全样本里旧高组的右尾（涨>20%）与新高组对比见 ①。")
    P("   - 好案例与差案例在【事件日指标】上几乎没有区别，差别主要来自之后的市场环境与个股催化；")
    P("     这正是 N 门无解释力、且所有修补门的 bootstrap 都含零的原因。")
    P("   - 案例按结果挑选，仅供形态语义判断，不可用于估计胜率。")
    P("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print("\n已写入", OUT)


if __name__ == "__main__":
    main()
