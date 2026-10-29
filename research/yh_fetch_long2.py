# -*- coding: utf-8 -*-
"""
长历史日线抓取 v2（腾讯 newfqkline，按时段倒序分页）。

发现：newfqkline 的 param 第 4 位是「结束日期」，第 5 位是条数（上限 640）。
       param=sh600519,day,,2022-12-31,640,qfq  →  返回截止该日的 640 根。
因此可从今日起逐段向前翻页，拼出完整历史，且速度远快于逐票翻页的通达信 MCP。

复权口径：每个窗口的前复权锚点不同（锚在该窗口最后一天），相邻窗口之间只差一个常数倍。
=> 用重叠区求该常数倍并缩放，即可把全部历史统一到「以今日为锚」的前复权口径。
"""
from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import CACHE_DIR, to_secid  # noqa: E402

OUT = os.path.join(CACHE_DIR, "kline_long2.pkl")
ROUNDS = 4                 # 4 段 ≈ 2500 根；再截断到 2018 年中，取干净区间
PER = 640
SINCE = "2018-06-01"       # 更早的窗口已发现少量复权异常，直接截断


def _limit(code: str) -> float:
    """分板块单日涨跌幅上限（留 1 个点余量）"""
    if code.startswith(("83", "87", "43", "92", "920")):
        return 0.31                       # 北交所 ±30%
    if code.startswith(("300", "301", "688", "689")):
        return 0.21                       # 创业板 / 科创板 ±20%
    return 0.11                           # 主板 ±10%


def sanitize(bars: list, code: str) -> list:
    """
    只剔除「明显损坏」的区段。注意不能按涨跌停限幅来卡：
    前复权后的收益包含分红调整，涨跌幅自然会略微超过名义涨跌停板（实测可达 13%）。
    因此这里只保留「负价 / 单日 >50%」这类真实损坏的判定。
    """
    bars = [b for b in bars if b["date"] >= SINCE]
    if len(bars) < 5:
        return []
    c = np.array([b["close"] for b in bars], dtype=float)
    prev = np.maximum(c[:-1], 1e-9)
    r = c[1:] / prev - 1.0
    viol = np.where(np.abs(r) > 0.50)[0] + 1
    viol = np.concatenate([np.where(c <= 0)[0], viol])
    viol = viol[viol >= 10]
    return bars[int(viol.max()) + 1:] if len(viol) else bars

_tl = threading.local()


def _sess():
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session()
        s.trust_env = False
        _tl.s = s
    return s


def fetch_window(code: str, end: str, per: int = PER, retry: int = 3):
    sid = to_secid(code)
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
           f"?param={sid},day,,{end},{per},qfq")
    for a in range(retry):
        try:
            r = _sess().get(url, headers={"Referer": "https://gu.qq.com/"}, timeout=25)
            blk = r.json()["data"][sid]
            rows = blk.get("qfqday") or blk.get("day")
            if not rows:
                return []
            out = []
            for row in rows:
                try:
                    out.append({
                        "date": row[0],
                        "open": float(row[1]),
                        "close": float(row[2]),
                        "high": float(row[3]),
                        "low": float(row[4]),
                        "volume": float(row[5]),
                        "turnover": float(row[7]) if len(row) > 7 and row[7] not in ("", None) else 0.0,
                        "amount": float(row[8]) * 1e4 if len(row) > 8 and row[8] not in ("", None) else 0.0,
                    })
                except (ValueError, IndexError, TypeError):
                    continue
            return out
        except Exception:
            time.sleep(0.6 + a)
    return []


def fetch_one(code: str):
    """从今日起逐段向前，返回统一到「今日为锚」的前复权完整历史"""
    import datetime as _dt

    wins = []
    end = ""
    for _ in range(ROUNDS):
        rows = fetch_window(code, end)
        if not rows:
            break
        wins.append(rows)                       # wins[0] 最新
        y, m, dd = map(int, rows[0]["date"].split("-"))
        # 下一段结束日 = 本段首日之后 30 自然日（约 20 个交易日重叠，用于对齐复权基准）
        nxt = _dt.date(y, m, dd) + _dt.timedelta(days=30)
        if nxt.year > 2013 + 0:
            if nxt > _dt.date.today():
                break
        if len(wins) >= ROUNDS:
            break
        end = nxt.isoformat()

    if not wins:
        return []

    merged: dict = {}                            # date -> bar（基准：最新窗口）
    for w in wins:
        if not merged:
            merged = {r["date"]: r for r in w}
            continue
        ov = [r for r in w if r["date"] in merged]
        if len(ov) < 5:
            continue                             # 无足够重叠则不敢缩放，丢弃该窗口
        ratios = [merged[r["date"]]["close"] / r["close"] for r in ov]
        k = sum(ratios) / len(ratios)
        # 若重载后的比值离散过大说明不是简单的常数倍关系，丢弃该窗口以免污染
        if max(ratios) / min(ratios) > 1.02:
            continue
        for r in w:
            if r["date"] in merged:
                continue
            rr = dict(r)
            rr["open"] *= k; rr["close"] *= k; rr["high"] *= k; rr["low"] *= k
            merged[r["date"]] = rr
    return sanitize([merged[d] for d in sorted(merged)], code)


def main(workers: int = 32):
    with open(os.path.join(CACHE_DIR, "kline_1100.pkl"), "rb") as f:
        codes = sorted(pickle.load(f).keys())
    out: dict = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, "rb") as f:
                out = pickle.load(f)
        except Exception:
            out = {}
    todo = [c for c in codes if c not in out]
    print(f"总数 {len(codes)}  已缓存 {len(out)}  待取 {len(todo)}", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, bars in zip(todo, ex.map(fetch_one, todo)):
            done += 1
            if bars:
                out[code] = bars
            if done % 1000 == 0:
                print(f"  {done}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
                with open(OUT, "wb") as f:
                    pickle.dump(out, f)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    ln = sorted(len(v) for v in out.values())
    print(f"完成 {len(out)} 只，bar数中位数 {ln[len(ln)//2]}，最长 {ln[-1]}，最短 {ln[0]}")
    if out:
        print("覆盖:", min(v[0]["date"] for v in out.values() if v),
              "~", max(v[-1]["date"] for v in out.values() if v))


if __name__ == "__main__":
    main()
