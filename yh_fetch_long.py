# -*- coding: utf-8 -*-
"""
长历史日线抓取（通达信 MCP，前复权）。

为什么要重抓：腾讯 newfqkline 接口单次上限 641 根（约 2024-02 起），
导致原事件表的有效样本几乎全部落在 2025-2026 两年，跨牛熊的统计不可靠。
通达信 MCP 支持 startxh 翻页，可取到 2020 年以前，覆盖 2021 顶部与 2022 整轮熊市。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yh_tdx  # noqa: E402
from yh_regime import _parse_kline_text  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT = os.path.join(CACHE, "kline_long.pkl")
PAGE = 700                       # 服务端每页实际上限
PAGES = 3                        # 0/700/1400 → 约 2100 根，回看至 2020 年初

_tl = threading.local()


def _client():
    c = getattr(_tl, "cli", None)
    if c is None:
        for attempt in range(3):
            try:
                c = yh_tdx.TdxMcp()
                break
            except Exception:
                time.sleep(2 + attempt * 3)
        _tl.cli = c
    return c


def setcode(code: str) -> str:
    if code[0] == "6":
        return "1"
    if code[0] in ("0", "3"):
        return "0"
    if code[:2] in ("43", "83", "87", "92"):
        return "2"
    return "0"


def _call(arguments: dict, retry: int = 3) -> str:
    for a in range(retry):
        try:
            cli = _client()
            r = cli.call("tdx_kline", arguments)
            if isinstance(r, str):
                return r
            if isinstance(r, dict):
                cont = r.get("content")
                if cont:
                    return "\n".join(c.get("text", "") for c in cont if isinstance(c, dict))
                return json.dumps(r, ensure_ascii=False)
            return json.dumps(r, ensure_ascii=False)
        except Exception:
            time.sleep(1.5 + a * 2.5)
            _tl.cli = None
    return ""


def fetch_one(code: str) -> list:
    rows = []
    for p in range(PAGES):
        txt = _call({"code": code, "setcode": setcode(code), "period": "4",
                     "wantNum": 1000, "startxh": p * PAGE, "tqFlag": "1"})
        got = _parse_kline_text(txt)
        if got:
            rows.extend(got)
    if not rows:
        return []
    df = {r["date"]: r for r in rows}
    return [df[d] for d in sorted(df)]


def main(workers: int = 24, limit: int | None = None):
    with open(os.path.join(CACHE, "kline_1100.pkl"), "rb") as f:
        codes = sorted(pickle.load(f).keys())
    if limit:
        codes = codes[:limit]

    out: dict = {}
    if os.path.exists(OUT):
        try:
            with open(OUT, "rb") as f:
                out = pickle.load(f)
        except Exception:
            out = {}
    todo = [c for c in codes if c not in out]
    print(f"总数 {len(codes)}  已缓存 {len(out)}  待取 {len(todo)}", flush=True)

    done, t0, fail = 0, time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for fu in as_completed(futs):
            c = futs[fu]
            try:
                bars = fu.result()
            except Exception:
                bars = []
            done += 1
            if bars:
                out[c] = bars
            else:
                fail += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  失败{fail}  用时{el:.0f}s  预计剩余"
                      f"{el / done * (len(todo) - done) / 60:.1f}min", flush=True)
                with open(OUT, "wb") as f:
                    pickle.dump(out, f)

    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    ln = sorted(len(v) for v in out.values())
    print(f"完成 {len(out)} 只，失败 {fail} 只，bar数中位 {ln[len(ln)//2]}  最长 {ln[-1]}")
    if out:
        first = min(v[0]["date"] for v in out.values() if v)
        last = max(v[-1]["date"] for v in out.values() if v)
        print("覆盖区间:", first, "~", last)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
