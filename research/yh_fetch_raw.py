# -*- coding: utf-8 -*-
"""按 TSP regime_builder 的口径，拉取「未复权日线 + 股票名称」。

为什么需要未复权价：
  涨停/跌停判定用的是交易所真实价格（raw_close vs 涨停价），
  前复权价会把历史价格按除权因子缩放，无法判断当天是否封板。

数据来源与现有 yh_fetch_long2 完全一致（腾讯 newfqkline），唯一区别是不带 qfq 参数。

输出：
  cache/kline_raw.pkl   {code: [{date, open, close, high, low, volume, turnover, amount}, ...]}
  cache/names_raw.pkl   {code: name}

用法：
  python -m research.yh_fetch_raw            # 增量：只补缺失的票
  python -m research.yh_fetch_raw --force    # 全量重拉
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yh_datasource import _get, to_secid  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache")
OUT_KLINE = os.path.join(CACHE, "kline_raw.pkl")
OUT_NAMES = os.path.join(CACHE, "names_raw.pkl")

DAYS = 2000  # 约 2018 年至今
WORKERS = 24


def fetch_raw(code: str, days: int = DAYS):
    """返回 (rows, name)。

    ⚠️ 接口选择是踩过坑的：
      - newfqkline/get 不带 qfq  → {"code":1,"msg":"bad params"}   ❌
      - newfqkline/get 带 qfq    → 前复权，且 count 有上限（2000 只给 641 根）
      - kline/kline              → 未复权，一次给满 2001 根          ✅
    本函数用 kline/kline。它只返回 6 个字段，成交额/换手需另行按日期对齐补。
    """
    sid = to_secid(code)
    url = ("https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
           f"?param={sid},day,,,{days}")
    r = _get(url, referer="https://gu.qq.com/")
    if r is None:
        return None, None
    try:
        blk = r.json()["data"][sid]
        rows = blk.get("day") or blk.get("qfqday")
        if not rows:
            return None, None
        name = None
        qt = blk.get("qt") or {}
        arr = qt.get(sid) if isinstance(qt, dict) else None
        if isinstance(arr, list) and len(arr) > 1:
            name = arr[1]
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
                })
            except (ValueError, IndexError, TypeError):
                continue
        return (out or None), name
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = os.path.join(CACHE, "kline_320.pkl")
    with open(src, "rb") as f:
        codes = sorted(pickle.load(f).keys())
    print(f"标的数: {len(codes)}", flush=True)

    done_k, done_n = {}, {}
    if not args.force and os.path.exists(OUT_KLINE):
        with open(OUT_KLINE, "rb") as f:
            done_k = pickle.load(f)
        if os.path.exists(OUT_NAMES):
            with open(OUT_NAMES, "rb") as f:
                done_n = pickle.load(f)
        print(f"已有缓存: {len(done_k)} 只", flush=True)

    todo = [c for c in codes if c not in done_k]
    print(f"待拉取: {len(todo)} 只", flush=True)
    if not todo:
        print("无需拉取", flush=True)
        return

    # 预检：先试 3 只，全失败直接退出（上一次 5575 只全失败就是没做这一步）
    print("预检 3 只 ...", flush=True)
    probe_ok = 0
    for c in codes[:3]:
        rows, name = fetch_raw(c)
        if rows:
            probe_ok += 1
            print(f"  {c}: {len(rows)} 根  最新 {rows[-1]['date']}  收盘 {rows[-1]['close']}  名称 {name}", flush=True)
        else:
            print(f"  {c}: 失败", flush=True)
    if probe_ok == 0:
        raise SystemExit("预检失败：3 只全部拉不到，接口或网络有问题，已中止")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_raw, c): c for c in todo}
        for fu in as_completed(futs):
            code = futs[fu]
            try:
                rows, name = fu.result()
            except Exception:
                rows, name = None, None
            if rows:
                done_k[code] = rows
                if name:
                    done_n[code] = name
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 500 == 0:
                print(f"  进度 {ok + fail}/{len(todo)}  成功 {ok} 失败 {fail}", flush=True)
                with open(OUT_KLINE, "wb") as f:
                    pickle.dump(done_k, f)

    with open(OUT_KLINE, "wb") as f:
        pickle.dump(done_k, f)
    with open(OUT_NAMES, "wb") as f:
        pickle.dump(done_n, f)

    print(f"\n完成: 成功 {ok} 失败 {fail} 累计 {len(done_k)} 只", flush=True)
    print(f"输出: {OUT_KLINE}", flush=True)
    print(f"输出: {OUT_NAMES}", flush=True)

    # 抽样校验：未复权 vs 前复权 应该不同（除权过），完全相同说明拉错了
    if os.path.exists(os.path.join(CACHE, "kline_320.pkl")):
        with open(os.path.join(CACHE, "kline_320.pkl"), "rb") as f:
            adj = pickle.load(f)
        sample = [c for c in list(done_k)[:50] if c in adj]
        same = diff = 0
        for c in sample:
            a = {r["date"]: r["close"] for r in adj[c]}
            r = {x["date"]: x["close"] for x in done_k[c]}
            common = set(a) & set(r)
            if not common:
                continue
            if any(abs(a[d] - r[d]) > 1e-6 for d in common):
                diff += 1
            else:
                same += 1
        print(f"\n抽样校验（{len(sample)} 只）：与前复权不同 {diff} 只，相同 {same} 只", flush=True)
        print("  说明：完全相同不代表错（该股期间没除权）；但若全部相同，则很可能没拉成未复权。",
              flush=True)


if __name__ == "__main__":
    main()
