# -*- coding: utf-8 -*-
"""
区块 bootstrap：把 fwd60 的【持有期重叠】考虑进去，重算关键结论

问题：此前所有 CI 都用「按交易日抽样」，隐含假设每天独立。
      但 fwd60 = 未来 60 个交易日收益，相邻两事件日的窗口重叠约 98%。
      → 独立性被高估 → CI 偏窄 → 显著性失真。

修正：区块 bootstrap。把时间轴切成互不重叠的 60 交易日区块，
      以【区块】为抽样单位（区块内部整体保留，模拟真正的独立观测）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_rev_fine import load_events, load_fin, attach_fin, base_v2  # noqa: E402

RNG = np.random.default_rng(20261027)
HB = 60      # 区块长度 = 持有期（交易日）
NB = 4000


def add_block(ev_all: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """给事件打上全局区块号（按 60 交易日切一刀）"""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(ev_all["date"]).unique()))
    dmap = {dt: i for i, dt in enumerate(days)}
    d = d.copy()
    d["_bi"] = pd.to_datetime(d["date"]).map(lambda x: dmap.get(x, np.nan))
    d = d.dropna(subset=["_bi"])
    d["_bi"] = (d["_bi"] // HB).astype(int)
    return d


def daily_series(d: pd.DataFrame, col: str = "ab_fwd60") -> pd.Series:
    return d.groupby("date")[col].mean()


def block_boot(d: pd.DataFrame, col: str = "ab_fwd60", n_boot: int = NB, ci: float = 0.90):
    """区块 bootstrap：以区块为单位抽样，返回 (点估计, lo, hi)"""
    # 先按日截面平均 → 每天一个值；再把天归入区块
    dm = daily_series(d, col)
    bi = d[["date", "_bi"]].drop_duplicates().set_index("date")["_bi"]
    dm = dm.to_frame("v").join(bi)
    dm["v"] = dm["v"].fillna(0.0)          # 该日无入选则贡献 0
    blocks = dm.groupby("_bi")["v"].mean()  # 区块内先取均值
    if len(blocks) < 3:
        return (np.nan, np.nan, np.nan)
    arr = blocks.values
    idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
    bs = arr[idx].mean(axis=1)
    pt = float(arr.mean())
    return (pt, float(np.percentile(bs, (1 - ci) / 2 * 100)),
            float(np.percentile(bs, (1 + ci) / 2 * 100)))


def paired_block(a: pd.DataFrame, b: pd.DataFrame, col: str = "ab_fwd60",
                 n_boot: int = NB, ci: float = 0.90):
    da = daily_series(a, col)
    db = daily_series(b, col)
    common = da.index.intersection(db.index)
    if len(common) < 10:
        return (np.nan, np.nan, np.nan, 0)
    bmap = a[["date", "_bi"]].drop_duplicates().set_index("date")["_bi"]
    diff = (da.loc[common] - db.loc[common]).to_frame("v").join(bmap)
    blocks = diff.groupby("_bi")["v"].mean()
    if len(blocks) < 3:
        return (np.nan, np.nan, np.nan, len(common))
    arr = blocks.values
    idx = RNG.integers(0, len(arr), size=(n_boot, len(arr)))
    bs = arr[idx].mean(axis=1)
    return (float(arr.mean()), float(np.percentile(bs, (1 - ci) / 2 * 100)),
            float(np.percentile(bs, (1 + ci) / 2 * 100)), len(common))


def verdict(pt, lo, hi):
    if np.isnan(pt):
        return "样本不足"
    if lo > 0:
        return "显著为正"
    if hi < 0:
        return "显著为负"
    return "含零·无效"


def bar(t):
    print("=" * 116)
    print(t)
    print("=" * 116)


def main():
    ev = load_events()
    fin = load_fin()
    e = attach_fin(ev, fin)
    print("回测交易日 %d；区块长度 %d → 全样本最多约 %d 个独立区块"
          % (ev["date"].nunique(), HB, ev["date"].nunique() // HB))
    print()

    bar("区块 bootstrap vs 原「按日 bootstrap」：关键结论对照")
    print("  (区块长度=60 交易日，与 fwd60 持有期一致)")
    print()
    print("  %-30s %10s %10s %18s %18s" % ("配置 / 比较", "按日CI", "→", "区块CI", "判定"))
    print("  " + "-" * 92)

    for rg in ["DOWN", "UP", "CHOP"]:
        se = e[(e["regime"] == rg) & e["YSTZ"].notna()]
        base = add_block(ev, base_v2(se))
        print("  【%s】" % rg)
        pt0, lo0, hi0 = block_boot(base)
        print("     %-26s %18s   [%+6.2f,%+6.2f]  %s"
              % ("基准池(v2四纪律)", "%+.2f%%" % (pt0 * 100), lo0 * 100, hi0 * 100, verdict(pt0, lo0, hi0)))
        for thr in [10, 20, 30, 50]:
            d = add_block(ev, base_v2(se).pipe(lambda x: x[x["YSTZ"] >= thr]))
            if len(d) < 15:
                continue
            pt, lo, hi = block_boot(d)
            df, dlo, dhi, nday = paired_block(d, base)
            print("     %-26s %18s   [%+6.2f,%+6.2f]  %s   │ 相对基准 %+6.2fpp [%+6.2f,%+6.2f] %s"
                  % ("+营收>=%d%%" % thr, "%+.2f%%" % (pt * 100), lo * 100, hi * 100,
                     verdict(pt, lo, hi), df * 100, dlo * 100, dhi * 100, verdict(df, dlo, dhi)))
        print()

    bar("判读")
    print("  区块 bootstrap 把抽样单位放大到 60 交易日，CI 会明显变宽。")
    print("  哪些结论在新口径下仍然不含零，才值得采信；放宽的都应降级为『方向性线索』。")


if __name__ == "__main__":
    main()
