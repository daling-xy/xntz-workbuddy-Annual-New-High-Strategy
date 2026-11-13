# -*- coding: utf-8 -*-
"""
成交额门槛 4亿 → 2亿 单门实证（2026-11-12 用户指示改参后补做）
================================================================
背景：2026-11-10 的 yh_relax4.py 只验了 prox / width / turn / ret60 四门，
      成交额门从未单独检验过。本次是【正式参数改动】，按既有纪律补做复核。

方法（完全复用 yh_relax4.py 的口径与函数，防幸存者偏差）：
  ① 边际组逐年检验：在【不施加 amt 门】的样本上，
     A = 2亿 ≤ amt30 < 4亿（本次新放进来的票）
     B = amt30 ≥ 4亿（原门内）
     比 ab_fwd60（β调整60日超额）与大亏率（ab_fwd60 < −20%），逐年一致性。
  ② 整池配对区块 bootstrap：放宽池 − 现池，区块 60 交易日 / 4000 次 / 90%CI，
     全样本 + UP / CHOP / DOWN 分层。

采纳逻辑（与四门实证一致）：边际组【不更差】才支持放宽；
  收益维度 A−B 一致为负、或风险维度 A 大亏率一致更高 → 证据不支持放宽。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yh_relax4 import (BASE_GATES, gate, load_events, report_relax, P, _lines)  # noqa: E402
from yh_blockboot_all import make_dmap  # noqa: E402

OUT = os.path.join(BASE_DIR, "output", "amt_relax_20261112.txt")
OLD, NEW = 4e8, 2e8


def main() -> None:
    t0 = time.time()
    ev = load_events()
    dmap = make_dmap(ev)
    P("=" * 104)
    P("成交额门槛放宽实证 · 2026-11-12 · 4亿 → 2亿")
    P("=" * 104)
    P("事件表 events_fixed.pkl：%s 行 / %s 只 / %s 交易日（%s ~ %s）" % (
        f"{len(ev):,}", ev["code"].nunique(), ev["date"].nunique(),
        ev["date"].min().date(), ev["date"].max().date()))
    P("指标：ab_fwd60 = fwd60 − beta×mkt_fwd60（β调整60日超额）；大亏 = ab_fwd60<−20%")
    P("基准门：prox≥0.95 / 箱宽≤15% / 额≥4亿 / 换手≤6% / ret60≤19%")
    P("检验门：amt30_min 4亿 → 2亿（唯一变量）")

    # ---------- 样本规模 ----------
    d_noamt = gate(ev, drop="amt")
    a = d_noamt[(d_noamt["amt30"] >= NEW) & (d_noamt["amt30"] < OLD)]
    b = d_noamt[d_noamt["amt30"] >= OLD]
    P("\n【样本规模】不施加成交额门后共 %s 条；其中" % f"{len(d_noamt):,}")
    P("  边际组 A（2~4亿，本次新放入）: %s 条，占比 %.1f%%" % (
        f"{len(a):,}", len(a) / len(d_noamt) * 100))
    P("  原门内 B（≥4亿）           : %s 条" % f"{len(b):,}")
    P("  边际组中位成交额 %.2f亿 / 原门内 %.2f亿" % (
        a["amt30"].median() / 1e8, b["amt30"].median() / 1e8))
    P("  边际组 β 中位 %.2f / 原门内 %.2f（越小越低波）" % (
        a["beta"].median(), b["beta"].median()))

    # ---------- 边际组逐年 + 整池配对 ----------
    pool_base = gate(ev, over={"amt": OLD})
    pool_relax = gate(ev, over={"amt": NEW})
    res = report_relax(
        "H_amt：成交额门槛 4亿 → 2亿（边际组 = 2~4亿）",
        d_noamt,
        lambda g: (g["amt30"] >= NEW) & (g["amt30"] < OLD),
        lambda g: (g["amt30"] >= OLD),
        "2~4亿，新放入", "≥4亿，原门内",
        pool_relax, pool_base, dmap)

    # ---------- 补充：四种口径的整池平均 ----------
    P("\n【补充口径】整池均值对照（放宽池 vs 现池，简单均值，非配对）")
    for name, q in [("现池 4亿", pool_base), ("放宽 2亿", pool_relax)]:
        P("  %-10s 事件 %6d  日均 %.2f 只  ab_fwd60 %+.2f%%  fwd60 %+.2f%%  大亏率 %.1f%%  β %.2f" % (
            name, len(q), q.groupby("date").size().mean(),
            q["ab_fwd60"].mean() * 100, q["fwd60"].mean() * 100,
            (q["ab_fwd60"] < -0.20).mean() * 100, q["beta"].median()))

    P("\n【裁决口径】收益方向一致性 %d/%d 年；风险方向一致性（边际组不更危险）%d/%d 年；及格线 6/8" % (
        res["ret"][0], res["ret"][1], res["bad"][0], res["bad"][1]))
    P("耗时 %.1f 秒" % (time.time() - t0))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print("已写出", OUT, flush=True)


if __name__ == "__main__":
    main()
