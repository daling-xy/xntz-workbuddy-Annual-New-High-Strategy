# -*- coding: utf-8 -*-
"""把 cache/ 里的数据打包成 GitHub Release 附件。

为什么要单独打包而不是放进 git：
  1. GitHub 单文件上限 100 MB，kline_long2.pkl 有 840+ MB，push 会被拒
  2. git 对二进制是"完整快照"存储，不是增量，每天重生成的 pkl 会让仓库线性膨胀
  3. 数据是可从公开源重建的缓存，不适合当代码管理

用法:
    python tools/pack_data.py                  # 核心包：日常选股所需，约 950 MB
    python tools/pack_data.py --include-events # 完整包：额外含研究用事件表，约 3.0 GB
    python tools/pack_data.py --no-compress    # 不压缩（打包快，上传慢）

输出:
    dist/cache_core_YYYYMMDD.zip  或  dist/cache_full_YYYYMMDD.zip
用法（组员侧）:
    unzip cache_core_YYYYMMDD.zip -d cache/
"""
from __future__ import annotations

import argparse
import os
import time
import zipfile
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, "cache")
DIST = os.path.join(BASE, "dist")

# 核心包排除的研究用大文件（跟日常选股无关）
RESEARCH_ONLY = ("events_",)
# 只在全历史回测/因子研究里用到，日常 run_daily 不读
RESEARCH_ONLY_EXACT = ("kline_1100.pkl", "kline_feat.pkl")


def mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


def collect(include_events: bool):
    picked = []
    for name in sorted(os.listdir(CACHE)):
        path = os.path.join(CACHE, name)
        if not os.path.isfile(path) or not name.endswith(".pkl"):
            continue
        if not include_events:
            if name.startswith(RESEARCH_ONLY):
                continue
            if name in RESEARCH_ONLY_EXACT:
                continue
        picked.append((name, path))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-events", action="store_true",
                    help="把 events_*.pkl 等研究用事件表一起打包")
    ap.add_argument("--no-compress", action="store_true",
                    help="不压缩（更快，但文件更大）")
    args = ap.parse_args()

    files = collect(args.include_events)
    if not files:
        raise SystemExit("cache/ 里没有可打包的 pkl，请先跑一次数据准备")

    total = sum(mb(p) for _, p in files)
    os.makedirs(DIST, exist_ok=True)
    tag = "full" if args.include_events else "core"
    out = os.path.join(DIST, "cache_%s_%s.zip" % (tag, datetime.now().strftime("%Y%m%d")))

    mode = zipfile.ZIP_STORED if args.no_compress else zipfile.ZIP_DEFLATED
    print("打包 %d 个文件，原始 %.0f MB" % (len(files), total), flush=True)
    print("输出 %s（%s）" % (out, "不压缩" if args.no_compress else "zlib 压缩"),
          flush=True)

    t0 = time.time()
    with zipfile.ZipFile(out, "w", mode,
                         compresslevel=None if args.no_compress else 6) as zf:
        for i, (name, path) in enumerate(files, 1):
            zf.write(path, arcname=name)
            print("  [%2d/%2d] %-52s %7.1f MB   累计 %3.0fs"
                  % (i, len(files), name, mb(path), time.time() - t0), flush=True)

    print("\n完成: %s" % out, flush=True)
    print("体积: %.0f MB（压缩率 %.2fx）"
          % (mb(out), total / max(mb(out), 0.01)), flush=True)
    print("用时: %.0f 秒" % (time.time() - t0), flush=True)
    print("\n组员侧解压: unzip %s -d cache/" % os.path.basename(out), flush=True)


if __name__ == "__main__":
    main()
