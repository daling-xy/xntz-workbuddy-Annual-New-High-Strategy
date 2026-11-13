# -*- coding: utf-8 -*-
"""打一个"发给别人就能用"的代码包。

只装代码和文档，不装数据（数据另外发 dist/cache_core_*.zip）。

    python tools/make_dist.py
    python tools/make_dist.py --out dist/xxx.zip

排除规则写在下面，改这里就能调整包里有什么。
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 整个目录跳过
EXCLUDE_DIRS = {
    ".git", "cache", "output", "dist", "versions", "archive",
    "__pycache__", ".venv", "venv", ".idea", ".vscode", ".pytest_cache",
}

# 按后缀跳过（数据、日志、缓存）
EXCLUDE_SUFFIX = {
    ".pkl", ".parquet", ".feather", ".h5", ".hdf5",
    ".pyc", ".pyo", ".log", ".zip", ".tmp",
}

# 单独点名跳过
EXCLUDE_FILES = {
    "state.json", "alerts.jsonl", "reminder_state.json",
    "research/yh_regime_rebuild.py",  # 用户指定不外发
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDE_FILES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIX:
        return True
    name = path.name
    if name.startswith(".env") and name != ".env.example":
        return True
    return False


def collect() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        current = Path(dirpath)
        # 原地修改 dirnames 才会真正跳过子目录
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            p = current / name
            if should_skip(p):
                continue
            files.append(p)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="输出路径，默认 dist/yearly_high_strategy_code_<日期>.zip")
    ap.add_argument("--prefix", default="yearly_high_strategy", help="压缩包里的一级目录名")
    args = ap.parse_args()

    files = collect()
    out = Path(args.out) if args.out else ROOT / "dist" / f"yearly_high_strategy_code_{date.today():%Y%m%d}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    total_raw = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            arc = f"{args.prefix}/{p.relative_to(ROOT).as_posix()}"
            zf.write(p, arc)
            total_raw += p.stat().st_size

    size = out.stat().st_size
    print(f"文件数 : {len(files)}")
    print(f"原始   : {total_raw / 1024:.0f} KB")
    print(f"压缩后 : {size / 1024:.0f} KB")
    print(f"输出   : {out}")

    # 打印一级目录构成，方便核对有没有漏/多
    tops: dict[str, int] = {}
    for p in files:
        top = p.relative_to(ROOT).parts[0] if len(p.relative_to(ROOT).parts) > 1 else "(根目录)"
        tops[top] = tops.get(top, 0) + 1
    print("\n一级构成：")
    for k, v in sorted(tops.items()):
        print(f"  {k:<14} {v} 个")

    # 安全检查：包里不能有数据文件或密钥
    bad = [p for p in files if p.suffix.lower() in {".pkl", ".parquet"}]
    if bad:
        print(f"\n❌ 包里有数据文件：{bad[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
