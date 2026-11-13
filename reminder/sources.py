# -*- coding: utf-8 -*-
"""数据源适配层 —— 这是唯一需要"接别人数据"的地方。

接口只有一个方法：

    class MySource:
        def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
            ...

约定：
  * 返回值形如 {"600519": {"close": 1500.0, "change_pct": 1.2, ...}, ...}
  * symbols 为空序列 = "把你有的全给我"
  * 取不到的标的可以不返回，或返回 None
  * 字段名随便定，只要和规则 JSON 里写的一致
  * 嵌套结构直接放 dict，规则里用 `kline.ma20` 这种点号访问

本文件自带四个可直接用的实现（静态字典 / 函数包装 / JSONL 回放 / CSV），
以及一个可选的腾讯实时行情示例 —— 不需要它也能跑。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


class QuoteSource:
    """所有数据源的结构约定（不是必须继承，鸭子类型即可）。"""

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        raise NotImplementedError


def _pick(
    snapshot: Mapping[str, Mapping[str, Any]],
    symbols: Sequence[str],
) -> Mapping[str, Mapping[str, Any]]:
    if not symbols:
        return snapshot
    return {s: snapshot[s] for s in symbols if s in snapshot}


class MappingSource(QuoteSource):
    """固定快照。最适合写测试、做演示。"""

    def __init__(self, snapshot: Mapping[str, Mapping[str, Any]]):
        self.snapshot = dict(snapshot)

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        return _pick(self.snapshot, symbols)

    def update(self, patch: Mapping[str, Mapping[str, Any]]) -> None:
        for sym, fields in patch.items():
            self.snapshot.setdefault(sym, {}).update(fields)


class CallableSource(QuoteSource):
    """把任意函数包装成数据源。

        CallableSource(lambda symbols: {"600519": {"close": 1500}})
    """

    def __init__(self, fn: Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]]]):
        self.fn = fn

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        return self.fn(list(symbols)) or {}


class JSONLSource(QuoteSource):
    """逐行回放 JSONL，每行是一个完整的快照。

    用于把历史行情、录下来的推送流当成"实时"重放。
    fetch() 每次调用前进一行；走完返回空字典。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._rows: List[Dict[str, Any]] = []
        self._idx = 0
        self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))

    def reset(self) -> None:
        self._idx = 0

    @property
    def exhausted(self) -> bool:
        return self._idx >= len(self._rows)

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        if self.exhausted:
            return {}
        row = self._rows[self._idx]
        self._idx += 1
        # 允许 {"snapshot": {...}} 或直接 {...}
        snap = row.get("snapshot") if isinstance(row.get("snapshot"), dict) else row
        return _pick(snap, symbols)


class CSVSource(QuoteSource):
    """读取 CSV，返回每个标的最新一行。

    必需列：symbol
    其余列自动变成字段（能转数字就转数字）。
    """

    def __init__(self, path: Path, time_col: str = ""):
        self.path = Path(path)
        self.time_col = time_col
        self._snapshot: Dict[str, Dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _cast(v: str) -> Any:
        s = (v or "").strip()
        if s == "":
            return None
        try:
            return int(s) if s.lstrip("-").isdigit() else float(s)
        except ValueError:
            return s

    def _load(self) -> None:
        with open(self.path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "symbol" not in reader.fieldnames:
                raise ValueError(f"{self.path} 缺少必需列 symbol")
            for row in reader:
                sym = (row.get("symbol") or "").strip()
                if not sym:
                    continue
                fields = {k: self._cast(v) for k, v in row.items() if k != "symbol"}
                if self.time_col and self.time_col in fields:
                    fields["_time"] = fields[self.time_col]
                self._snapshot[sym] = fields

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        return _pick(self._snapshot, symbols)


# ------------------------------------------------------------------ 可选示例
class TencentSource(QuoteSource):
    """腾讯实时行情（示例，可选）。

    不依赖第三方库，只用 urllib。字段名已转成常用写法：

        name, close, prev_close, open, high, low, volume, amount,
        change, change_pct, turnover, pe, float_cap, total_cap,
        limit_up, limit_down, amplitude, time

    用法：
        from reminder.sources import TencentSource
        src = TencentSource()
        src.fetch(["600519", "000001"])
    """

    URL = "https://qt.gtimg.cn/q={q}"

    _FIELDS = {
        1: "name", 2: "code", 3: "close", 4: "prev_close", 5: "open",
        6: "volume", 30: "time", 31: "change", 32: "change_pct",
        33: "high", 34: "low", 36: "volume_hand", 37: "amount_wan",
        38: "turnover", 39: "pe", 43: "amplitude", 44: "float_cap",
        45: "total_cap", 46: "pb", 47: "limit_up", 48: "limit_down",
    }
    _NUMERIC = {
        "close", "prev_close", "open", "volume", "change", "change_pct",
        "high", "low", "volume_hand", "amount_wan", "turnover", "pe",
        "amplitude", "float_cap", "total_cap", "pb", "limit_up", "limit_down",
    }

    @staticmethod
    def normalize(symbol: str) -> str:
        s = str(symbol).strip()
        if s[:2].lower() in ("sh", "sz", "bj"):
            return s.lower()
        if s.startswith(("60", "68", "51", "58", "56")):
            return "sh" + s
        if s.startswith(("00", "30", "15", "16", "12")):
            return "sz" + s
        if s.startswith(("4", "8", "92")):
            return "bj" + s
        return "sh" + s

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        from urllib.request import Request, urlopen

        syms = [self.normalize(s) for s in (symbols or [])]
        if not syms:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(syms), 60):  # 一次最多几十只
            batch = syms[i:i + 60]
            req = Request(self.URL.format(q=",".join(batch)),
                          headers={"User-Agent": "Mozilla/5.0"})
            raw = urlopen(req, timeout=8).read().decode("gbk", errors="replace")
            for line in raw.split(";"):
                line = line.strip()
                if not line or "=" not in line:
                    continue
                _, _, payload = line.partition("=")
                parts = payload.strip().strip('"').split("~")
                if len(parts) < 40:
                    continue
                code = parts[2]
                rec: Dict[str, Any] = {}
                for idx, key in self._FIELDS.items():
                    if idx >= len(parts):
                        continue
                    val = parts[idx]
                    if key in self._NUMERIC:
                        try:
                            val = float(val)
                        except (TypeError, ValueError):
                            continue
                    rec[key] = val
                out[code] = rec
        return out


__all__ = [
    "QuoteSource", "MappingSource", "CallableSource",
    "JSONLSource", "CSVSource", "TencentSource",
]
