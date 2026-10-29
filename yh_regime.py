# -*- coding: utf-8 -*-
"""
基准指数日线（通达信 MCP）→ 中级市场周期（regime）标签。

设计要点：
  1) 不从单一指数判断牛熊，而是用「宽基组合 + 市场宽度」共同描述中级周期：
       - 沪深300 / 中证500 / 中证1000 / 创业板指 / 上证指数
       - 等权全A指数（由本地全市场日线自行合成，最能代表本策略的实际选股池）
  2) 周期定义为「中级」= 数周至数月的趋势状态，典型口径：
       - 指数与其 60 日均线的相对位置
       - 60 日均线自身的斜率
  3) 输出每日 regime 标签，供事件表按周期分层回测。
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yh_tdx  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
os.makedirs(CACHE, exist_ok=True)

# code, setcode, name
INDICES = [
    ("000300", "62", "沪深300"),
    ("000905", "62", "中证500"),
    ("000852", "62", "中证1000"),
    ("399006", "0", "创业板指"),
    ("000001", "1", "上证指数"),
]


def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    s = np.maximum(0, idx - w + 1)
    return (csum[idx + 1] - csum[s]) / (idx - s + 1)


def roll_max(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.maximum.accumulate(x)
    sw = sliding_window_view(x, w).max(axis=1)
    out = np.empty(len(x), dtype=float)
    out[:w - 1] = np.maximum.accumulate(x)[:w - 1]
    out[w - 1:] = sw
    return out


def _call_text(cli, arguments) -> str:
    """MCP call 已直接返回文本"""
    r = cli.call("tdx_kline", arguments)
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        cont = r.get("content")
        if cont:
            return "\n".join(c.get("text", "") for c in cont if isinstance(c, dict))
        return json.dumps(r, ensure_ascii=False)
    return json.dumps(r, ensure_ascii=False)


def _parse_kline_text(txt: str) -> list:
    """通达信返回的 '【名称】...\n详细K线数据:\n{...}' 结构"""
    out = []
    i = txt.find("{")
    while i >= 0:
        j = txt.find("\n}", i)                       # JSON 块结束
        seg = txt[i:] if j < 0 else txt[i:j + 2]
        try:
            obj = json.loads(seg)
        except Exception:
            # 尝试按大括号配对截取
            depth, k = 0, i
            while k < len(txt):
                if txt[k] == "{":
                    depth += 1
                elif txt[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            try:
                obj = json.loads(txt[i:k + 1])
            except Exception:
                return out
        for row in obj.get("Rows", []) or []:
            d = str(row.get("Data", ""))
            if len(d) < 8 or not d[:8].isdigit():
                continue
            try:
                out.append({
                    "date": d[:4] + "-" + d[4:6] + "-" + d[6:8],
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "amount": float(row.get("Amount", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
        return out
    return out


def fetch_index(code: str, setcode: str, want: int = 2600) -> pd.DataFrame:
    path = os.path.join(CACHE, f"tdx_idx_{code}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    cli = yh_tdx.TdxMcp()
    rows, off = [], 0
    while len(rows) < want:
        txt = _call_text(cli, {
            "code": code, "setcode": setcode, "period": "4",
            "wantNum": min(1000, want - len(rows)), "startxh": off, "tqFlag": "0",
        })
        got = _parse_kline_text(txt)
        if not got:
            break
        rows = got + rows                     # 每段按时间升序，往前拼接
        if len(got) < 100:
            break
        off += len(got)

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df



def build_equal_weight_index(kmap: dict, tail: int = 1200) -> pd.DataFrame:
    """用本地全市场日线合成等权全A指数（尾盘不稳缺失自动对齐）"""
    path = os.path.join(CACHE, "eqw_index.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    series = {}
    for code, bars in kmap.items():
        if not bars or len(bars) < 260:
            continue
        d = [b["date"] for b in bars][-tail:]
        c = [float(b["close"]) for b in bars][-tail:]
        series[code] = pd.Series(c, index=d)
    px = pd.DataFrame(series).sort_index()
    px = px.apply(lambda col: col / col.dropna().iloc[0] if col.dropna().shape[0] else col)
    ret = px.pct_change()
    eqret = ret.mean(axis=1, skipna=True)          # 等权日收益
    idx = (1 + eqret.fillna(0)).cumprod()
    df = pd.DataFrame({"date": idx.index, "close": idx.values})
    df["high"] = df["close"]
    df["low"] = df["close"]
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def label_regime(idx: pd.DataFrame, ma_w: int = 60, slope_w: int = 20,
                 name: str = "") -> pd.DataFrame:
    """对一个指数序列打中级周期标签"""
    c = idx["close"].values.astype(float)
    ma = roll_mean(c, ma_w)
    ma_hi = roll_max(c, ma_w)
    slope = np.full(len(c), np.nan)
    if len(c) > slope_w:
        slope[slope_w:] = ma[slope_w:] / ma[:-slope_w] - 1.0
    ret60 = np.full(len(c), np.nan)
    if len(c) > 60:
        ret60[60:] = c[60:] / c[:-60] - 1.0
    dd = c / ma_hi - 1.0

    above = c > ma
    out = pd.DataFrame({
        "date": idx["date"].values,
        f"{name}_close": c,
        f"{name}_ma{ma_w}": ma,
        f"{name}_slope": slope,
        f"{name}_ret60": ret60,
        f"{name}_dd60": dd,
        f"{name}_above": above.astype(float),
    })
    return out


if __name__ == "__main__":
    for code, setcode, nm in INDICES:
        try:
            df = fetch_index(code, setcode)
            print(f"{nm:8s} {code} setcode={setcode:>2s}  bar数={len(df):5d}  "
                  f"{df['date'].min() if len(df) else '-'} ~ {df['date'].max() if len(df) else '-'}")
        except Exception as e:
            print(f"{nm:8s} {code} 失败: {type(e).__name__} {str(e)[:120]}")
