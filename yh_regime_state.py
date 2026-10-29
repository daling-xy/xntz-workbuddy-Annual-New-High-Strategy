# -*- coding: utf-8 -*-
"""
中级市场周期（Regime）判定。

为什么不用单一指数：本策略在全A范围内选票，沪深300只代表大盘，
创业板指只代表成长。真正决定「新高股还能不能继续走强」的，是市场整体的
参与广度。因此主标签用「市场宽度 breadth」= 全A中站上自身 N 日均线的股票占比。

三态定义（中级 = 数周至数月级别，故用 60 日口径并对状态做最小持续期过滤）：
    UP   上涨 ：breadth60 偏高 且 指数在 MA60 上方 且 MA60 斜率向上
    DOWN 下跌 ：breadth60 偏低 且 指数在 MA60 下方 且 MA60 斜率向下
    CHOP 震荡 ：其余
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yh_datasource import CACHE_DIR  # noqa: E402
from yh_regime import INDICES, fetch_index, roll_mean, roll_max  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_klines(days: int = 1100) -> dict:
    """直接读本地 K 线缓存，避免重复联网"""
    path = os.path.join(CACHE_DIR, f"kline_{days}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_long() -> dict:
    """读拼接后的全历史缓存（2018 中至今，约 2000 根）"""
    path = os.path.join(CACHE_DIR, "kline_long2.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("先运行 yh_fetch_long2.py 生成长历史缓存")
    with open(path, "rb") as f:
        return pickle.load(f)

# 阈值：全样本宽度的经验分布（30%/70% 分位约 0.30/0.56），取整便于人工复算
BREADTH_UP = 0.55
BREADTH_DN = 0.35
SLOPE_W = 20                       # 宽度自身的 20 日变化，用于确认方向
MIN_HOLD = 5                       # 状态最短持续交易日，滤掉来回抖动


def roll_mean_np(x: np.ndarray, w: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    s = np.maximum(0, idx - w + 1)
    return (csum[idx + 1] - csum[s]) / (idx - s + 1)


def build_breadth(src: str = "long", ma_w: int = 60, force: bool = False) -> pd.DataFrame:
    """市场宽度：每个交易日全A中有多少比例的股票站上自身 MA{ma_w}"""
    path = os.path.join(CACHE, f"breadth_{src}_{ma_w}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    kmap = load_long() if src == "long" else load_klines(int(src))

    # 全局交易日轴（各股共用同一交易日历）
    axis = sorted({b["date"] for bars in kmap.values() for b in bars})
    aidx = pd.Index(axis)
    L = len(axis)

    cnt = np.zeros(L); s20 = np.zeros(L); s60 = np.zeros(L)
    s120 = np.zeros(L); sbull = np.zeros(L)

    for code, bars in kmap.items():
        if not bars or len(bars) < ma_w + 5:
            continue
        pos = aidx.get_indexer([b["date"] for b in bars])
        keep = pos >= 0
        if not keep.all():
            continue
        c = np.array([float(b["close"]) for b in bars])
        ma20 = roll_mean_np(c, 20)
        ma60 = roll_mean_np(c, ma_w)
        ma120 = roll_mean_np(c, 120)
        cnt[pos] += 1
        s20[pos] += (c > ma20)
        s60[pos] += (c > ma60)
        s120[pos] += (c > ma120)
        sbull[pos] += (ma20 > ma60)

    valid = cnt >= 2000
    br = pd.DataFrame({
        "date": np.array(axis, dtype=object)[valid],
        "breadth20": (s20[valid] / cnt[valid]).astype(float),
        "breadth": (s60[valid] / cnt[valid]).astype(float),
        "breadth120": (s120[valid] / cnt[valid]).astype(float),
        "bull_stack": (sbull[valid] / cnt[valid]).astype(float),
        "nstock": cnt[valid].astype(int),
    })

    with open(path, "wb") as f:
        pickle.dump(br, f)
    return br


def index_panel(force: bool = False) -> pd.DataFrame:
    """把各指数的 MA60 位置与斜率合到一张表"""
    path = os.path.join(CACHE, "index_panel.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    parts = []
    for code, setcode, nm in INDICES:
        try:
            df = fetch_index(code, setcode)
        except Exception as e:
            print(f"  [跳过] {nm}: {e}")
            continue
        if not len(df):
            continue
        c = df["close"].values.astype(float)
        ma60 = roll_mean(c, 60)
        slope = np.full(len(c), np.nan)
        slope[20:] = ma60[20:] / ma60[:-20] - 1.0
        ret60 = np.full(len(c), np.nan)
        ret60[60:] = c[60:] / c[:-60] - 1.0
        dd = c / roll_max(c, 250) - 1.0
        parts.append(pd.DataFrame({
            "date": df["date"].values,
            f"{nm}_pos": c / ma60 - 1.0,
            f"{nm}_slope": slope,
            f"{nm}_ret60": ret60,
            f"{nm}_dd250": dd,
        }))
    if not parts:
        raise RuntimeError("五大指数全部拉取失败（通达信接口不可用？）—— 周期判定本身只用市场宽度，"
                           "指数面板仅作观察字段；可沿用旧缓存继续。")
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def build_flag_refWindows():  # noqa: N802 - 占位，保持 API 稳定
    return None


def assign_state(br: pd.DataFrame, ipx: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    三态标签。判定只依赖市场宽度本身（水平 + 方向），不掺入市值加权指数：
    本策略是在全A选个股，2021 年「沪深300 下跌、多数个股上涨」的分化市若用
    沪深300 做趋势过滤会被误判为下跌，因此指数只作为观察字段保留。
    """
    df = br.copy().sort_values("date").reset_index(drop=True)
    if ipx is not None and len(ipx):
        df = df.merge(ipx, on="date", how="left")

    b = df["breadth"].values
    slope = np.zeros(len(df))
    if "breadth_slope" in df.columns:
        slope = np.nan_to_num(df["breadth_slope"].values, nan=0.0)
    else:
        s = pd.Series(b).diff(SLOPE_W)
        slope = np.nan_to_num(s.values, nan=0.0)
        df["breadth_slope"] = slope

    raw = np.full(len(df), 0, dtype=int)          # 0=CHOP 1=UP -1=DOWN
    raw[(b >= BREADTH_UP) & (slope > 0)] = 1
    raw[(b <= BREADTH_DN) & (slope < 0)] = -1

    # 去抖（因果）：只有当「过去 MIN_HOLD-1 天 + 当日」都属于同一状态时，才确认该状态。
    # 必须用回溯窗口而非居中窗口，否则最新几个交易日的标签会依赖未来数据而无法实盘使用。
    n = len(raw)
    keep = np.zeros(n, dtype=bool)
    for t in range(n):
        lo = max(0, t - MIN_HOLD + 1)
        seg = raw[lo: t + 1]
        keep[t] = bool(np.all(seg == raw[t]))
    state = np.where(keep, raw, 0)

    df["state_raw"] = raw
    df["state"] = state
    df["regime"] = np.select([state == 1, state == -1], ["UP", "DOWN"], default="CHOP")
    return df


def regime_map(state_df: pd.DataFrame) -> dict:
    return dict(zip(state_df["date"], state_df["regime"]))


def describe_by_year(state_df: pd.DataFrame):
    s = state_df.copy()
    s["year"] = s["date"].str[:4]
    return s.groupby(["year", "regime"]).size().unstack(fill_value=0)


if __name__ == "__main__":
    print("计算市场宽度（长历史）...")
    br = build_breadth("long")
    ok = br[br["nstock"] >= 2000]
    print(ok.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).T.to_string())
    print("区间:", ok["date"].min(), "~", ok["date"].max())
    ok["year"] = ok["date"].str[:4]
    print("\n各年度宽度中位数：")
    print(ok.groupby("year")["breadth"].median().round(3).to_string())
    print("\n宽度全样本分位：", {q: round(float(ok['breadth'].quantile(q)), 3)
                                for q in [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]})
