# -*- coding: utf-8 -*-
"""
年度新高选股策略 - 数据层

数据源（全部为公开 HTTP 接口，无需鉴权）：
  1) 腾讯财经 fqkline  —— 前复权日线：OHLC + 成交量 + 换手率 + 成交额
  2) 腾讯财经 qt.gtimg —— 实时快照：名称 / 流通市值 / 总市值 / 市盈率 / 市净率 / 是否 ST
  3) 东方财富 datacenter —— 业绩报表：营收、营收同比、净利同比、加权ROE、毛利率、行业
  4) 东方财富 emweb F10 —— 十大流通股东（基金家数）、股东户数

设计原则：
  - 所有网络请求自带重试 + 本地缓存，重跑不重复取数
  - 数据层与策略层解耦，换数据源只需替换本文件的 fetch_* 函数
"""
from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
DEFAULT_TIMEOUT = 25


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False          # 绕过本机代理，直连
    s.headers.update(UA)
    return s


def _get(url: str, referer: str = "", tries: int = 3, timeout: int = DEFAULT_TIMEOUT) -> Optional[requests.Response]:
    for i in range(tries):
        try:
            h = dict(UA)
            if referer:
                h["Referer"] = referer
            r = _session().get(url, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(0.4 * (i + 1))
    return None


# ---------------------------------------------------------------- 财报时效：披露截止日与缓存命名
# A 股定期报告法定披露截止日：一季报 4/30、半年报 8/31、三季报 10/31、年报次年 4/30
_REPORT_DEADLINE = {
    (3, 31): (4, 30, 0),      # 一季报 -> 同年 4/30
    (6, 30): (8, 31, 0),      # 半年报 -> 同年 8/31
    (9, 30): (10, 31, 0),     # 三季报 -> 同年 10/31
    (12, 31): (4, 30, 1),     # 年报   -> 次年 4/30
}


def report_deadline(report_date: str):
    """返回该报告期的法定披露截止日（datetime.date）"""
    import datetime as _dt
    y, m, d = map(int, report_date.split("-")[:3])
    mm, dd, yoff = _REPORT_DEADLINE.get((m, d), (m, d, 0))
    return _dt.date(y + yoff, mm, dd)


def report_available_date(report_date: str):
    """该报告期【最晚可得日】= 法定截止日 + 3 天缓冲（同 yh_rev_fine 的口径）"""
    import datetime as _dt
    return report_deadline(report_date) + _dt.timedelta(days=3)


def fin_cache_name(report_dates: Sequence[str]) -> str:
    """财报缓存文件名 —— 必须与 fetch_fundamentals 内部的 key 算法保持一致。

    此前 run_daily.py 把这个文件名写死成了旧字符串，导致 REPORT_DATES 变更后
    缓存清除失效（刷新形同虚设）。这里集中定义，杜绝再次漂移。
    """
    return "fin_" + "_".join(d[:7] for d in report_dates) + ".pkl"


def fin_still_disclosing(report_dates: Sequence[str], today=None) -> list:
    """返回【仍在陆续披露中】的报告期列表。

    在截止日之前，每天都会有新的公司披露，因此财报必须每天增量重拉，
    否则会拿着不完整的口径跑选股，个股会随时间机械性翻转。
    """
    import datetime as _dt
    today = today or _dt.date.today()
    return [rd for rd in report_dates if today <= report_available_date(rd)]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


# ---------------------------------------------------------------- 代码与 secid
def to_secid(code: str) -> str:
    """A股代码 -> 腾讯 secid，如 600519 -> sh600519"""
    code = str(code).strip().zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh" + code
    if code.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return "sz" + code
    if code.startswith(("43", "83", "87", "88", "920")):
        return "bj" + code
    return "sh" + code


def to_em_suffix(code: str) -> str:
    """A股代码 -> 东财 SECUCODE 后缀，如 600519 -> 600519.SH"""
    code = str(code).strip().zfill(6)
    if code.startswith(("6", "9")):
        return code + ".SH"
    if code.startswith(("4", "8")):
        return code + ".BJ"
    return code + ".SZ"


# ---------------------------------------------------------------- 日线（前复权）
def fetch_kline(code: str, days: int = 320) -> Optional[List[dict]]:
    """
    返回按日期升序的日线列表，字段：
      date, open, close, high, low, volume(手), turnover(%), amount(元)
    价格为前复权，amount 已由「万元」换算成「元」，便于直接与 4 亿等门槛比较。
    """
    sid = to_secid(code)
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
           f"?param={sid},day,,,{days},qfq")
    r = _get(url, referer="https://gu.qq.com/")
    if r is None:
        return None
    try:
        blk = r.json()["data"][sid]
        rows = blk.get("qfqday") or blk.get("day")
        if not rows:
            return None
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
        return out or None
    except Exception:
        return None


def fetch_klines(codes: Sequence[str], days: int = 320, workers: int = 24,
                 cache_tag: str = "") -> Dict[str, List[dict]]:
    """并发拉取多只股票日线，带本地 pickle 缓存"""
    tag = f"kline_{cache_tag or days}.pkl"
    path = _cache_path(tag)
    cached: Dict[str, List[dict]] = {}
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
        except Exception:
            cached = {}

    todo = [c for c in codes if c not in cached]
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_kline, c, days): c for c in todo}
            for fu in as_completed(futs):
                c = futs[fu]
                try:
                    res = fu.result()
                except Exception:
                    res = None
                if res:
                    cached[c] = res
                done += 1
                if done % 500 == 0:
                    print(f"  日线已取 {done}/{len(todo)} ...", flush=True)
        with open(path, "wb") as f:
            pickle.dump(cached, f)
    return cached


# ---------------------------------------------------------------- 实时快照
def fetch_snapshots(codes: Sequence[str], batch: int = 100, workers: int = 16) -> Dict[str, dict]:
    """
    腾讯批量快照。返回字段：
      name, price, turnover(%), pe_ttm, pb, float_mv(元), total_mv(元), amount(元)
    """
    path = _cache_path("snapshot.pkl")
    result: Dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
        except Exception:
            result = {}
    todo = [c for c in codes if c not in result]
    if not todo:
        return result

    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def one(grp: List[str]):
        q = ",".join(to_secid(c) for c in grp)
        r = _get(f"https://qt.gtimg.cn/q={q}", referer="https://gu.qq.com/")
        if r is None:
            return {}
        txt = r.content.decode("gbk", errors="ignore")
        out = {}
        for line in txt.split(";"):
            if "=" not in line:
                continue
            try:
                payload = line.split('="', 1)[1].rstrip('"')
                f = payload.split("~")
                if len(f) < 50:
                    continue
                code = f[2].strip().zfill(6)
                out[code] = {
                    "name": f[1].strip(),
                    "price": float(f[3]) if f[3] else 0.0,
                    "turnover": float(f[38]) if f[38] else 0.0,
                    "pe_ttm": float(f[39]) if f[39] else 0.0,
                    "pb": float(f[46]) if f[46] else 0.0,
                    "float_mv": float(f[44]) * 1e8 if f[44] else 0.0,     # 亿 -> 元
                    "total_mv": float(f[45]) * 1e8 if f[45] else 0.0,
                    "amount": float(f[37]) * 1e4 if f[37] else 0.0,       # 万 -> 元
                    "volume": float(f[36]) if f[36] else 0.0,
                }
            except Exception:
                continue
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fu in as_completed([ex.submit(one, g) for g in groups]):
            try:
                result.update(fu.result())
            except Exception:
                pass
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------- 基本面（东财业绩报表）
_EM_FIN_COLS = [
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORTDATE", "UPDATE_DATE",
    "TOTAL_OPERATE_INCOME", "YSTZ", "PARENT_NETPROFIT", "SJLTZ",
    "WEIGHTAVG_ROE", "XSMLL", "BASIC_EPS", "BPS", "MGJYXJJE",
    "BOARD_NAME", "TRADE_MARKET", "QDATE",
]


def fetch_fundamentals(report_dates: Sequence[str] = ("2026-06-30", "2025-06-30"),
                       max_pages: int = 200) -> List[dict]:
    """
    拉取全 A 股业绩报表。report_dates 里给出需要覆盖的报告期，
    用于在本地自行计算「多报告期营收增速中位数」等稳健指标。
    """
    key = "_".join(d[:7] for d in report_dates)
    path = _cache_path(f"fin_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    rdstr = ",".join(f"'{d}'" for d in report_dates)
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    rows: List[dict] = []
    page = 1
    while page <= max_pages:
        params = {
            "reportName": "RPT_LICO_FN_CPD",
            "columns": ",".join(_EM_FIN_COLS),
            "filter": f'(SECURITY_TYPE_CODE="058001001")(REPORTDATE in ({rdstr}))',
            "pageSize": 500,
            "pageNumber": page,
            "source": "WEB",
            "client": "WEB",
            "sortColumns": "SECURITY_CODE,REPORTDATE",
            "sortTypes": "1,-1",
        }
        r = _session().get(url, params=params, headers={**UA, "Referer": "https://data.eastmoney.com/"}, timeout=30)
        try:
            j = r.json()
        except Exception:
            break
        res = (j or {}).get("result") or {}
        data = res.get("data") or []
        rows.extend(data)
        if page >= res.get("pages", 0):
            break
        page += 1

    with open(path, "wb") as f:
        pickle.dump(rows, f)
    return rows


# ---------------------------------------------------------------- 十大流通股东（基金家数）
def fetch_holders(code: str) -> Optional[dict]:
    """
    东财 F10 股东研究。返回：
      top10_float: [{"name","type","num","ratio","is_fund"}, ...]
      fund_count: 十大流通股东中基金类股东家数
      holder_num / holder_num_chg: 最新股东户数及其环比
    """
    path = _cache_path("holders.pkl")
    store: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            store = pickle.load(f)
    if code in store:
        return store[code]

    secu = to_em_suffix(code)
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/PageAjax?code={secu.split('.')[1]}.{secu}"
    market = secu.split(".")[1]
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/ShareholderResearch/PageAjax?code={market}{str(code).zfill(6)}"
    r = _get(url, referer="https://emweb.securities.eastmoney.com/")
    if r is None:
        return None
    try:
        d = r.json()
    except Exception:
        return None

    FUND_KEYWORDS = ("基金", "资产管理", "资管", "理财产品", "信托", "社保基金", "年金", "组合")
    top10 = []
    for it in (d.get("sdltgd") or [])[:10]:
        name = (it.get("HOLDER_NAME") or "").strip()
        htype = (it.get("HOLDER_TYPE") or "").strip()
        blob = name + htype
        # 基金股东口径：名称含「基金」，排除社保/年金/私募（口径=公募及公募专户）
        is_fund = ("基金" in blob) and ("社保" not in blob) and ("私募" not in blob) and ("年金" not in blob)
        top10.append({
            "name": name,
            "type": htype,
            "num": it.get("HOLD_NUM"),
            "ratio": it.get("FREE_HOLDNUM_RATIO"),
            "raw_kw": any(k in blob for k in FUND_KEYWORDS),
            "is_fund": bool(is_fund),
        })
    gdrs = (d.get("gdrs") or [])
    holder_num = None
    holder_chg = None
    if gdrs:
        holder_num = gdrs[0].get("HOLDER_TOTAL_NUM")
        holder_chg = gdrs[0].get("TOTAL_NUM_RATIO")

    out = {
        "top10_float": top10,
        "fund_count": sum(1 for x in top10 if x["is_fund"]),
        "holder_num": holder_num,
        "holder_num_chg": holder_chg,
        "end_date": (top10[0].get("name") and None) or ((d.get("sdltgd") or [{}])[0].get("END_DATE")),
    }
    store[code] = out
    with open(path, "wb") as f:
        pickle.dump(store, f)
    return out


def fetch_holders_many(codes: Sequence[str], workers: int = 12) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_holders, c): c for c in codes}
        for fu in as_completed(futs):
            c = futs[fu]
            try:
                v = fu.result()
            except Exception:
                v = None
            if v:
                out[c] = v
    return out


if __name__ == "__main__":
    # 自检
    k = fetch_kline("688120", days=320)
    print("华海清科 bars:", len(k) if k else None, k[-1] if k else None)
    s = fetch_snapshots(["688120", "300209", "002463", "301205"])
    for c, v in s.items():
        print(c, v)
