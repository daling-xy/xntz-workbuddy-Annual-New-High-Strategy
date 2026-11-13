# -*- coding: utf-8 -*-
"""接自己数据的模板 —— 复制这个文件，改 fetch 里的三行就能用。

不需要继承任何类，有 fetch(symbols) 这个方法就行。

用法：
  1. 把这个文件放到你自己的目录，比如 my_quotes.py
  2. 在 config.json 里写：
       "source": {"factory": "my_quotes:MySource", "args": {"token": "xxx"}}
  3. 跑 python -m reminder once --config config.json
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


class MySource:
    """返回 {"代码": {"字段名": 值, ...}}。

    symbols 为空列表 = "把你有的全部给我"。
    取不到的标的可以不返回，返回 None 也行。
    """

    def __init__(self, token: str = ""):
        self.token = token
        # 这里可以做一次初始化：读本地文件、建连接、登录……

    def fetch(self, symbols: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        wanted: List[str] = list(symbols)  # 空列表 = 全量

        # ------- 改这里：把你自己的数据取回来 -------
        raw = self._load_my_data(wanted)
        # -------------------------------------------

        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            code = str(item["code"])
            out[code] = {
                "name": item.get("name", ""),
                "close": float(item["price"]),
                "prev_close": float(item.get("prev_close", 0) or 0),
                "change_pct": float(item.get("change_pct", 0) or 0),
                # 想加什么字段就加什么，规则里写上同名就行
            }
        return out

    # 下面只是个占位实现，换成你真实的取数逻辑
    def _load_my_data(self, wanted: List[str]) -> List[Dict[str, Any]]:
        rows = [
            {"code": "600519", "name": "贵州茅台", "price": 1450.0,
             "prev_close": 1446.0, "change_pct": 0.28},
            {"code": "000001", "name": "平安银行", "price": 11.2,
             "prev_close": 11.0, "change_pct": 1.82},
        ]
        if not wanted:
            return rows
        keep = set(wanted)
        return [r for r in rows if r["code"] in keep]


# 更省事的写法：不用写类，一个函数就行
def fetch_quotes(symbols: Sequence[str]):
    return {
        "600519": {"name": "贵州茅台", "close": 1450.0},
    }


# 在 config.json 里这样挂函数：
#   "source": {"factory": "my_quotes:fetch_quotes"}
# 但这个函数的签名必须收一个参数（可以是空列表）
