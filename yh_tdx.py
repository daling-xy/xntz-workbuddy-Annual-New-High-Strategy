# -*- coding: utf-8 -*-
"""
通达信 MCP 客户端封装。

本机 tdx-connector 以 MCP-over-HTTP(SSE) 提供，需要先 initialize 换取 session id，
之后所有调用都要带上 mcp-session-id。本模块把这些细节封装成一个 call() 方法。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests


class TdxMcp:
    def __init__(self, entry: str = "tdx-connector"):
        cfg = json.loads(os.environ["CODEBUDDY_MCP_CONFIG"])
        e = cfg["mcpServers"][entry]
        self.url = e["url"]
        self.headers = dict(e.get("headers") or {})
        self.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })
        self.s = requests.Session()
        self.s.trust_env = False
        self._rpc_id = 0
        self._connect()

    def _post(self, payload: dict, timeout: int = 120) -> requests.Response:
        return self.s.post(self.url, headers=self.headers, json=payload, timeout=timeout)

    def _connect(self) -> None:
        r = self._post({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "yhs", "version": "1.0"}},
        })
        r.raise_for_status()
        sid = r.headers.get("mcp-session-id")
        if sid:
            self.headers["mcp-session-id"] = sid
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    @staticmethod
    def _parse(resp: requests.Response) -> Any:
        """响应是 SSE：逐行找 data: {...}，按 utf-8 解码"""
        txt = resp.content.decode("utf-8", errors="replace")
        for line in txt.splitlines():
            if line.startswith("data:"):
                body = line[5:].strip()
                if body and body != "[DONE]":
                    try:
                        return json.loads(body)
                    except Exception:
                        continue
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt[:2000]}

    def call(self, tool: str, arguments: Dict[str, Any], timeout: int = 180) -> Any:
        self._rpc_id += 1
        r = self._post({
            "jsonrpc": "2.0", "id": self._rpc_id, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }, timeout=timeout)
        d = self._parse(r)
        if isinstance(d, dict) and "error" in d:
            return {"_error": d["error"]}
        res = (d or {}).get("result", d)
        content = (res or {}).get("content") if isinstance(res, dict) else None
        if isinstance(content, list):
            txt = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            try:
                return json.loads(txt)
            except Exception:
                return txt
        return res

    def list_tools(self) -> List[Dict]:
        self._rpc_id += 1
        r = self._post({"jsonrpc": "2.0", "id": self._rpc_id, "method": "tools/list", "params": {}})
        d = self._parse(r)
        return (d.get("result") or {}).get("tools", [])


def kline(code: str, period: str = "day", count: int = 500, retry: int = 3) -> Optional[dict]:
    """通达信 K 线，统一入口"""
    for _ in range(retry):
        try:
            return TdxMcp().call("tdx_kline", {"code": code, "period": period, "count": count})
        except Exception:
            continue
    return None


if __name__ == "__main__":
    cli = TdxMcp()
    tools = cli.list_tools()
    print("通达信工具数:", len(tools))
    for t in tools:
        print(" -", t["name"])
