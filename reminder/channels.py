# -*- coding: utf-8 -*-
"""推送渠道 —— 告警往哪里发。

每个渠道只要实现一个方法：

    def send(self, alert: Alert) -> None:
        ...

内置：控制台、文件、Windows 弹窗、Webhook（通用/企微/飞书）、自定义函数。
换成微信、钉钉、邮件、Telegram，都是同样的写法 —— 传进来的 alert 是纯数据。
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional


@dataclass
class Alert:
    rule_id: str
    rule_name: str
    symbol: str
    severity: str
    message: str
    ts: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "symbol": self.symbol,
            "severity": self.severity,
            "message": self.message,
            "fields": self.fields,
        }

    def line(self) -> str:
        return f"[{self.ts}] [{self.severity.upper()}] {self.message}"


class Channel:
    def send(self, alert: Alert) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ConsoleChannel(Channel):
    """打到终端。severity 用颜色区分（Windows 新版终端支持 ANSI）。"""

    _COLOR = {"info": "\033[36m", "warn": "\033[33m", "critical": "\033[31m"}
    _RESET = "\033[0m"

    def __init__(self, color: bool = True):
        self.color = color and sys.stdout.isatty()

    def send(self, alert: Alert) -> None:
        text = alert.line()
        if self.color:
            text = f"{self._COLOR.get(alert.severity, '')}{text}{self._RESET}"
        print(text, flush=True)


class FileChannel(Channel):
    """追加写入文本文件，按大小自动切分。"""

    def __init__(self, path: Path | str, max_bytes: int = 5 * 1024 * 1024):
        self.path = Path(path)
        self.max_bytes = max_bytes

    def send(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > self.max_bytes:
            self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(alert.line() + "\n")


class WindowsToastChannel(Channel):
    """Windows 右下角气泡通知，不装任何第三方库。

    PowerShell 起一个 NotifyIcon，显示 6 秒后自动退出。
    非 Windows 或没有 powershell 时降级为控制台输出。
    """

    def __init__(self, duration_seconds: int = 6, title: str = "投资提醒"):
        self.duration = max(2, int(duration_seconds))
        self.title = title

    def send(self, alert: Alert) -> None:
        if sys.platform != "win32":
            print(alert.line(), flush=True)
            return
        title = f"{self.title} · {alert.rule_name}"
        body = alert.message.replace('"', "'")
        title = title.replace('"', "'")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms\n"
            "Add-Type -AssemblyName System.Drawing\n"
            "$n = New-Object System.Windows.Forms.NotifyIcon\n"
            "$n.Icon = [System.Drawing.SystemIcons]::Information\n"
            "$n.Visible = $true\n"
            f'$n.ShowBalloonTip(6000, "{title}", "{body}", '
            "[System.Windows.Forms.ToolTipIcon]::Warning)\n"
            f"Start-Sleep -Seconds {self.duration}\n"
            "$n.Dispose()\n"
        )
        try:
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            print(alert.line(), flush=True)


class WebhookChannel(Channel):
    """POST JSON 到任意 webhook。

    format:
      generic —— {"text": ..., "rule": ..., "symbol": ..., "severity": ...}
      wecom   —— 企业微信群机器人 {"msgtype":"text","text":{"content":...}}
      feishu  —— 飞书自定义机器人 {"msg_type":"text","content":{"text":...}}

    失败只打印警告，不影响其它渠道（一条渠道挂了不能拖垮整个循环）。
    """

    TEMPLATES = {
        "generic": lambda a: {
            "text": a.message, "rule": a.rule_id, "rule_name": a.rule_name,
            "symbol": a.symbol, "severity": a.severity, "ts": a.ts,
        },
        "wecom": lambda a: {"msgtype": "text", "text": {"content": a.line()}},
        "feishu": lambda a: {"msg_type": "text", "content": {"text": a.line()}},
    }

    def __init__(self, url: str, fmt: str = "generic", timeout: float = 5.0,
                 extra_headers: Optional[Mapping[str, str]] = None):
        if not url:
            raise ValueError("WebhookChannel 需要 url")
        if fmt not in self.TEMPLATES:
            raise ValueError(f"format 必须是 {sorted(self.TEMPLATES)} 之一")
        self.url = url
        self.fmt = fmt
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def send(self, alert: Alert) -> None:
        from urllib.request import Request, urlopen

        payload = json.dumps(self.TEMPLATES[self.fmt](alert), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        headers.update(self.extra_headers)
        try:
            req = Request(self.url, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception as e:
            print(f"[warn] webhook 推送失败（{self.url}）：{e}", file=sys.stderr, flush=True)


class CallableChannel(Channel):
    """把任意函数当成渠道，方便嵌进别人的系统。"""

    def __init__(self, fn: Callable[[Alert], None]):
        self.fn = fn

    def send(self, alert: Alert) -> None:
        try:
            self.fn(alert)
        except Exception as e:
            print(f"[warn] 自定义渠道异常：{e}", file=sys.stderr, flush=True)


class MultiChannel(Channel):
    """把多个渠道当一个用。"""

    def __init__(self, channels: List[Channel]):
        self.channels = list(channels)

    def send(self, alert: Alert) -> None:
        for ch in self.channels:
            try:
                ch.send(alert)
            except Exception as e:
                print(f"[warn] 渠道 {type(ch).__name__} 失败：{e}", file=sys.stderr, flush=True)


__all__ = [
    "Alert", "Channel", "ConsoleChannel", "FileChannel",
    "WindowsToastChannel", "WebhookChannel", "CallableChannel", "MultiChannel",
]
