# reminder — 规则化实时提醒

给它一份行情字典，它按你写的规则判断，触发就推送。

**数据源完全可插拔**：你只需要写一个 `fetch(symbols)` 方法，剩下的事它全包了。

---

## 30 秒上手

```bash
# 1. 用假数据看一遍完整链路（不联网）
python -m reminder demo

# 2. 校验规则文件写得对不对
python -m reminder validate

# 3. 用真实行情跑一轮（默认接腾讯，可换）
python -m reminder once

# 4. 常驻运行，触发就弹窗
python -m reminder run
```

要求：Python 3.9+，**零第三方依赖**（只用标准库）。

---

## 一、接自己的数据（唯一需要改的地方）

### 接口就一个方法

```python
def fetch(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
    ...
```

| 约定 | 说明 |
|---|---|
| 返回值 | `{"600519": {"close": 1450.0, "change_pct": 0.28}, ...}` |
| `symbols` 为空列表 | 表示"把你有的全部给我" |
| 取不到某只票 | 可以不返回，或返回 `None` |
| 字段名 | **随便定**，只要和规则 JSON 里写的一致 |
| 嵌套字段 | 直接放 dict，规则里用 `kline.ma20` 访问 |

### 最小实现（10 行）

```python
class MySource:
    def fetch(self, symbols):
        rows = my_api.get_quotes(symbols)          # ← 换成你的取数
        return {
            r["code"]: {
                "close": r["price"],
                "name": r["name"],
                "change_pct": r["pct"],
            }
            for r in rows
        }
```

### 挂到配置里

```json
{
  "source": {"factory": "my_quotes:MySource", "args": {"token": "xxx"}}
}
```

`factory` 的格式是 `模块路径:类名`。项目里放了一份可直接复制的模板：
`reminder/examples/custom_source.py`。

### 不想写类？用函数也行

```python
def fetch_quotes(symbols):
    return {"600519": {"close": 1450.0}}
```

```json
{"source": {"factory": "my_quotes:fetch_quotes"}}
```

### 项目自带的四个数据源

| 类 | 用途 |
|---|---|
| `MappingSource` | 固定字典，测试用 |
| `CallableSource` | 包一个函数 |
| `JSONLSource` | 逐行回放历史快照 |
| `CSVSource` | 读 CSV，每个代码取最新一行 |
| `TencentSource` | 腾讯实时行情（示例，可选） |

---

## 二、写规则

一条规则一个 JSON 文件，放在 `reminder/rules/` 下。

```json
{
  "id": "stop_loss_600519",
  "name": "贵州茅台跌破止损",
  "enabled": true,
  "scope": "symbols",
  "symbols": ["600519"],
  "conditions": [
    {"field": "close", "op": "<=", "value": 1450.0}
  ],
  "logic": "and",
  "cooldown_seconds": 3600,
  "severity": "critical",
  "message": "{symbol} {name} 现价 {close}，跌破止损"
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✅ | 小写字母/数字/下划线，1-40 字符，**不能重复** |
| `name` | ✅ | 人看的名字 |
| `conditions` | ✅ | 条件数组，至少一条 |
| `enabled` | | 默认 `true` |
| `scope` | | `symbols`（默认）或 `all`（全市场） |
| `symbols` | | `scope=symbols` 时必填 |
| `logic` | | `and`（默认）/ `or` |
| `cooldown_seconds` | | 默认 300，**按"规则+标的"计** |
| `severity` | | `info` / `warn` / `critical` |
| `channels` | | 为空 = 按 severity 走全局路由 |
| `message` | | 支持 `{字段名}` 占位符 |

### 条件运算符

| 运算符 | 含义 | 例子 |
|---|---|---|
| `>` `>=` `<` `<=` `==` `!=` | 数值比较（字符串数字会自动转） | `{"field": "close", "op": "<", "value": 10}` |
| `cross_above` | **上穿**：上轮 ≤ 阈值，本轮 > 阈值 | `{"field": "close", "op": "cross_above", "ref": "kline.ma20"}` |
| `cross_below` | **下穿** | 同上 |

### 拿别的字段当阈值

不写 `value`，改写成 `ref`，就能"现价 vs 涨停价"这种比较：

```json
{"field": "close", "op": ">=", "ref": "limit_up"}
```

### 三条容易踩的坑

1. **`cross_*` 第一轮永远不报** —— 没有"上一轮"就没法判断穿越。这是对的，不是 bug。
2. **冷却按标的算** —— A 股票触发不会压住 B 股票。同一只票在冷却期内只报一次。
3. **冷却状态存盘** —— 重启进程不会重置，否则一天能收几百条重复。

---

## 三、推送到哪

| 渠道 | 说明 |
|---|---|
| `ConsoleChannel` | 终端彩色输出 |
| `FileChannel` | 追加写文件，按大小自动切分 |
| `WindowsToastChannel` | Windows 右下角气泡，不装第三方库 |
| `WebhookChannel` | POST JSON，支持 `generic` / `wecom` / `feishu` |
| `CallableChannel` | 包一个函数，接你自己的微信/钉钉/Telegram |

企业微信群机器人配置示例：

```json
"channels": {
  "wechat": {
    "factory": "reminder.channels:WebhookChannel",
    "args": {"url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx",
             "format": "wecom"}
  }
}
```

接自己的推送（比如 Hermes 的微信直投）：

```python
from reminder.channels import CallableChannel

def my_push(alert):
    requests.post("http://...", json={"text": alert.message})

channels = {"wechat": CallableChannel(my_push)}
```

**渠道挂了不影响别的渠道**，也不影响主循环 —— 只会打一行警告。

---

## 四、配置文件

| 键 | 默认 | 说明 |
|---|---|---|
| `poll_interval` | 30 | 轮询间隔（秒） |
| `rules_dir` | `rules` | 规则目录 |
| `state_file` | `state.json` | 冷却与穿越状态 |
| `alert_log` | `alerts.jsonl` | 告警历史 |
| `source` | — | 数据源工厂 |
| `channels` | console | 渠道表 |
| `severity_routes` | 全走 console | severity → 渠道名列表 |

路径都相对于配置文件所在目录。

---

## 五、目录结构

```
reminder/
  models.py      规则与校验（纯函数，不碰行情）
  conditions.py  条件求值、穿越、模板
  sources.py     数据源适配（← 别人接数据的地方）
  channels.py    推送渠道
  engine.py      主循环、冷却、状态持久化
  config.py      工厂加载
  cli.py         命令行
  rules/         规则示例（全部默认停用）
  examples/      自定义数据源模板
  demo.py        假数据演示
  selftest.py    42 项自检
```

---

## 六、为什么这样设计

| 决定 | 原因 |
|---|---|
| **一规则一个 JSON 文件** | 能手改、能进 git、坏一个不影响别的、冲突好解决 |
| **条件是 (field, op, value)** | 加新指标只是加一行配置，不用改代码 |
| **冷却落盘** | 重启不清零，否则每次重启都重复轰炸 |
| **冷却按"规则+标的"** | 一只票触发不该压住另一只 |
| **cross 需要两轮** | 穿越是个"变化量"，单点快照算不出来 |
| **取数失败只警告** | 网络抖一下不该把常驻进程打死 |
| **缺数据不写状态** | 否则一个字段暂时缺失，下次真穿越会被吞掉 |

规则模型的字段设计参考了开源项目 tick-stock-panel 的监控模块，
但**没有复制它的代码** —— 它的评估引擎是 83KB 单体，绑着整套后端，
这里是从零实现的 ~600 行独立版本。

---

## 七、跑自检

```bash
python -m reminder.selftest
```

42 项断言，覆盖条件运算、穿越语义、模板、校验、持久化、冷却、路由、配置工厂、回放。
