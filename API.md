# PMC 医学文献 RAG API 调用文档

基于 FastAPI 的检索增强生成（RAG）问答服务。本文档面向"调用方"，只描述接口契约与用法；服务内部实现见 [README.md](README.md)。

## 启动服务

```powershell
$env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000
```

首次启动需加载 BGE 嵌入模型、ChromaDB 索引、重排序模型并连接 Ollama，耗时约 8-15 秒；启动完成前 `/health` 会返回 `ready: false`。

可选环境变量（均有默认值，默认指向小规模 `test_dir_mode` 测试集合）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_DB_DIR` | `d:\Rag-Med\pipeline_output\chroma_db` | ChromaDB 持久化目录 |
| `RAG_COLLECTION` | `test_dir_mode` | 检索用的 ChromaDB 集合名 |
| `RAG_BM25_CACHE` | `.../bm25_index_test_dir_mode.pkl` | BM25 索引 pickle 缓存路径 |
| `RAG_LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama 模型名称 |

**注意：** 若把 `RAG_COLLECTION` 指向全量 `pmc_full` 集合，加载 HNSW 索引会占用约 18-21GB 内存，请确认机器内存充足（详见 README"已知限制"）。

## 通用约定

### 统一响应格式

所有接口（除 SSE 流式接口外）返回统一 JSON 结构：

```json
{
  "code": 0,
  "message": "success",
  "data": { ... },
  "request_id": "c332253ea1384fe5a11516f92a9d1354",
  "timestamp": "2026-07-27 10:58:35"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | int | `0` 表示成功；非 0 见下方错误码表 |
| `message` | string | 成功时固定为 `"success"`；失败时为具体错误描述 |
| `data` | object \| null | 成功时为接口数据；失败时为 `null` |
| `request_id` | string \| null | 请求唯一 ID，同时也出现在响应头 `X-Request-ID` 中，用于日志排查 |
| `timestamp` | string | 服务器处理时间 |

每个响应还带有两个诊断用响应头：`X-Request-ID`、`X-Elapsed-Seconds`（本次请求耗时）。

### 错误码

| code | HTTP 状态 | 含义 |
|---|---|---|
| 1001 | 422 | 请求参数错误（如 `query` 为空、`top_k` 超出范围） |
| 1002 | 422 | 查询内容过长 |
| 1003 | 422 | `top_k` 超出允许范围 |
| 2001 | 401 | 认证失败 |
| 2002 | 401 | 认证已过期 |
| 3001 | 404 | 文档不存在 |
| 3002 | 404 | 会话不存在或已过期 |
| 4001 | 502 | 模型调用失败（如 Ollama 不可达/超时） |
| 4002 | 502 | 检索服务调用失败 |
| 5000 | 500 | 服务内部错误（兜底，理论上不应出现未分类异常） |
| 5001 | 503 | 服务尚未就绪（RAG 流水线仍在加载中） |

错误响应示例：

```json
{
  "code": 1001,
  "message": "top_k: Input should be less than or equal to 20",
  "data": null,
  "request_id": "ff179ea67e514c5481c606bf88175160",
  "timestamp": "2026-07-27 12:02:41"
}
```

---

## GET /health

健康检查。`ready=false` 表示流水线仍在加载，此时调用问答接口会返回 `5001 服务尚未就绪`。

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "code": 0,
  "message": "success",
  "data": {"status": "ok", "ready": true},
  "request_id": null,
  "timestamp": "2026-07-27 10:58:35"
}
```

---

## POST /api/v1/qa — 同步问答

一次性返回完整答案（检索 → 证据评估 → 答案生成 → 批判性审查 → 最终组装 全部执行完毕后才返回）。

### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | 是 | — | 用户问题，1-2000 字符 |
| `top_k` | int | 否 | 8 | 最终返回结果数量，范围 1-20 |
| `fusion_strategy` | `"rrf"` \| `"weighted"` \| `"simple"` | 否 | `"rrf"` | 多路检索融合策略 |
| `session_id` | string \| null | 否 | null | 传入后关联历史对话（见下方"会话管理"） |
| `run_evaluation` | bool | 否 | true | 是否执行证据评估阶段 |
| `run_review` | bool | 否 | true | 是否执行批判性审查阶段 |

### 示例

```bash
curl -X POST http://127.0.0.1:8000/api/v1/qa \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What genes have been associated with susceptibility to type 2 diabetes?",
        "top_k": 6
      }'
```

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/api/v1/qa",
    json={"query": "What genes have been associated with susceptibility to type 2 diabetes?", "top_k": 6},
    timeout=60,
)
data = resp.json()["data"]
print(data["answer"])
```

### 响应 `data` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `answer` | string | 最终答案（含 `## Core Answer` / `## Evidence Summary` / `**References:**` 结构化章节与免责声明） |
| `sources` | array | 引用来源列表，每项含 `rank`（对应正文 `[来源 N]` 编号）、`chunk_id`、`journal`、`pub_year`、`pmc_id`、`relevance_score` |
| `session_id` | string \| null | 回显传入的 `session_id` |
| `total_time_seconds` | float | 本次生成总耗时 |
| `citation_retry_attempts` | int | 引用校验触发的重试次数（0 表示首次生成即通过校验） |
| `format_check_pass` | bool | 是否同时满足章节标题、缩写全称、参考文献完整性三项格式规范 |

---

## POST /api/v1/qa/stream — 流式问答（SSE）

检索与证据评估同步完成后，答案生成阶段逐 token 以 Server-Sent Events 推送；审查与最终组装仍在服务端同步完成，最后推送包含完整结果的 `done` 事件。**请求体字段与同步接口完全一致。**

流式路径不做引用重试循环（重试需整段重新生成，与"边生成边显示"冲突），引用正确性由末尾的兜底校验保证；因此 `citation_retry_attempts` 恒为 0。

### 事件格式

响应 `Content-Type: text/event-stream`，每行 `data: {JSON}\n\n`：

| event | 字段 | 说明 |
|---|---|---|
| `status` | `stage` | 当前阶段：`retrieving` / `evaluating_evidence` / `generating` / `reviewing` / `finalizing` |
| `token` | `text` | 单个文本片段，按顺序拼接即为答案草稿 |
| `done` | `data` | 与同步接口 `data` 字段结构完全一致的最终结果 |
| `error` | `message` | 流水线内部异常时推送，之后流结束 |

### 示例

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/qa/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What statistical methods are used to analyze survival data in cancer studies?", "top_k": 5}'
```

```python
import json
import requests

with requests.post(
    "http://127.0.0.1:8000/api/v1/qa/stream",
    json={"query": "What statistical methods are used to analyze survival data in cancer studies?"},
    stream=True,
    timeout=60,
) as resp:
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if event["event"] == "token":
            print(event["text"], end="", flush=True)
        elif event["event"] == "status":
            print(f"\n[{event['stage']}]", flush=True)
        elif event["event"] == "done":
            print("\n\n最终结果:", event["data"])
```

---

## GET /api/v1/qa/sessions/{session_id} — 查看会话历史

用于调试/前端展示当前会话已保存的历史轮次（调用方一般不需要手动请求此接口，仅用于排查会话状态）。

```bash
curl http://127.0.0.1:8000/api/v1/qa/sessions/test-session-001
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "session_id": "test-session-001",
    "turns": [
      {"query": "What does GWAS mean?", "answer": "## Core Answer\n...", "timestamp": 1785150187.74}
    ]
  },
  "request_id": "...",
  "timestamp": "2026-07-27 12:03:07"
}
```

会话不存在或已过期（默认 1 小时无活动过期）时返回 `3002`（HTTP 404）。

---

## 会话管理说明

1. 首次提问时自行生成一个 `session_id`（任意唯一字符串即可，服务端不做格式校验）并传入请求体。
2. 后续追问传入相同 `session_id`，服务会自动把最近 3 轮对话渲染成简短上下文，**仅用于消解指代**（如"它"、"那个药"指代上一轮提到的实体），**不会**作为检索依据或事实来源——每轮提问的检索仍然只使用当前这一句 `query`。
3. 会话状态保存在服务进程内存中，服务重启即丢失；单会话最多保留 20 轮，超出自动淘汰最旧的。
4. `session_id` 不存在或已过期时，同步/流式问答接口**不会报错**，会按新会话继续处理（只有 `GET /sessions/{id}` 查询历史时才会返回 404）。

## 完整调用示例（多轮对话）

```python
import requests

BASE = "http://127.0.0.1:8000"
session_id = "demo-session-001"

def ask(query):
    resp = requests.post(f"{BASE}/api/v1/qa", json={"query": query, "session_id": session_id}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if body["code"] != 0:
        raise RuntimeError(f"[{body['code']}] {body['message']}")
    return body["data"]["answer"]

print(ask("What does GWAS mean?"))
print(ask("Is it used in the diabetes gene studies you have access to?"))  # "it" 会被正确解析为 GWAS
```
