# PMC 医学文献 RAG API 调用文档

基于 FastAPI 的检索增强生成（RAG）问答服务。本文档面向"调用方"，只描述接口契约与用法；服务内部实现见 [README.md](README.md)。

## 启动服务

```powershell
$env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000
```

首次启动需加载 BGE 嵌入模型、ChromaDB 索引、重排序模型、连接 Ollama，并扫描一次集合构建文档级索引，耗时约 8-15 秒；启动完成前 `/health` 会返回 `ready: false`。

### 环境变量配置（.env）

复制 [.env.example](.env.example) 为 `.env` 并按需修改（`.env` 已加入 `.gitignore`，不会被提交）。配置通过 `api/config.py` 的 `pydantic-settings` 读取，真实环境变量优先级高于 `.env` 文件。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_DB_DIR` | `d:\Rag-Med\pipeline_output\chroma_db` | ChromaDB 持久化目录 |
| `RAG_COLLECTION` | `test_dir_mode` | 检索用的 ChromaDB 集合名 |
| `RAG_BM25_CACHE` | `.../bm25_index_test_dir_mode.pkl` | BM25 索引 pickle 缓存路径 |
| `RAG_LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama 模型名称 |
| `RAG_LLM_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `API_HOST` | `0.0.0.0` | 服务监听地址 |
| `API_PORT` | `8000` | 服务监听端口 |
| `SESSION_TTL_SECONDS` | `3600` | 会话过期时间（秒） |
| `SESSION_MAX_TURNS` | `20` | 单会话最多保留对话轮数 |

**注意：** 若把 `RAG_COLLECTION` 指向全量 `pmc_full` 集合，加载 HNSW 索引会占用约 18-21GB 内存，请确认机器内存充足（详见 README"已知限制"）。

### 交互式文档 / OpenAPI

服务启动后自动提供：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- OpenAPI schema：`http://127.0.0.1:8000/openapi.json`（项目根目录的 [openapi.json](openapi.json) 是同一份 schema 的静态快照）

也可以直接导入 [postman_collection.json](postman_collection.json) 到 Postman，覆盖本文档下方全部端点，并带有基础断言（`pm.test`）——先运行 "Sessions > Create Session" 会自动把返回的 `session_id` 存进 collection variable，供后续请求复用。

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

## 会话管理接口

会话状态保存在服务进程内存中（服务重启即丢失），单会话最多保留 20 轮，默认 1 小时无活动即过期。"添加消息"没有独立接口——由 `POST /api/v1/qa`（同步/流式均可）在生成完成后自动调用，不需要调用方手动维护。

### POST /api/v1/sessions — 创建会话

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
```

```json
{
  "code": 0,
  "message": "success",
  "data": {"session_id": "3ea10b371dcc459ca59dba038a893ccf", "created_at": 1785233756.53},
  "request_id": "...",
  "timestamp": "2026-07-28 11:15:56"
}
```

调用方也可以不预先创建，直接在 `POST /api/v1/qa` 里传入任意自选的 `session_id` 字符串——服务端会在第一次问答时自动建立该会话（见下方"会话管理说明"）。

### GET /api/v1/sessions/{session_id} — 获取会话信息（历史消息列表）

```bash
curl http://127.0.0.1:8000/api/v1/sessions/3ea10b371dcc459ca59dba038a893ccf
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "session_id": "3ea10b371dcc459ca59dba038a893ccf",
    "created_at": 1785233756.53,
    "last_active": 1785233970.26,
    "turn_count": 1,
    "turns": [
      {"query": "What does GWAS mean?", "answer": "## Core Answer\n...", "timestamp": 1785233970.26}
    ]
  },
  "request_id": "...",
  "timestamp": "2026-07-28 11:19:13"
}
```

会话不存在或已过期时返回 `3002`（HTTP 404）。

### DELETE /api/v1/sessions/{session_id} — 删除会话

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/sessions/3ea10b371dcc459ca59dba038a893ccf
```

```json
{"code": 0, "message": "success", "data": {"session_id": "...", "deleted": true}, "request_id": "...", "timestamp": "..."}
```

会话不存在时同样返回 `3002`（HTTP 404），删除操作是幂等的（重复删除同一个已不存在的会话，两次都会得到 404，而不是第二次"假装成功"）。

### 会话管理说明

1. 首次提问时可以先调用 `POST /api/v1/sessions` 拿到 `session_id`，也可以自己生成任意唯一字符串直接使用（服务端不做格式校验）。
2. 后续追问传入相同 `session_id`，服务会自动把最近 3 轮对话渲染成简短上下文，**仅用于消解指代**（如"它"、"那个药"指代上一轮提到的实体），**不会**作为检索依据或事实来源——每轮提问的检索仍然只使用当前这一句 `query`。
3. `session_id` 不存在或已过期时，同步/流式问答接口**不会报错**，会按新会话继续处理（只有 `GET`/`DELETE /sessions/{id}` 才会返回 404）。

---

## GET /api/v1/stats — 运营统计

问答调用次数/平均耗时/成功率、语料规模、各组件（LLM / 向量库 / "数据库"）健康状态。适合监控面板轮询。

```bash
curl http://127.0.0.1:8000/api/v1/stats
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "qa": {
      "total_calls": 12,
      "success_count": 11,
      "failure_count": 1,
      "success_rate": 0.9167,
      "avg_latency_seconds": 14.32
    },
    "corpus": {
      "total_documents": 99,
      "total_chunks": 1854,
      "index_size_bytes": null,
      "incremental_update_count": 0
    },
    "components": [
      {"name": "llm", "status": "ok", "detail": "Ollama 可达，已拉取模型数=1", "latency_seconds": 0.02},
      {"name": "vector_store", "status": "ok", "detail": "ChromaDB 集合 test_dir_mode 向量数=1,854", "latency_seconds": 0.001},
      {"name": "database", "status": "ok", "detail": "BM25 关键词索引文档数=1,854（会话存储为进程内内存，非持久化数据库）", "latency_seconds": 0.0}
    ],
    "active_sessions": 2
  },
  "request_id": "...",
  "timestamp": "..."
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `qa.*` | `POST /api/v1/qa`（同步+流式）调用计数，服务启动后累计，重启清零 |
| `corpus.total_documents` / `total_chunks` | 当前集合的文档数/chunk 数（启动时扫描一次得到） |
| `corpus.incremental_update_count` | **预留字段，当前恒为 0**——本版本文档接口只读，尚无写入/更新入口驱动此计数 |
| `components` | `llm`（Ollama 连通性）、`vector_store`（ChromaDB）、`database`（本系统用 BM25 关键词索引承担该角色；没有独立关系型数据库，会话数据是进程内内存，非持久化） |
| `components[].status` | `ok` / `degraded`（可用但异常，如索引为空）/ `down`（不可达） |

---

## 文档管理接口（只读）

### GET /api/v1/documents — 文档列表查询

按 `doc_id` 聚合的文档级视图（分页），标题/摘要/期刊/年份等元数据取自该文档的摘要类 chunk。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `page` | int | 1 | 页码，从 1 开始 |
| `page_size` | int | 20 | 每页数量，1-100 |

```bash
curl "http://127.0.0.1:8000/api/v1/documents?page=1&page_size=10"
```

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "doc_id": "PMC176545",
        "title": "The Transcriptome of the Intraerythrocytic Developmental Cycle of Plasmodium falciparum",
        "abstract": "...",
        "journal": "PLoS Biology",
        "pub_date": "2003",
        "pmid": "12929205",
        "doi": "10.1371/journal.pbio.0000005",
        "article_type": "research-article",
        "chunk_count": 58
      }
    ],
    "page_info": {"page": 1, "page_size": 10, "total": 99, "total_pages": 10}
  },
  "request_id": "...",
  "timestamp": "..."
}
```

### GET /api/v1/documents/{doc_id} — 按 ID 查询单篇文档

```bash
curl http://127.0.0.1:8000/api/v1/documents/PMC176545
```

响应 `data` 结构与列表接口的单个 `items` 元素一致。文档不存在时返回 `3001`（HTTP 404）。

### DocumentIn（写入模型，预留）

`api/models.py` 中定义了 `DocumentIn`（`doc_id` / `title` / `abstract` / `journal` / `pub_date` / `pmid` / `doi` / `article_type`），供未来文档录入/更新接口使用。**当前版本尚未开放写入接口**，文档数据只能通过重新运行 `pmc_document_splitter.py` + `pmc_vector_index.py` 离线流水线写入 ChromaDB，再重启服务触发文档索引重建。

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

---

## 测试

`tests/test_api.py` 是基于标准库 `unittest` + FastAPI `TestClient` 的单元/集成测试，覆盖本文档列出的全部端点（含参数校验、错误路径、会话完整生命周期、真实调用生成流水线的 smoke test）：

```powershell
$env:PYTHONUTF8="1"; python -m unittest tests.test_api -v
```

RAG 流水线通过 `setUpModule`/`tearDownModule` 在整个测试文件范围内只加载一次；真正触发 LLM 生成的用例数量刻意控制得很少（同步/流式各一个端到端用例 + 会话生命周期里的一次问答），其余校验/错误路径测试在到达生成阶段前就被拒绝，不产生实际 LLM 调用。完整运行一次约 40-50 秒（含约 7-10 秒的一次性模型加载）。

也可以导入 [postman_collection.json](postman_collection.json) 用 Postman 手动/半自动测试，见上文"交互式文档 / OpenAPI"一节。

部署相关内容见 [DEPLOYMENT.md](DEPLOYMENT.md)。
