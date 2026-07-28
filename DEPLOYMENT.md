# 部署文档

接口契约与调用示例见 [API.md](API.md)；本文档只讲"怎么把服务跑起来"。

## 前置依赖

1. **Python 3.11+**，安装 `requirements.txt`（含 CUDA 12.8 版 PyTorch，安装顺序见 [README.md](README.md)）。
2. **Ollama** 已安装并拉取模型：
   ```powershell
   ollama pull qwen2.5:7b-instruct
   ```
3. **离线数据流水线已跑过一次**，即已存在：
   - ChromaDB 持久化目录（`pmc_document_splitter.py` + `pmc_vector_index.py` 的产出）
   - BM25 索引 pickle 缓存（首次启动若不存在会自动构建并写入缓存，之后复用）

   本仓库默认指向小规模 `test_dir_mode` 集合（1854 chunks），开箱即用；若要用全量 `pmc_full`，注意下方"资源约束"一节。

## 本地开发环境

```powershell
git clone <repo>
cd Rag-Med
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

copy .env.example .env
# 按需修改 .env（一般默认值即可跑通 test_dir_mode）

$env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` 仅用于开发（代码改动自动重启，会重新触发一次模型加载）。首次请求前先轮询 `GET /health` 确认 `ready: true`。

## 资源约束（务必先读）

本服务的"状态"（BGE 嵌入模型、ChromaDB 连接、BM25 索引、reranker 模型）挂在单个进程的 `app.state` 上，由 `lifespan` 在启动时构建一次：

- **不要用多 worker 启动**（`uvicorn ... --workers N`，N>1）。每个 worker 是独立进程，会各自重新加载一整套模型——N 个 worker 就是 N 倍的显存/内存占用，在共享 GPU 的机器上（本项目开发环境的 GPU 同时被桌面环境占用，可用显存本就有限）很容易 OOM。需要更高并发就纵向加资源，或部署多台独立实例由反向代理做负载均衡，而不是单机多 worker。
- 若把 `RAG_COLLECTION` 指向全量 `pmc_full` 集合：加载 HNSW 索引本身就要吃掉约 18-21GB 内存，叠加 BM25/embedder/reranker 后单进程内存占用会逼近系统上限。生产环境请先确认机器内存/显存余量，见 README"已知限制"一节的实测数据。
- Ollama 是独立于本服务的外部进程/服务，本身可以处理一定并发，但仍与 API 服务共享同一张 GPU——高并发场景建议压测确认吞吐后再上线。

## 生产环境运行建议

```powershell
$env:PYTHONUTF8="1"; uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

- 用 Nginx / Caddy 等反向代理做 TLS 终止、请求超时控制（LLM 生成耗时可达数十秒，反向代理的超时时间需相应放宽，建议 ≥120s）。
- 日志已按用途分文件落盘在 `logs/`：
  - `api_service.log` —— 服务级日志（启动/关闭、异常堆栈）
  - `api_requests.jsonl` —— 每个 HTTP 请求一行（request_id/path/status/耗时），可直接喂给日志采集系统
  - `api_generation.jsonl` —— 每次问答调用一行（阶段耗时、成功情况），供离线分析生成质量
  - 建议配合 `logrotate`（Linux）或 Windows 任务计划做日志轮转，这些文件会持续增长。
- 用 `GET /health` 做存活探针（liveness），`ready` 字段做就绪探针（readiness）——编排系统应等 `ready: true` 后再把流量切过来，启动阶段（模型加载中）不应算作"不健康"而被杀掉重启。
- 用 `GET /api/v1/stats` 接入监控面板：`qa.success_rate`/`qa.avg_latency_seconds` 适合做告警阈值；`components[].status != "ok"` 适合做组件级告警。

## Docker（可选，需自行验证 GPU 直通）

Ollama 与 GPU 直通的容器化配置高度依赖宿主机的 Docker 版本/驱动，这里只给 API 服务本身的 Dockerfile 作为起点，Ollama 建议仍运行在宿主机或单独容器里，通过 `RAG_LLM_BASE_URL` 指向它。

```dockerfile
# Dockerfile（API 服务，不含 Ollama）
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt

COPY api/ ./api/
COPY generation/ ./generation/
COPY retrieval/ ./retrieval/
COPY pmc_vector_index.py .

ENV PYTHONUTF8=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```yaml
# docker-compose.yml（示意；Ollama 假设已在宿主机运行）
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_LLM_BASE_URL=http://host.docker.internal:11434
      - RAG_DB_DIR=/data/chroma_db
      - RAG_BM25_CACHE=/data/bm25_index_test_dir_mode.pkl
    volumes:
      - ./pipeline_output/chroma_db:/data/chroma_db
      - ./pipeline_output/bm25_index_test_dir_mode.pkl:/data/bm25_index_test_dir_mode.pkl
```

reranker 模型若要用 GPU，需要给容器加 `--gpus all`（Linux + NVIDIA Container Toolkit）；Windows 下建议直接用上方"本地开发环境"方式运行，不经容器。

## 故障排查

| 现象 | 排查方向 |
|---|---|
| `/health` 一直 `ready: false` 且服务未崩溃 | 首次启动模型下载慢（BGE/reranker 从 HuggingFace Hub 拉取），检查网络；看 `logs/api_service.log` 的加载进度日志 |
| 问答接口返回 `5001 服务尚未就绪` | 同上，还未初始化完成，等待或检查启动日志报错 |
| 问答接口返回 `4001 模型调用失败` | Ollama 未启动/端口不对/模型未拉取，检查 `RAG_LLM_BASE_URL` 与 `ollama list` |
| `GET /api/v1/stats` 里 `components` 某项 `status: down` | 对应组件不可达，`detail` 字段有具体错误信息 |
| 内存占用异常高 / 启动缓慢 | 检查 `RAG_COLLECTION` 是否被设成了 `pmc_full`，见上方"资源约束" |
| 会话历史查询总是 404 | 服务重启会清空所有会话（进程内内存存储，非持久化），需要重新创建会话 |
