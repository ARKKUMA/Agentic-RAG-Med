"""
config.py — 环境变量配置管理
基于 pydantic-settings：自动从 .env 文件 + 真实环境变量读取配置，
带类型校验与默认值。真实环境变量优先级高于 .env 文件。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # RAG 数据源
    rag_db_dir: str = r"d:\Rag-Med\pipeline_output\chroma_db"
    rag_collection: str = "test_dir_mode"
    rag_bm25_cache: str = r"d:\Rag-Med\pipeline_output\bm25_index_test_dir_mode.pkl"
    rag_llm_model: str = "qwen2.5:7b-instruct"
    rag_llm_base_url: str = "http://localhost:11434"

    # 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 会话
    session_ttl_seconds: int = 3600
    session_max_turns: int = 20


settings = Settings()
