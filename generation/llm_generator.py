"""
llm_generator.py — 本地 LLM 集成与生成（Ollama）
封装 Ollama HTTP API：连接检测、单条生成（含 JSON 格式约束与容错提取、可选缓存）、批量生成。
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from .cache import GenerationCache

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_TIMEOUT = 120


class LLMGenerator:
    """
    Ollama 本地 LLM 生成器封装。

    Args:
        model_name: Ollama 模型名称
        base_url: Ollama 服务地址
        timeout: 请求超时时间（秒）
        cache: 可选的 GenerationCache 实例；传入后对低温（确定性）调用自动
               读写缓存，高温调用直接跳过缓存（详见 cache.py 模块说明）
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        log: logging.Logger | None = None,
        cache: GenerationCache | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log = log or logging.getLogger("llm_generator")
        self.cache = cache
        self._check_connection()

    # ── 初始化并测试连接 ──────────────────────────────────────────
    def _check_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"无法连接到 Ollama 服务 {self.base_url}：{e}") from e

        models = [m.get("name", "") for m in resp.json().get("models", [])]
        self.log.info(f"Ollama 连接成功，已拉取模型：{models}")
        if self.model_name not in models and not any(m.startswith(self.model_name.split(":")[0]) for m in models):
            self.log.warning(
                f"模型 {self.model_name!r} 不在已拉取列表中，调用时 Ollama 可能报错，"
                f"请先执行 `ollama pull {self.model_name}`"
            )

    # ── JSON 提取与容错修复 ───────────────────────────────────────
    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        从模型输出中提取 JSON 对象；容忍前后多余文字、markdown 代码块包裹，
        以及常见格式错误（多余尾随逗号、缺失收尾括号）。
        """
        if not text:
            return None
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 常见修复：去除多余尾随逗号 + 补齐缺失的收尾括号
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        fixed += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    # ── 单条生成 ──────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        require_json: bool = False,
    ) -> dict:
        """
        调用 Ollama /api/generate 生成文本；若配置了 cache 且 temperature 足够低，
        优先查缓存命中（结果附加 "cached": True），否则调用模型并按规则写入缓存。

        Returns:
            {"text": str, "elapsed_seconds": float, "model": str, "done": bool,
             "cached": bool, "json": dict | None}
             （json 字段仅在 require_json=True 时存在）
        """
        use_cache = self.cache is not None and self.cache.is_cacheable_temperature(temperature)
        cache_key = None
        if use_cache:
            cache_key = self.cache.make_key(
                model=self.model_name,
                system=system_prompt or "",
                prompt=prompt,
                temperature=str(temperature),
                json_mode=str(require_json),
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = dict(cached)
                result["cached"] = True
                return result

        final_prompt = prompt
        if require_json:
            # Ollama 的 format="json" 约束输出必须是合法 JSON，但部分模型仍需
            # 在 prompt 中显式重申要求才能稳定遵循 schema
            final_prompt = prompt + "\n\n请仅返回合法 JSON 对象，不要添加任何额外说明文字。"

        payload: dict = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if require_json:
            payload["format"] = "json"

        t0 = time.time()
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - t0

        raw_text = data.get("response", "")
        result = {
            "text": raw_text,
            "elapsed_seconds": round(elapsed, 2),
            "model": self.model_name,
            "done": data.get("done", True),
            "cached": False,
            # Ollama /api/generate 在 done=True 时附带的真实 token 计数
            # （非估算），用于 execution_trace 里的"生成 Token 数"字段。
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }
        if require_json:
            result["json"] = self._extract_json(raw_text)

        if use_cache:
            self.cache.set(cache_key, result, temperature=temperature)
        return result

    # ── 流式生成 ──────────────────────────────────────────────────
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 900,
    ):
        """
        调用 Ollama /api/generate（stream=True），逐 token yield 文本片段。
        不接入缓存——流式场景本身就是为了实时展示生成过程，命中缓存会导致
        "整段文字瞬间出现"而非逐字流出，违背流式接口的初衷；且流式通常用于
        高温（创造性）的 answer_generator 阶段，本就不满足缓存的温度门控条件。
        """
        payload: dict = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt

        with requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    # ── 批量生成 ──────────────────────────────────────────────────
    def generate_batch(self, requests_list: list[dict]) -> list[dict]:
        """
        依次执行多条生成请求（本地单模型服务通常串行处理，不做并发）。

        Args:
            requests_list: [{"prompt":.., "system_prompt":.., "temperature":..,
                              "max_tokens":.., "require_json":..}, ...]
        """
        results = []
        for i, req in enumerate(requests_list):
            self.log.info(f"批量生成 [{i + 1}/{len(requests_list)}]…")
            results.append(self.generate(**req))
        return results
