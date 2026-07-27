"""
pipeline.py — 医学生成流水线（MedicalGenerationPipeline）
检索 -> 上下文组装 -> 证据评估（可选）-> 答案草稿 -> 批判性审查（可选）->
最终答案组装 -> 生成后处理（引用/格式/免责声明）
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from .citation_validator import CitationValidator
from .context_assembler import ContextAssembler, DocumentChunk
from .format_checker import FormatChecker
from .llm_generator import LLMGenerator
from .prompt_templates import (
    DISCLAIMERS,
    LANGUAGE_DIRECTIVES,
    MEDICAL_PROMPT_STAGES,
    REFERENCES_HEADER,
    PromptStage,
    build_hard_constraints,
    detect_query_language,
)

MAX_CITATION_RETRIES = 2  # 引用校验失败时的最大重试次数，超过后走兜底修正（剔除无效引用）


class MedicalGenerationPipeline:
    """
    端到端医学问答生成流水线，串联 RetrievalPipeline 与四阶段提示词工程：
    证据评估 -> 答案生成 -> 批判性审查 -> 最终组装。
    """

    def __init__(
        self,
        retrieval_pipeline,
        llm: LLMGenerator,
        context_assembler: ContextAssembler | None = None,
        prompt_stages: dict[str, PromptStage] | None = None,
        log: logging.Logger | None = None,
        log_path: str | Path | None = None,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.llm = llm
        self.context_assembler = context_assembler or ContextAssembler()
        self.stages = prompt_stages or MEDICAL_PROMPT_STAGES
        self.log = log or logging.getLogger("medical_generation_pipeline")
        self.citation_validator = CitationValidator()
        self.format_checker = FormatChecker(log=self.log)

        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # BatchGenerationProcessor 会从多个线程并发调用 generate() -> _log_run()，
        # 未加锁的并发 append 写可能在文件层面交错出错乱的 JSONL 行
        self._log_lock = threading.Lock()

    # ── 提示词填充 + 单阶段调用 ───────────────────────────────────
    def _run_stage(
        self,
        stage_key: str,
        require_json: bool,
        extra_system_prompt: str = "",
        **template_vars,
    ) -> dict:
        """
        extra_system_prompt：多层次 system_prompt 设计的"第 0 层"——若提供，
        前置拼接到该阶段自身的任务专属 system_prompt（"第 1 层"）之前。
        目前用于给 answer_generator/final_assembler 注入 build_hard_constraints()
        渲染出的强约束指令块。
        """
        stage = self.stages[stage_key]
        prompt = stage.user_prompt_template.format(**template_vars)
        system_prompt = (
            f"{extra_system_prompt}\n\n{stage.system_prompt}" if extra_system_prompt else stage.system_prompt
        )
        return self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=stage.temperature,
            max_tokens=stage.max_tokens,
            require_json=require_json,
        )

    # ── 答案生成 + 引用校验重试循环 ───────────────────────────────
    def _generate_grounded_answer(
        self,
        query: str,
        context_text: str,
        valid_ids: set[int],
        language_directive: str,
        hard_constraints: str,
        conversation_context: str = "",
    ) -> tuple[str, dict, int]:
        """
        调用 answer_generator 生成草稿，并校验其 [来源 N] 引用是否都落在
        valid_ids 范围内。校验失败时，把具体问题反馈给模型重试（最多
        MAX_CITATION_RETRIES 次）；重试次数用尽仍未通过则兜底剔除无效引用标记，
        保证绝不会有虚假来源编号流出到最终答案。

        Returns:
            (draft_answer, last_validation, attempts_used)
        """
        retry_note = ""
        draft_answer = ""
        validation: dict = {}

        for attempt in range(MAX_CITATION_RETRIES + 1):
            draft_out = self._run_stage(
                "answer_generator", require_json=False,
                extra_system_prompt=hard_constraints,
                query=query, context=context_text or "（未检索到相关文献）",
                language_directive=language_directive,
                retry_note=retry_note,
                conversation_context=conversation_context,
            )
            draft_answer = draft_out["text"].strip()
            validation = self.citation_validator.validate(draft_answer, valid_ids)
            if validation["pass"]:
                return draft_answer, validation, attempt

            self.log.warning(f"引用校验未通过（第 {attempt + 1} 次尝试）：{validation}")
            retry_note = self.citation_validator.build_retry_instruction(validation)

        # 重试次数用尽仍未通过 -> 兜底修正：直接剔除无效引用标记
        fixed = self.citation_validator.strip_invalid_citations(draft_answer, valid_ids)
        return fixed, validation, MAX_CITATION_RETRIES + 1

    # ── 基于评估结果筛选上下文（剔除被判定不相关的来源）──────────
    @staticmethod
    def _filter_context_by_evaluation(
        context_text: str,
        selected_chunks: list[DocumentChunk],
        evaluation: dict | None,
    ) -> tuple[str, list[DocumentChunk]]:
        if not evaluation:
            return context_text, selected_chunks

        irrelevant_ids = set(evaluation.get("irrelevant_source_ids") or [])
        if not irrelevant_ids:
            return context_text, selected_chunks

        kept_chunks = [
            c for i, c in enumerate(selected_chunks, start=1)
            if i not in irrelevant_ids
        ]
        if not kept_chunks:
            # 评估器把所有来源都判为不相关时保底不清空上下文，避免答案生成阶段无米下炊
            return context_text, selected_chunks

        # 按 "[来源 N | ...]" 标题行重新切分原始 context_text，仅保留被保留来源对应的段落
        blocks = re.split(r"(?=\[来源 \d+ \|)", context_text)
        kept_indices = {i for i in range(1, len(selected_chunks) + 1) if i not in irrelevant_ids}
        filtered_blocks = [
            b for b in blocks
            if b.strip() and any(f"[来源 {i} |" in b for i in kept_indices)
        ]
        filtered_text = "\n".join(filtered_blocks).strip()
        return filtered_text, kept_chunks

    # ── 生成后处理：引用标记 + 格式美化 + 免责声明 ────────────────
    @staticmethod
    def _postprocess(answer: str, selected_chunks: list[DocumentChunk], language: str = "zh") -> str:
        """
        补全参考文献列表。引用编号必须使用 ContextAssembler 分配的原始 _citation_id
        （而非当前列表位置）——证据评估阶段可能已剔除部分来源，若按位置重新编号，
        会与正文中模型引用的 [来源 N] 编号错位。

        参考文献标题与免责声明按 language（"zh"/"en"）本地化，避免固定中文样板
        文字混入英文回答，稀释 ROUGE 等文本相似度评估。
        """
        if not answer:
            return answer
        text = answer.strip()
        if selected_chunks:
            refs = [REFERENCES_HEADER.get(language, REFERENCES_HEADER["zh"])]
            for i, c in enumerate(selected_chunks, start=1):
                citation_id = c.metadata.get("_citation_id", i)
                title = c.metadata.get("source_title") or "Untitled"
                journal = c.metadata.get("journal", "?")
                year = c.metadata.get("pub_year", "?")
                pmc_id = c.metadata.get("pmc_id", "?")  # 已含 "PMC" 前缀，不再重复拼接
                # 格式须与 format_checker._REF_ENTRY_RE 保持一致（标题/期刊/年份齐全，用 "—" 分隔）
                refs.append(f"[{citation_id}] {title} — {journal} ({year}), {pmc_id}")
            text = text + "\n".join(refs)
        return text + DISCLAIMERS.get(language, DISCLAIMERS["zh"])

    @staticmethod
    def _format_sources(selected_chunks: list[DocumentChunk]) -> list[dict]:
        return [
            {
                "rank": c.metadata.get("_citation_id", i),
                "chunk_id": c.chunk_id,
                "journal": c.metadata.get("journal"),
                "pub_year": c.metadata.get("pub_year"),
                "pmc_id": c.metadata.get("pmc_id"),
                "relevance_score": c.relevance_score,
            }
            for i, c in enumerate(selected_chunks, start=1)
        ]

    # ── JSONL 运行日志 ────────────────────────────────────────────
    def _log_run(self, result: dict) -> None:
        if not self.log_path:
            return
        record = {
            "timestamp": result["timestamp"],
            "query": result["query"],
            "answer_length": len(result["answer"]),
            "total_time_seconds": result["generation_metrics"]["total_time_seconds"],
            "stage_times": result["generation_metrics"]["stage_times"],
            "stage_success": result["generation_metrics"]["stage_success"],
            "n_sources": len(result["sources"]),
        }
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 主入口 ────────────────────────────────────────────────────
    def generate(
        self,
        query: str,
        top_k: int = 8,
        fusion_strategy: str = "rrf",
        run_evaluation: bool = True,
        run_review: bool = True,
        conversation_context: str = "",
    ) -> dict:
        """
        执行完整生成流水线：检索 -> 上下文组装 -> 证据评估(可选) -> 答案草稿 ->
        批判性审查(可选) -> 最终组装 -> 后处理。

        Args:
            conversation_context: 可选的多轮对话历史前缀（由 api/session.py 的
                SessionManager.build_context_prefix() 渲染），只注入答案生成阶段
                用于消解指代（"它"/"那个药"），不参与检索、不作为事实依据。

        Returns:
            见模块顶部注释描述的 result 结构（query/answer/context_metadata/
            generation_metrics/intermediate_results/sources/timestamp）。
        """
        t_start = time.time()
        stage_times: dict[str, float] = {}
        stage_success: dict[str, bool] = {}
        token_counts: dict[str, int] = {}
        # 显式检测查询语言并生成强制指令，而不是仅依赖 system_prompt 里的软性规则——
        # 本地模型（尤其中文背景较重的）经常在自由文本生成时默认切回中文
        query_language = detect_query_language(query)
        language_directive = LANGUAGE_DIRECTIVES[query_language]
        hard_constraints = build_hard_constraints(query_language)

        # 1. 检索
        t0 = time.time()
        retrieval_out = self.retrieval_pipeline.retrieve(query, top_k=top_k, fusion_strategy=fusion_strategy)
        stage_times["retrieval"] = round(time.time() - t0, 2)

        # 2. 上下文组装
        t0 = time.time()
        context_result = self.context_assembler.assemble(retrieval_out["results"], query=query)
        stage_times["context_assembly"] = round(time.time() - t0, 2)
        token_counts["context"] = context_result["metadata"]["estimated_tokens"]

        context_text = context_result["context_text"]
        selected_chunks = context_result["selected_chunks"]

        # 3. 证据评估（可选）
        evaluation_json = None
        if run_evaluation and context_text:
            t0 = time.time()
            try:
                eval_out = self._run_stage(
                    "evidence_evaluator", require_json=True,
                    query=query, context=context_text,
                )
                evaluation_json = eval_out.get("json")
                stage_success["evidence_evaluation"] = evaluation_json is not None
                token_counts["evidence_evaluation"] = self.context_assembler.estimate_tokens(eval_out["text"])
            except Exception as e:
                self.log.warning(f"证据评估阶段失败：{e}")
                stage_success["evidence_evaluation"] = False
            stage_times["evidence_evaluation"] = round(time.time() - t0, 2)

            context_text, selected_chunks = self._filter_context_by_evaluation(
                context_text, selected_chunks, evaluation_json
            )

        # 4. 答案草稿生成（含引用校验重试循环）
        t0 = time.time()
        draft_answer = ""
        citation_validation: dict = {}
        citation_retry_attempts = 0
        valid_citation_ids = {
            c.metadata.get("_citation_id") for c in selected_chunks
            if c.metadata.get("_citation_id") is not None
        }
        try:
            draft_answer, citation_validation, citation_retry_attempts = self._generate_grounded_answer(
                query=query,
                context_text=context_text,
                valid_ids=valid_citation_ids,
                language_directive=language_directive,
                hard_constraints=hard_constraints,
                conversation_context=conversation_context,
            )
            stage_success["answer_generation"] = bool(draft_answer)
            token_counts["draft_answer"] = self.context_assembler.estimate_tokens(draft_answer)
        except Exception as e:
            self.log.error(f"答案草稿生成失败：{e}")
            stage_success["answer_generation"] = False
        stage_times["answer_generation"] = round(time.time() - t0, 2)

        # 5. 批判性审查（可选）
        review_json = None
        if run_review and draft_answer:
            t0 = time.time()
            try:
                review_out = self._run_stage(
                    "critical_reviewer", require_json=True,
                    query=query, context=context_text, draft_answer=draft_answer,
                )
                review_json = review_out.get("json")
                stage_success["critical_review"] = review_json is not None
                token_counts["critical_review"] = self.context_assembler.estimate_tokens(review_out["text"])
            except Exception as e:
                self.log.warning(f"批判性审查阶段失败：{e}")
                stage_success["critical_review"] = False
            stage_times["critical_review"] = round(time.time() - t0, 2)

        # 6. 最终答案组装：审查通过 or 无审查 -> 直接用草稿；判定 revise -> 重新组装
        t0 = time.time()
        final_answer = draft_answer
        if review_json:
            if review_json.get("overall_verdict") == "revise":
                try:
                    final_out = self._run_stage(
                        "final_assembler", require_json=False,
                        extra_system_prompt=hard_constraints,
                        query=query, draft_answer=draft_answer,
                        review_feedback=json.dumps(review_json, ensure_ascii=False),
                        language_directive=language_directive,
                    )
                    final_answer = final_out["text"].strip() or draft_answer
                    stage_success["final_assembly"] = True
                except Exception as e:
                    self.log.warning(f"最终组装阶段失败，回退使用草稿：{e}")
                    stage_success["final_assembly"] = False
            else:
                stage_success["final_assembly"] = True  # 审查通过，直接采用草稿
        else:
            stage_success["final_assembly"] = False  # 没有审查结果，直接使用草稿
        stage_times["final_assembly"] = round(time.time() - t0, 2)
        token_counts["final_answer"] = self.context_assembler.estimate_tokens(final_answer)

        # 7. 最终安全网：无论最终答案来自草稿还是 final_assembler 重写，都再校验一次
        # 引用有效性——final_assembler 在"修复"审查意见时有可能引入新的无效引用，
        # 这一步兜底剔除任何残留的虚假来源编号，确保绝不外泄
        final_answer = self.citation_validator.strip_invalid_citations(final_answer, valid_citation_ids)

        # 8. 生成后处理
        final_answer = self._postprocess(final_answer, selected_chunks, language=query_language)

        # 9. 格式合规检查（章节标题 / 缩写全称 / 参考文献完整性）
        format_check = self.format_checker.check(final_answer, language=query_language)

        result = {
            "query": query,
            "answer": final_answer,
            "context_metadata": context_result["metadata"],
            "format_check": format_check,
            "generation_metrics": {
                "total_time_seconds": round(time.time() - t_start, 2),
                "stage_times": stage_times,
                "token_counts": token_counts,
                "citation_retry_attempts": citation_retry_attempts,
                "stage_success": stage_success,
            },
            "intermediate_results": {
                "evidence_evaluation": evaluation_json,
                "draft_answer": draft_answer,
                "review_feedback": review_json,
                "citation_validation": citation_validation,
            },
            "sources": self._format_sources(selected_chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._log_run(result)
        return result

    # ── 流式主入口 ────────────────────────────────────────────────
    def generate_stream(
        self,
        query: str,
        top_k: int = 8,
        fusion_strategy: str = "rrf",
        run_evaluation: bool = True,
        run_review: bool = True,
        conversation_context: str = "",
    ):
        """
        流式版本：检索/上下文组装/证据评估同步完成后，逐 token yield 答案草稿文本；
        批判性审查与最终组装仍同步执行（内部质量控制步骤，是结构化 JSON 或对草稿的
        整体重写，不适合逐字展示），完成后 yield 一个 "done" 事件，携带与 generate()
        完全一致结构的最终结果，供调用方落库/统计。

        与 generate() 的行为差异：流式路径跳过引用校验重试循环（重试需要整段重新
        生成，与"边生成边显示"的体验冲突），引用正确性改由末尾的兜底 strip 安全网
        保证——代价是放弃了非流式路径下"反馈错误后重新生成"的主动修正能力。

        Yields:
            {"event": "status", "stage": str}
            {"event": "token", "text": str}
            {"event": "done", "result": dict}   # result 结构与 generate() 返回值一致
            {"event": "error", "message": str}
        """
        t_start = time.time()
        stage_times: dict[str, float] = {}
        stage_success: dict[str, bool] = {}
        token_counts: dict[str, int] = {}
        query_language = detect_query_language(query)
        language_directive = LANGUAGE_DIRECTIVES[query_language]
        hard_constraints = build_hard_constraints(query_language)

        try:
            yield {"event": "status", "stage": "retrieving"}
            t0 = time.time()
            retrieval_out = self.retrieval_pipeline.retrieve(query, top_k=top_k, fusion_strategy=fusion_strategy)
            stage_times["retrieval"] = round(time.time() - t0, 2)

            t0 = time.time()
            context_result = self.context_assembler.assemble(retrieval_out["results"], query=query)
            stage_times["context_assembly"] = round(time.time() - t0, 2)
            token_counts["context"] = context_result["metadata"]["estimated_tokens"]
            context_text = context_result["context_text"]
            selected_chunks = context_result["selected_chunks"]

            evaluation_json = None
            if run_evaluation and context_text:
                yield {"event": "status", "stage": "evaluating_evidence"}
                t0 = time.time()
                try:
                    eval_out = self._run_stage(
                        "evidence_evaluator", require_json=True, query=query, context=context_text,
                    )
                    evaluation_json = eval_out.get("json")
                    stage_success["evidence_evaluation"] = evaluation_json is not None
                    token_counts["evidence_evaluation"] = self.context_assembler.estimate_tokens(eval_out["text"])
                except Exception as e:
                    self.log.warning(f"证据评估阶段失败：{e}")
                    stage_success["evidence_evaluation"] = False
                stage_times["evidence_evaluation"] = round(time.time() - t0, 2)
                context_text, selected_chunks = self._filter_context_by_evaluation(
                    context_text, selected_chunks, evaluation_json
                )

            valid_citation_ids = {
                c.metadata.get("_citation_id") for c in selected_chunks
                if c.metadata.get("_citation_id") is not None
            }

            # 流式答案生成
            yield {"event": "status", "stage": "generating"}
            t0 = time.time()
            stage = self.stages["answer_generator"]
            prompt = stage.user_prompt_template.format(
                query=query, context=context_text or "（未检索到相关文献）",
                language_directive=language_directive, retry_note="",
                conversation_context=conversation_context,
            )
            system_prompt = f"{hard_constraints}\n\n{stage.system_prompt}"
            draft_answer = ""
            for chunk in self.llm.generate_stream(
                prompt=prompt, system_prompt=system_prompt,
                temperature=stage.temperature, max_tokens=stage.max_tokens,
            ):
                draft_answer += chunk
                yield {"event": "token", "text": chunk}
            draft_answer = draft_answer.strip()
            stage_success["answer_generation"] = bool(draft_answer)
            token_counts["draft_answer"] = self.context_assembler.estimate_tokens(draft_answer)
            stage_times["answer_generation"] = round(time.time() - t0, 2)

            citation_validation = self.citation_validator.validate(draft_answer, valid_citation_ids)

            # 批判性审查（可选，同步）
            review_json = None
            if run_review and draft_answer:
                yield {"event": "status", "stage": "reviewing"}
                t0 = time.time()
                try:
                    review_out = self._run_stage(
                        "critical_reviewer", require_json=True,
                        query=query, context=context_text, draft_answer=draft_answer,
                    )
                    review_json = review_out.get("json")
                    stage_success["critical_review"] = review_json is not None
                    token_counts["critical_review"] = self.context_assembler.estimate_tokens(review_out["text"])
                except Exception as e:
                    self.log.warning(f"批判性审查阶段失败：{e}")
                    stage_success["critical_review"] = False
                stage_times["critical_review"] = round(time.time() - t0, 2)

            t0 = time.time()
            final_answer = draft_answer
            if review_json:
                if review_json.get("overall_verdict") == "revise":
                    yield {"event": "status", "stage": "finalizing"}
                    try:
                        final_out = self._run_stage(
                            "final_assembler", require_json=False,
                            extra_system_prompt=hard_constraints,
                            query=query, draft_answer=draft_answer,
                            review_feedback=json.dumps(review_json, ensure_ascii=False),
                            language_directive=language_directive,
                        )
                        final_answer = final_out["text"].strip() or draft_answer
                        stage_success["final_assembly"] = True
                    except Exception as e:
                        self.log.warning(f"最终组装阶段失败，回退使用草稿：{e}")
                        stage_success["final_assembly"] = False
                else:
                    stage_success["final_assembly"] = True
            else:
                stage_success["final_assembly"] = False
            stage_times["final_assembly"] = round(time.time() - t0, 2)
            token_counts["final_answer"] = self.context_assembler.estimate_tokens(final_answer)

            final_answer = self.citation_validator.strip_invalid_citations(final_answer, valid_citation_ids)
            final_answer = self._postprocess(final_answer, selected_chunks, language=query_language)
            format_check = self.format_checker.check(final_answer, language=query_language)

            result = {
                "query": query,
                "answer": final_answer,
                "context_metadata": context_result["metadata"],
                "format_check": format_check,
                "generation_metrics": {
                    "total_time_seconds": round(time.time() - t_start, 2),
                    "stage_times": stage_times,
                    "token_counts": token_counts,
                    "citation_retry_attempts": 0,  # 流式路径不做重试循环，见上方说明
                    "stage_success": stage_success,
                },
                "intermediate_results": {
                    "evidence_evaluation": evaluation_json,
                    "draft_answer": draft_answer,
                    "review_feedback": review_json,
                    "citation_validation": citation_validation,
                },
                "sources": self._format_sources(selected_chunks),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._log_run(result)
            yield {"event": "done", "result": result}
        except Exception as e:
            self.log.error(f"流式生成失败：{e}")
            yield {"event": "error", "message": str(e)}
