"""
document_index.py — 文档级索引（DocumentIndex）
ChromaDB 里存的是 chunk 级记录，一篇文档对应多个 chunk。启动时扫描一次集合，
按 doc_id 聚合出文档级视图（标题/摘要/期刊/发表年份等优先取该文档的摘要类
chunk，没有摘要 chunk 则退化为遇到的第一个 chunk），供文档列表/详情接口使用。

注意：当前实现会一次性拉取集合内全部 chunk 的 metadata+documents 到内存构建
索引，适合原型/中小规模集合（如 test_dir_mode 的 1854 chunks）。若指向百万级
pmc_full 集合，扫描耗时与内存占用会显著上升，需要改为增量更新而非启动时全量
构建——这也是"增量更新次数"这一统计指标预留的意义：当前版本尚无增量更新
入口（Part 2 只开放只读查询），该计数器恒为 0。
"""

from __future__ import annotations

import logging


class DocumentIndex:
    def __init__(self, log: logging.Logger | None = None):
        self.log = log or logging.getLogger("document_index")
        self._docs: dict[str, dict] = {}   # doc_id -> 聚合后的文档视图
        self._order: list[str] = []        # 保持首次出现顺序，用于稳定分页
        self.incremental_update_count = 0  # 预留：未来增量写入接口调用一次此计数 +1

    def build_from_collection(self, collection, page_size: int = 5000) -> None:
        total = collection.count()
        offset = 0
        chunk_counts: dict[str, int] = {}

        while offset < total:
            batch = collection.get(limit=page_size, offset=offset, include=["metadatas", "documents"])
            for doc_text, meta in zip(batch["documents"], batch["metadatas"]):
                doc_id = meta.get("doc_id") or meta.get("pmc_id")
                if not doc_id:
                    continue
                chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
                is_abstract = meta.get("chunk_type") == "abstract"

                if doc_id not in self._docs:
                    self._order.append(doc_id)
                    self._docs[doc_id] = self._to_doc_view(doc_id, meta, doc_text, is_abstract)
                elif is_abstract and not self._docs[doc_id]["_from_abstract"]:
                    # 优先用摘要 chunk 覆盖（标题/摘要信息更完整）
                    self._docs[doc_id] = self._to_doc_view(doc_id, meta, doc_text, is_abstract)
            offset += page_size

        for doc_id, count in chunk_counts.items():
            self._docs[doc_id]["chunk_count"] = count

        self.log.info(f"文档索引构建完成：{len(self._docs):,} 篇文档（来自 {total:,} 个 chunk）")

    @staticmethod
    def _to_doc_view(doc_id: str, meta: dict, doc_text: str, is_abstract: bool) -> dict:
        return {
            "doc_id": doc_id,
            "title": meta.get("source_title") or "Untitled",
            "abstract": doc_text if is_abstract else None,
            "journal": meta.get("journal") or None,
            "pub_date": str(meta.get("pub_year")) if meta.get("pub_year") else None,
            "pmid": meta.get("pmid") or None,
            "doi": meta.get("doi") or None,
            "article_type": meta.get("article_type") or None,
            "chunk_count": 0,
            "_from_abstract": is_abstract,
        }

    def get(self, doc_id: str) -> dict | None:
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        return {k: v for k, v in doc.items() if not k.startswith("_")}

    def list_documents(self, page: int, page_size: int) -> tuple[list[dict], int]:
        total = len(self._order)
        start = (page - 1) * page_size
        end = start + page_size
        page_ids = self._order[start:end]
        items = [{k: v for k, v in self._docs[i].items() if not k.startswith("_")} for i in page_ids]
        return items, total

    def total_chunk_count(self) -> int:
        return sum(doc["chunk_count"] for doc in self._docs.values())

    def __len__(self) -> int:
        return len(self._docs)
