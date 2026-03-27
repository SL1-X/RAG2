from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
import re
from collections import defaultdict

from app.utils.logger import get_logger
from app.services.settings_service import settings_service
from app.services.retrieval_service import retrieval_service
from app.utils.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)

# 与检索设置中的 vector / keyword / hybrid 不同：这里表示「整条 RAG 流水线」如何跑
PIPELINE_MODE_FULL = "full"
PIPELINE_MODE_RETRIEVE_ONLY = "retrieve_only"
PIPELINE_MODE_GENERATE_ONLY = "generate_only"
PIPELINE_MODE_VECTOR_GENERATE = "vector_generate"
PIPELINE_MODE_KEYWORD_GENERATE = "keyword_generate"
PIPELINE_MODE_HYBRID_GENERATE = "hybrid_generate"
# 三路检索并行 + 三路生成并行，各自上下文独立
PIPELINE_MODE_TRIPLE_PARALLEL = "triple_parallel"
VALID_PIPELINE_MODES = frozenset(
    {
        PIPELINE_MODE_FULL,
        PIPELINE_MODE_RETRIEVE_ONLY,
        PIPELINE_MODE_GENERATE_ONLY,
        PIPELINE_MODE_VECTOR_GENERATE,
        PIPELINE_MODE_KEYWORD_GENERATE,
        PIPELINE_MODE_HYBRID_GENERATE,
        PIPELINE_MODE_TRIPLE_PARALLEL,
    }
)

_TRIPLE_BRANCH_ORDER = ("vector", "keyword", "hybrid")
STRICT_RAG_REFUSAL = "抱歉，当前检索到的文档中没有足够依据回答该问题。"


class RAGService:
    def __init__(self):
        pass

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_bool(value, default=True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _has_role_fallback(settings: dict, role: str) -> bool:
        prefix = f"{role}_llm"
        provider = str(settings.get(f"{prefix}_fallback_provider", "")).strip()
        model_name = str(settings.get(f"{prefix}_fallback_model_name", "")).strip()
        return bool(provider and model_name)

    @staticmethod
    def _doc_uid(doc) -> str:
        meta = doc.metadata or {}
        return str(meta.get("chunk_id") or meta.get("id") or meta.get("doc_id") or id(doc))

    def _should_rewrite_query(self, question: str, history, settings: dict) -> bool:
        if not history:
            return False
        if not self._as_bool(settings.get("enable_query_rewrite", True), True):
            return False
        if not self._as_bool(settings.get("rewrite_only_when_needed", True), True):
            return True
        q = (question or "").strip()
        if not q:
            return False
        short_question = len(q) <= 20
        pronoun_hints = (
            "它",
            "他",
            "她",
            "这个",
            "那个",
            "上述",
            "前面",
            "刚才",
            "这段",
            "that",
            "this",
            "it",
            "they",
            "them",
            "he",
            "she",
            "former",
            "latter",
        )
        needs_reference_resolution = any(h in q.lower() for h in pronoun_hints)
        return short_question or needs_reference_resolution

    def _get_rag_prompt(self, settings: dict) -> ChatPromptTemplate:
        rag_system_prompt = settings.get("rag_system_prompt")
        rag_query_prompt = settings.get("rag_query_prompt")
        return ChatPromptTemplate.from_messages(
            [("system", rag_system_prompt), ("human", rag_query_prompt)]
        )

    def _retrieve_documents(
        self,
        kb_id,
        question,
        settings: dict | None = None,
        *,
        retrieval_mode: str | None = None,
    ):
        """retrieval_mode 显式指定时忽略设置里的 retrieval_mode（用于三路并行）。"""
        settings = settings or settings_service.get()
        collection_name = f"kb_{kb_id}"
        retrieval_mode = (
            retrieval_mode
            if retrieval_mode is not None
            else settings.get("retrieval_mode", "vector")
        )
        # 总结问题做多查询融合召回，提升覆盖率与主旨提取能力
        if self._is_summary_intent(question):
            return self._retrieve_documents_multi_query(
                kb_id=kb_id,
                question=question,
                settings=settings,
                retrieval_mode=retrieval_mode,
            )
        if retrieval_mode == "vector":
            docs = retrieval_service.vector_search(
                collection_name=collection_name,
                query=question,
                rerank=True,
                settings=settings,
            )
        elif retrieval_mode == "keyword":
            docs = retrieval_service.keyword_search(
                collection_name=collection_name,
                query=question,
                rerank=True,
                settings=settings,
            )
        elif retrieval_mode == "hybrid":
            docs = retrieval_service.hybrid_search(
                collection_name=collection_name,
                query=question,
                settings=settings,
            )
        else:
            logger.warning(f"未知的检索模型:{retrieval_mode},转化使用向量检索")
            docs = retrieval_service.vector_search(
                collection_name=collection_name,
                query=question,
                settings=settings,
            )
        logger.info(f"使用{retrieval_mode}模型检索到{len(docs)}个文档")
        return docs

    def _build_summary_queries(self, question: str) -> list[str]:
        q = (question or "").strip()
        if not q:
            return []
        variants = [
            q,
            f"{q}\n请关注文章的主题、关键观点、结论和核心依据。",
            f"{q}\n提取主要内容、结构脉络与关键结论。",
            f"{q}\nFocus on major themes, key points, and final conclusions.",
        ]
        # 去重保序
        seen = set()
        out = []
        for item in variants:
            k = item.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(item)
        return out

    def _retrieve_documents_multi_query(
        self, kb_id: str, question: str, settings: dict, retrieval_mode: str
    ):
        collection_name = f"kb_{kb_id}"
        queries = self._build_summary_queries(question)
        def _run_one_query(q: str):
            if retrieval_mode == "vector":
                return retrieval_service.vector_search(
                    collection_name=collection_name,
                    query=q,
                    rerank=True,
                    settings=settings,
                )
            if retrieval_mode == "keyword":
                return retrieval_service.keyword_search(
                    collection_name=collection_name,
                    query=q,
                    rerank=True,
                    settings=settings,
                )
            return retrieval_service.hybrid_search(
                collection_name=collection_name,
                query=q,
                settings=settings,
            )

        all_docs = [[] for _ in queries]
        max_workers = min(4, max(1, len(queries)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one_query, q): idx for idx, q in enumerate(queries)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    all_docs[idx] = fut.result() or []
                except Exception as e:
                    logger.warning(f"多查询召回子查询失败（忽略）：{e}")
                    all_docs[idx] = []

        # 融合：优先多查询重复命中的片段，其次看最佳排名
        merged = {}
        for docs in all_docs:
            for rank, doc in enumerate(docs or [], start=1):
                uid = self._doc_uid(doc)
                if uid not in merged:
                    merged[uid] = {"doc": doc, "hit_count": 0, "best_rank": rank}
                merged[uid]["hit_count"] += 1
                merged[uid]["best_rank"] = min(merged[uid]["best_rank"], rank)
        fused = sorted(
            merged.values(),
            key=lambda x: (-x["hit_count"], x["best_rank"]),
        )
        top_k = self._as_int(settings.get("top_k", 5), 5)
        boost_k = max(top_k + 4, min(top_k * 3, 15))
        docs = [item["doc"] for item in fused[:boost_k]]
        logger.info(
            f"总结多查询召回完成：queries={len(queries)}, merged={len(fused)}, return={len(docs)}"
        )
        return docs

    @staticmethod
    def build_context_from_documents(docs) -> str:
        return "\n\n".join(
            [
                f"文档{i + 1} ({doc.metadata.get('doc_name', '未知文档')}):\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )

    @staticmethod
    def build_context_from_history(history) -> str:
        """
        将历史消息格式化为可注入提示词的文本，默认仅保留最近 10 条以控制长度。
        """
        if not history:
            return ""
        lines = []
        for item in history[-10:]:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if not content:
                continue
            speaker = "用户" if role in {"user", "human"} else "助手"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def _merge_context_and_history(self, context: str, history) -> str:
        history_text = self.build_context_from_history(history)
        if not history_text:
            return context or ""
        if not context:
            return f"对话历史：\n{history_text}"
        return f"对话历史：\n{history_text}\n\n文档上下文：\n{context}"

    @staticmethod
    def _tokenize_for_citation(text: str) -> set[str]:
        if not text:
            return set()
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        return {t for t in tokens if t and len(t.strip()) > 0}

    @staticmethod
    def _split_answer_sentences(answer: str) -> list[str]:
        if not answer:
            return []
        segments = re.findall(r"[^。！？!?；;\n]+(?:[。！？!?；;]+)?|\n+", answer)
        if not segments:
            return [answer]
        return segments

    def _inject_inline_citations(self, answer: str, sources) -> tuple[str, list[dict]]:
        """
        为回答按句注入 [^n] 引用（n 对应 sources 的序号，1-based）。
        规则：每句最多一个引用，基于句子 token 与来源 chunk token 的重叠度匹配。
        """
        if not answer or not sources:
            return answer, []
        source_tokens = [
            self._tokenize_for_citation((src or {}).get("content", "")[:2200])
            for src in sources
        ]
        if not any(source_tokens):
            return answer, []

        used_indexes = set()
        cited_segments = []
        for seg in self._split_answer_sentences(answer):
            if not seg or seg.isspace():
                cited_segments.append(seg)
                continue
            if re.search(r"\[\^\d+\]\s*$", seg.rstrip()):
                cited_segments.append(seg)
                continue

            core = seg.strip()
            sentence_tokens = self._tokenize_for_citation(core)
            if len(sentence_tokens) < 2 or len(core) < 8:
                cited_segments.append(seg)
                continue

            best_idx = None
            best_score = 0.0
            for idx, st in enumerate(source_tokens, start=1):
                if not st:
                    continue
                overlap = sentence_tokens & st
                overlap_count = len(overlap)
                if overlap_count < 2:
                    continue
                score = overlap_count / max(1, len(sentence_tokens))
                score += 0.02 / idx
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score >= 0.12:
                used_indexes.add(best_idx)
                stripped = seg.rstrip()
                trailing = seg[len(stripped) :]
                cited_segments.append(f"{stripped} [^{best_idx}]{trailing}")
            else:
                cited_segments.append(seg)

        cited_answer = "".join(cited_segments)

        citation_map = [
            {
                "index": idx,
                "doc_id": (sources[idx - 1] or {}).get("doc_id"),
                "doc_name": (sources[idx - 1] or {}).get("doc_name"),
                "chunk_id": (sources[idx - 1] or {}).get("chunk_id"),
            }
            for idx in sorted(used_indexes)
        ]
        return cited_answer, citation_map

    @staticmethod
    def _split_context_sections(context: str) -> list[tuple[int, str]]:
        if not context:
            return []
        # 兼容 build_context_from_documents 输出：文档{i} (...) 开头
        pattern = r"(?:^|\n\n)(文档(\d+)\s*\([^)]+\):\n)"
        matches = list(re.finditer(pattern, context))
        if not matches:
            trimmed = context.strip()
            return [(1, trimmed)] if trimmed else []
        sections = []
        for idx, m in enumerate(matches):
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
            body = context[start:end].strip()
            try:
                doc_index = int(m.group(2))
            except Exception:
                doc_index = idx + 1
            if body:
                sections.append((doc_index, body))
        return sections

    def _invoke_summary_map_reduce(
        self, question: str, context: str, settings: dict, history=None
    ) -> str:
        merged_context = self._merge_context_and_history(context, history)
        sections = self._split_context_sections(merged_context)
        if not sections:
            return STRICT_RAG_REFUSAL
        # 控制 map 阶段成本
        max_sections = self._as_int(settings.get("summary_max_sections", 8), 8)
        max_sections = max(3, min(max_sections, 12))
        sections = sections[:max_sections]
        map_max_tokens = max(
            640,
            self._as_int(settings.get("summary_map_max_tokens", 768), 768),
        )
        reduce_max_tokens = max(
            1024,
            self._as_int(
                settings.get(
                    "summary_reduce_max_tokens",
                    max(self._as_int(settings.get("rag_llm_max_tokens", 1024), 1024), 1280),
                ),
                1280,
            ),
        )
        map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You summarize only from provided text. Do not use external knowledge. "
                    "Output 4-6 concise bullets that preserve key facts, terms, and constraints. "
                    "Each bullet must end with one citation [^n].",
                ),
                (
                    "human",
                    "Question:\n{question}\n\nSource section index: [^{idx}]\n\nSection content:\n{section}\n\n"
                    "Return only bullet points.",
                ),
            ]
        )
        def _run_map_one(item: tuple[int, str]) -> tuple[int, str]:
            idx, sec = item
            llm = LLMFactory.create_llm(
                settings,
                role="rag",
                max_tokens=map_max_tokens,
                temperature=min(
                    self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7), 0.35
                ),
            )
            chain = map_prompt | llm
            out = chain.invoke({"question": question, "section": sec, "idx": idx})
            text = (out.content if getattr(out, "content", None) else str(out)).strip()
            return idx, text

        partial_summaries_by_idx = {}
        map_workers = min(4, max(1, len(sections)))
        with ThreadPoolExecutor(max_workers=map_workers) as pool:
            futures = {pool.submit(_run_map_one, item): item[0] for item in sections}
            for fut in as_completed(futures):
                section_idx = futures[fut]
                try:
                    idx, text = fut.result()
                    if text:
                        partial_summaries_by_idx[idx] = text
                except Exception as e:
                    logger.warning(f"map摘要失败，跳过分段[{section_idx}]：{e}")
        partial_summaries = [
            partial_summaries_by_idx[i] for i, _ in sections if i in partial_summaries_by_idx
        ]
        if not partial_summaries:
            return STRICT_RAG_REFUSAL
        reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grounded summarizer. Merge partial summaries into one final answer. "
                    "Keep only evidence-supported claims. Every key point must include citations [^n]. "
                    "If evidence is insufficient, return exactly: " + STRICT_RAG_REFUSAL,
                ),
                (
                    "human",
                    "Question:\n{question}\n\nPartial summaries:\n{partials}\n\n"
                    "Return format:\n1) 一句话总览\n2) 5-8条关键要点（每条含引用）\n3) 结论与适用边界（含引用）",
                ),
            ]
        )
        llm = LLMFactory.create_llm(
            settings,
            role="rag",
            max_tokens=reduce_max_tokens,
            temperature=min(self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7), 0.35),
        )
        chain = reduce_prompt | llm
        out = chain.invoke({"question": question, "partials": "\n\n".join(partial_summaries)})
        reduced = (out.content if getattr(out, "content", None) else str(out)).strip()
        if not self._looks_truncated_summary(reduced) and not self._looks_too_brief_summary(
            reduced
        ):
            return reduced

        logger.warning("总结 reduce 输出疑似截断，执行一次补全重试")
        retry_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grounded summarizer. Produce a complete and concise Chinese answer "
                    "strictly based on the provided partial summaries. Every key point must include citations [^n]. "
                    "Do not end with an unfinished phrase.",
                ),
                (
                    "human",
                    "Question:\n{question}\n\nPartial summaries:\n{partials}\n\n"
                    "Output exactly:\n1) 一句话总览（完整句）\n2) 5-8条关键要点（每条含引用）\n3) 结论与适用边界（完整句，含引用）",
                ),
            ]
        )
        retry_chain = retry_prompt | llm
        retry_out = retry_chain.invoke(
            {"question": question, "partials": "\n\n".join(partial_summaries)}
        )
        retried = (
            retry_out.content if getattr(retry_out, "content", None) else str(retry_out)
        ).strip()
        return retried or reduced

    @staticmethod
    def _strip_inline_citations(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\[\^\d+\]", "", text)

    @staticmethod
    def _is_summary_intent(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        keywords = [
            "总结",
            "概述",
            "主要内容",
            "摘要",
            "main content",
            "summar",
            "overview",
            "key points",
            "tl;dr",
        ]
        return any(k in q for k in keywords)

    @staticmethod
    def _looks_truncated_summary(answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        if len(text) < 24:
            return True
        if re.search(r"[，、：;；,:]\s*$", text):
            return True
        if not re.search(r"[。！？.!?](?:\s|\]|$)", text):
            return True
        return False

    @staticmethod
    def _looks_too_brief_summary(answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        # 经验阈值：少于两行或整体过短时，通常无法覆盖“更详细准确”的要求
        non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(non_empty_lines) < 2:
            return True
        if len(text) < 220:
            return True
        return False

    @staticmethod
    def _is_refusal_answer(answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        text = re.sub(r"\[\^\d+\]", "", text).strip()
        refusal_prefixes = (
            "抱歉，当前检索到的文档中没有足够依据回答该问题",
            "抱歉，当前的知识库文档中没有找到与该问题相关的信息",
        )
        return any(text.startswith(prefix) for prefix in refusal_prefixes)

    def _is_supported_sentence(
        self, sentence: str, source_tokens: set[str], *, summary_mode: bool = False
    ) -> bool:
        sentence_clean = self._strip_inline_citations(sentence).strip()
        if not sentence_clean:
            return True
        sentence_tokens = self._tokenize_for_citation(sentence_clean)
        # 过短片段（如标题、连接词）不参与硬校验
        if len(sentence_tokens) < 3 or len(sentence_clean) < 8:
            return True
        overlap = sentence_tokens & source_tokens
        min_overlap = 1 if summary_mode else 2
        if len(overlap) < min_overlap:
            return False
        coverage = len(overlap) / max(1, len(sentence_tokens))
        min_coverage = 0.08 if summary_mode else 0.18
        return coverage >= min_coverage

    @staticmethod
    def _is_summary_key_line(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        if re.match(r"^[-*•]\s+", s):
            return True
        if re.match(r"^\d+[.)、]\s+", s):
            return True
        lowered = s.lower()
        if lowered.startswith(("overview", "summary", "conclusion", "结论", "总览", "要点")):
            return True
        return len(s) >= 24

    def _attach_evidence_citations_for_summary(self, answer: str, sources) -> str:
        """
        总结场景优先做“要点级证据对齐”：
        - 对每行关键要点匹配最相关来源；
        - 若该行尚无引用，则追加一个 [^n]。
        """
        if not answer or not sources:
            return answer
        source_tokens = [
            self._tokenize_for_citation((src or {}).get("content", "")[:2600])
            for src in sources
        ]
        if not any(source_tokens):
            return answer
        lines = answer.splitlines(keepends=True)
        out = []
        for line in lines:
            stripped = line.strip()
            if not self._is_summary_key_line(stripped):
                out.append(line)
                continue
            if re.search(r"\[\^\d+\]", stripped):
                out.append(line)
                continue
            tokens = self._tokenize_for_citation(stripped)
            if len(tokens) < 2:
                out.append(line)
                continue
            best_idx = None
            best_score = 0.0
            for idx, st in enumerate(source_tokens, start=1):
                if not st:
                    continue
                overlap = tokens & st
                if len(overlap) < 1:
                    continue
                score = len(overlap) / max(1, len(tokens))
                score += 0.02 / idx
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None or best_score < 0.06:
                out.append(line)
                continue
            no_newline = line.rstrip("\r\n")
            suffix = line[len(no_newline) :]
            out.append(f"{no_newline} [^{best_idx}]{suffix}")
        return "".join(out)

    def _enforce_answer_grounded(
        self, answer: str, sources, question: str = ""
    ) -> tuple[str, list[dict], str]:
        """
        强制答案基于检索文档：
        1) 必须存在可用 sources；
        2) 必须生成至少一个有效引用 [^n]；
        3) 关键句必须与 sources 有足够词元重叠。
        返回: (final_answer, citation_map, reason)
        """
        raw_answer = (answer or "").strip()
        if not raw_answer:
            return STRICT_RAG_REFUSAL, [], "empty_answer"
        if not sources:
            return STRICT_RAG_REFUSAL, [], "no_sources"
        summary_mode = self._is_summary_intent(question)
        if summary_mode:
            raw_answer = self._attach_evidence_citations_for_summary(raw_answer, sources)

        cited_answer, citation_map = self._inject_inline_citations(raw_answer, sources)
        if not citation_map:
            if summary_mode and sources:
                # 总结类问题允许降级为“基础引用”以避免误拒答
                stripped = raw_answer.rstrip()
                trailing = raw_answer[len(stripped) :]
                cited_answer = f"{stripped} [^1]{trailing}" if stripped else raw_answer
                citation_map = [
                    {
                        "index": 1,
                        "doc_id": (sources[0] or {}).get("doc_id"),
                        "doc_name": (sources[0] or {}).get("doc_name"),
                        "chunk_id": (sources[0] or {}).get("chunk_id"),
                    }
                ]
            else:
                return STRICT_RAG_REFUSAL, [], "no_valid_citations"

        source_tokens = set()
        for src in sources:
            source_tokens |= self._tokenize_for_citation((src or {}).get("content", "")[:3000])
        if not source_tokens:
            return STRICT_RAG_REFUSAL, [], "empty_source_tokens"

        unsupported = 0
        checked = 0
        for seg in self._split_answer_sentences(cited_answer):
            seg_clean = (seg or "").strip()
            if not seg_clean:
                continue
            if seg_clean.startswith("#"):
                continue
            checked += 1
            if not self._is_supported_sentence(
                seg_clean, source_tokens, summary_mode=summary_mode
            ):
                unsupported += 1
        if checked == 0:
            return STRICT_RAG_REFUSAL, [], "no_checkable_sentence"
        if not summary_mode and unsupported > 0:
            return STRICT_RAG_REFUSAL, [], f"unsupported_sentences={unsupported}"
        if summary_mode:
            # 总结场景更关注“整体可追溯”，放宽到“多数要点有依据”即可通过
            # 1) 短回答（<=2个可检句）允许 1 句不完全匹配；
            # 2) 常规回答要求不支持句占比 <= 60%。
            if checked <= 2:
                if unsupported > 1:
                    return STRICT_RAG_REFUSAL, [], f"unsupported_ratio={unsupported}/{checked}"
            else:
                if (unsupported / max(1, checked)) > 0.60:
                    return STRICT_RAG_REFUSAL, [], f"unsupported_ratio={unsupported}/{checked}"
        return cited_answer, citation_map, "ok"

    def _rewrite_query_from_history(
        self, question: str, history, settings: dict
    ) -> str:
        """
        基于历史对话把当前问题改写为“可独立检索”的查询语句。
        失败或无历史时回退到原问题。
        """
        history_text = self.build_context_from_history(history)
        if not history_text:
            return question
        try:
            llm = LLMFactory.create_llm(
                settings,
                temperature=self._as_float(
                    settings.get("rewrite_llm_temperature", 0.0), 0.0
                ),
                max_tokens=self._as_int(settings.get("rewrite_llm_max_tokens", 256), 256),
                streaming=False,
                role="rewrite",
            )
            rewrite_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a retrieval query rewriter. Rewrite the user's latest question "
                        "into one standalone search query, resolving coreference when needed. "
                        "Keep the same language as the latest user question (English question -> English rewrite). "
                        "Do not answer, do not explain, output only the rewritten query.",
                    ),
                    (
                        "human",
                        "Conversation history:\n{history}\n\nCurrent question:\n{question}\n\nRewritten retrieval query:",
                    ),
                ]
            )
            chain = rewrite_prompt | llm
            out = chain.invoke({"history": history_text, "question": question})
            rewritten = (out.content if getattr(out, "content", None) else str(out)).strip()
            if not rewritten:
                return question
            return rewritten.splitlines()[0].strip() or question
        except Exception as e:
            if self._has_role_fallback(settings, "rewrite"):
                try:
                    logger.warning(f"查询改写主模型失败，尝试 fallback: {e}")
                    fallback_llm = LLMFactory.create_llm(
                        settings,
                        temperature=self._as_float(
                            settings.get("rewrite_llm_temperature", 0.0), 0.0
                        ),
                        max_tokens=self._as_int(
                            settings.get("rewrite_llm_max_tokens", 256), 256
                        ),
                        streaming=False,
                        role="rewrite",
                        use_fallback=True,
                    )
                    rewrite_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a retrieval query rewriter. Rewrite the user's latest question "
                                "into one standalone search query, resolving coreference when needed. "
                                "Keep the same language as the latest user question (English question -> English rewrite). "
                                "Do not answer, do not explain, output only the rewritten query.",
                            ),
                            (
                                "human",
                                "Conversation history:\n{history}\n\nCurrent question:\n{question}\n\nRewritten retrieval query:",
                            ),
                        ]
                    )
                    chain = rewrite_prompt | fallback_llm
                    out = chain.invoke({"history": history_text, "question": question})
                    rewritten = (
                        out.content if getattr(out, "content", None) else str(out)
                    ).strip()
                    if rewritten:
                        return rewritten.splitlines()[0].strip() or question
                except Exception as fallback_error:
                    logger.warning(f"查询改写 fallback 也失败，回退原问题: {fallback_error}")
            else:
                logger.warning(f"查询改写失败，回退原问题: {e}")
            return question

    def retrieve(self, kb_id, question) -> tuple[list, str]:
        """仅检索：返回 LangChain Document 列表与拼好的上下文字符串。"""
        settings = settings_service.get()
        docs = self._retrieve_documents(kb_id, question, settings)
        context = self.build_context_from_documents(docs)
        return docs, context

    @staticmethod
    def _extract_retrieval_debug(docs) -> dict | None:
        for doc in docs or []:
            metadata = doc.metadata or {}
            debug = metadata.get("retrieval_debug")
            if isinstance(debug, dict):
                return debug
        return None

    def _stream_llm_answer(self, question: str, context: str, settings: dict, history=None):
        rag_prompt = self._get_rag_prompt(settings)
        merged_context = self._merge_context_and_history(context, history)

        def _run_stream(use_fallback: bool):
            llm = LLMFactory.create_llm(
                settings,
                role="rag",
                max_tokens=self._as_int(settings.get("rag_llm_max_tokens", 1024), 1024),
                temperature=self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7),
                use_fallback=use_fallback,
            )
            chain = rag_prompt | llm
            for chunk in chain.stream({"context": merged_context, "question": question}):
                if chunk.content:
                    yield chunk.content

        try:
            for text in _run_stream(use_fallback=False):
                yield text
        except Exception as e:
            if not self._has_role_fallback(settings, "rag"):
                raise
            logger.warning(f"RAG 主模型流式失败，尝试 fallback: {e}")
            for text in _run_stream(use_fallback=True):
                yield text

    def _invoke_rag_answer(
        self, question: str, context: str, settings: dict, history=None
    ) -> str:
        if self._is_summary_intent(question):
            try:
                summary_answer = self._invoke_summary_map_reduce(
                    question, context, settings, history=history
                )
                if self._looks_truncated_summary(summary_answer) or self._looks_too_brief_summary(
                    summary_answer
                ):
                    logger.warning("总结答案疑似不完整或过短，执行总结增强修复")
                    repaired = self._invoke_summary_repair(
                        question, context, settings, history=history, draft=summary_answer
                    )
                    if repaired:
                        return repaired
                    repaired2 = self._invoke_summary_repair(
                        question,
                        context,
                        settings,
                        history=history,
                        draft=summary_answer
                        + "\n\n请补齐遗漏的关键机制、流程、约束条件与例外情况，避免只有少量要点。",
                    )
                    if repaired2:
                        return repaired2
                return summary_answer
            except Exception as e:
                logger.warning(f"总结map-reduce失败，降级为单次生成: {e}")
        return self._invoke_rag_answer_single_pass(
            question, context, settings, history=history
        )

    def _invoke_summary_repair(
        self, question: str, context: str, settings: dict, history=None, draft: str = ""
    ) -> str:
        merged_context = self._merge_context_and_history(context, history)
        llm = LLMFactory.create_llm(
            settings,
            role="rag",
            # 修复阶段适当放宽 token，提升细节覆盖
            max_tokens=max(
                1280,
                self._as_int(
                    settings.get(
                        "summary_repair_max_tokens",
                        settings.get("rag_llm_max_tokens", 1024),
                    ),
                    1280,
                ),
            ),
            temperature=min(self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7), 0.35),
        )
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个严格基于给定文档作答的总结助手。"
                    "只能使用提供的文档上下文，不允许外推。"
                    "输出必须完整、细致，并且每条关键结论都带引用 [^n]。"
                    "若证据不足，明确说明不足点，不要写半句。",
                ),
                (
                    "human",
                    "【问题】\n{question}\n\n"
                    "【已有草稿（可能过短或不完整）】\n{draft}\n\n"
                    "【文档上下文】\n{context}\n\n"
                    "请按固定结构输出：\n"
                    "1) 一句话总览（完整句）\n"
                    "2) 5-7 条关键要点（每条含引用 [^n]）\n"
                    "3) 结论与适用边界（含引用 [^n]）",
                ),
            ]
        )
        chain = repair_prompt | llm
        out = chain.invoke(
            {"question": question, "draft": draft or "(无)", "context": merged_context}
        )
        text = (out.content if getattr(out, "content", None) else str(out)).strip()
        if self._looks_truncated_summary(text) or self._looks_too_brief_summary(text):
            return ""
        return text

    def _invoke_rag_answer_single_pass(
        self, question: str, context: str, settings: dict, history=None
    ) -> str:
        rag_prompt = self._get_rag_prompt(settings)
        merged_context = self._merge_context_and_history(context, history)
        try:
            llm = LLMFactory.create_llm(
                settings,
                role="rag",
                max_tokens=self._as_int(settings.get("rag_llm_max_tokens", 1024), 1024),
                temperature=self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7),
            )
            chain = rag_prompt | llm
            out = chain.invoke({"context": merged_context, "question": question})
            return out.content if getattr(out, "content", None) else str(out)
        except Exception as e:
            if not self._has_role_fallback(settings, "rag"):
                raise
            logger.warning(f"RAG 主模型调用失败，尝试 fallback: {e}")
            llm = LLMFactory.create_llm(
                settings,
                role="rag",
                max_tokens=self._as_int(settings.get("rag_llm_max_tokens", 1024), 1024),
                temperature=self._as_float(settings.get("rag_llm_temperature", 0.7), 0.7),
                use_fallback=True,
            )
            chain = rag_prompt | llm
            out = chain.invoke({"context": merged_context, "question": question})
            return out.content if getattr(out, "content", None) else str(out)

    def _run_triple_retrieval_branch(
        self, kb_id: str, question: str, settings: dict, branch: str
    ):
        """单路检索：返回分支、sources、上下文与耗时。"""
        started = perf_counter()
        try:
            docs = self._retrieve_documents(
                kb_id, question, settings, retrieval_mode=branch
            )
            context = self.build_context_from_documents(docs)
            sources = self._extract_citations(docs)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "sources": sources,
                "context": context,
                "retrieved_chunks": len(docs),
                "retrieval_debug": self._extract_retrieval_debug(docs),
                "retrieval_elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            logger.exception("triple_parallel 检索分支 %s 失败", branch)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "sources": [],
                "context": "",
                "retrieved_chunks": 0,
                "retrieval_debug": None,
                "retrieval_elapsed_ms": elapsed_ms,
                "error": str(e),
            }

    def _run_triple_generation_branch(
        self,
        *,
        question: str,
        settings: dict,
        branch: str,
        context: str,
        retrieval_error: str | None = None,
        history=None,
    ):
        """单路生成：返回分支答案与生成耗时。"""
        started = perf_counter()
        try:
            # 检索失败时给出可读结果，避免空白分支。
            if retrieval_error:
                answer = f"（{branch} 分支检索失败，无法基于该路文档生成答案：{retrieval_error}）"
            else:
                answer = self._invoke_rag_answer(
                    question, context, settings, history=history
                )
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "answer": answer,
                "generation_elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            logger.exception("triple_parallel 生成分支 %s 失败", branch)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "answer": f"（本路生成出错：{e}）",
                "generation_elapsed_ms": elapsed_ms,
                "error": str(e),
            }

    def generate_stream(self, question: str, context: str = "", history=None):
        """仅生成：不访问向量库，用传入的 context 与 question 走 RAG 提示词 + LLM。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        try:
            for text in self._stream_llm_answer(
                question, context, settings, history=history
            ):
                yield {"type": "content", "content": text}
        except Exception as e:
            logger.error(f"RAG 生成阶段出错: {e}")
            yield {"type": "error", "content": str(e)}
            return
        yield {
            "type": "done",
            "content": "",
            "sources": [],
            "metadata": {
                "question": question,
                "pipeline_mode": PIPELINE_MODE_GENERATE_ONLY,
                "context_chars": len(context or ""),
            },
        }

    def retrieve_only_stream(self, kb_id, question, retrieval_query: str | None = None):
        """仅检索：不调用大模型，在 done 中返回 sources 与元数据。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        query_for_retrieval = retrieval_query or question
        filtered_docs = self._retrieve_documents(kb_id, query_for_retrieval, settings)
        sources = self._extract_citations(filtered_docs)
        retrieval_debug = self._extract_retrieval_debug(filtered_docs)
        yield {
            "type": "done",
            "content": "",
            "sources": sources,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": query_for_retrieval,
                "retrieved_chunks": len(filtered_docs),
                "retrieval_debug": retrieval_debug,
                "pipeline_mode": PIPELINE_MODE_RETRIEVE_ONLY,
                "hint": "仅检索模式：未调用大模型。",
            },
        }

    def full_rag_stream(
        self,
        kb_id,
        question,
        retrieval_mode: str | None = None,
        retrieval_query: str | None = None,
        history=None,
    ):
        """完整 RAG：先检索再生成（原 ask_stream 行为）。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        query_for_retrieval = retrieval_query or question
        filtered_docs = self._retrieve_documents(
            kb_id, query_for_retrieval, settings, retrieval_mode=retrieval_mode
        )
        context = self.build_context_from_documents(filtered_docs)
        answer_parts = []
        summary_mode = self._is_summary_intent(question)
        try:
            if summary_mode:
                # 总结问题优先使用 map-reduce，一次性产出更稳定的高层摘要
                answer = self._invoke_rag_answer(question, context, settings, history=history)
                answer_parts.append(answer or "")
                if answer:
                    yield {"type": "content", "content": answer}
            else:
                for text in self._stream_llm_answer(
                    question, context, settings, history=history
                ):
                    answer_parts.append(text)
                    yield {"type": "content", "content": text}
        except Exception as e:
            logger.error(f"RAG 生成阶段出错: {e}")
            yield {"type": "error", "content": str(e)}
            return
        sources = self._extract_citations(filtered_docs)
        cited_answer, citation_map, grounding_reason = self._enforce_answer_grounded(
            "".join(answer_parts), sources, question=question
        )
        if (
            summary_mode
            and sources
            and self._is_refusal_answer(cited_answer)
            and len(sources) >= 3
        ):
            logger.warning("总结场景命中拒答模板，触发二次总结修复")
            repaired_answer = self._invoke_summary_repair(
                question,
                context,
                settings,
                history=history,
                draft="".join(answer_parts),
            )
            if repaired_answer:
                cited_answer, citation_map, grounding_reason = self._enforce_answer_grounded(
                    repaired_answer, sources, question=question
                )
        if self._is_summary_intent(question) and grounding_reason.startswith("unsupported_ratio"):
            fallback_answer = self._attach_evidence_citations_for_summary(
                "".join(answer_parts), sources
            )
            fallback_answer, fallback_map = self._inject_inline_citations(
                fallback_answer, sources
            )
            if fallback_map:
                cited_answer = fallback_answer
                citation_map = fallback_map
                grounding_reason = "ok_summary_fallback"
        if grounding_reason not in {"ok", "ok_summary_fallback"}:
            logger.warning(f"RAG grounded gate 触发拒答: {grounding_reason}")
        retrieval_debug = self._extract_retrieval_debug(filtered_docs)
        yield {
            "type": "done",
            "content": "",
            "sources": sources,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": query_for_retrieval,
                "retrieved_chunks": len(filtered_docs),
                "retrieval_debug": retrieval_debug,
                "pipeline_mode": PIPELINE_MODE_FULL,
                "answer_with_citations": cited_answer,
                "citation_map": citation_map,
                "grounding_reason": grounding_reason,
                "retrieval_mode": retrieval_mode
                or settings.get("retrieval_mode", "vector"),
            },
        }

    def triple_parallel_stream(
        self, kb_id, question, history=None, retrieval_query: str | None = None
    ):
        """
        两阶段并行：
        1) 向量 / 关键词 / 混合三路检索并发执行；
        2) 三路检索完成后，三路生成并发执行。
        SSE 按“分支完成顺序”输出 branch_start → content → branch_done。
        """
        settings = settings_service.get()
        pipeline_started = perf_counter()
        yield {
            "type": "start",
            "content": "",
            "metadata": {"pipeline_mode": PIPELINE_MODE_TRIPLE_PARALLEL},
        }

        retrieval_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            retrieval_futures = {
                pool.submit(
                    self._run_triple_retrieval_branch,
                    kb_id,
                    retrieval_query or question,
                    settings,
                    b,
                ): b
                for b in _TRIPLE_BRANCH_ORDER
            }
            for fut in as_completed(retrieval_futures):
                result = fut.result()
                retrieval_results[result["branch"]] = result

        generation_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            generation_futures = {
                pool.submit(
                    self._run_triple_generation_branch,
                    question=question,
                    settings=settings,
                    branch=branch,
                    context=retrieval_results.get(branch, {}).get("context", ""),
                    retrieval_error=retrieval_results.get(branch, {}).get("error"),
                    history=history,
                ): branch
                for branch in _TRIPLE_BRANCH_ORDER
            }
            for fut in as_completed(generation_futures):
                result = fut.result()
                branch = result["branch"]
                generation_results[branch] = result
                retrieval_result = retrieval_results.get(branch, {})
                sources = retrieval_result.get("sources", [])
                cited_answer, citation_map, grounding_reason = self._enforce_answer_grounded(
                    result.get("answer", ""), sources, question=question
                )
                if self._is_summary_intent(question) and grounding_reason.startswith(
                    "unsupported_ratio"
                ):
                    fallback_answer = self._attach_evidence_citations_for_summary(
                        result.get("answer", ""), sources
                    )
                    fallback_answer, fallback_map = self._inject_inline_citations(
                        fallback_answer, sources
                    )
                    if fallback_map:
                        cited_answer = fallback_answer
                        citation_map = fallback_map
                        grounding_reason = "ok_summary_fallback"
                if grounding_reason not in {"ok", "ok_summary_fallback"}:
                    logger.warning(
                        "triple_parallel 分支 %s grounded gate 触发拒答: %s",
                        branch,
                        grounding_reason,
                    )
                result["answer"] = cited_answer
                result["citation_map"] = citation_map
                yield {"type": "branch_start", "branch": branch}
                yield {"type": "content", "branch": branch, "content": result["answer"]}
                yield {
                    "type": "branch_done",
                    "branch": branch,
                    "sources": sources,
                    "metadata": {
                        "kb_id": kb_id,
                        "question": question,
                        "retrieval_query": retrieval_query or question,
                        "retrieval_mode": branch,
                        "answer_with_citations": cited_answer,
                        "citation_map": citation_map,
                        "grounding_reason": grounding_reason,
                        "retrieved_chunks": retrieval_result.get("retrieved_chunks", 0),
                        "retrieval_debug": retrieval_result.get("retrieval_debug"),
                        "retrieval_elapsed_ms": retrieval_result.get(
                            "retrieval_elapsed_ms", 0
                        ),
                        "generation_elapsed_ms": result.get("generation_elapsed_ms", 0),
                    },
                }

        triple_payload = {}
        for branch in _TRIPLE_BRANCH_ORDER:
            retrieval_result = retrieval_results.get(branch, {})
            generation_result = generation_results.get(branch, {})
            triple_payload[branch] = {
                "answer": generation_result.get("answer", ""),
                "sources": retrieval_result.get("sources", []),
                "retrieved_chunks": retrieval_result.get("retrieved_chunks", 0),
                "retrieval_debug": retrieval_result.get("retrieval_debug"),
                "retrieval_elapsed_ms": retrieval_result.get("retrieval_elapsed_ms", 0),
                "generation_elapsed_ms": generation_result.get("generation_elapsed_ms", 0),
                "error": retrieval_result.get("error") or generation_result.get("error"),
                "citation_map": generation_result.get("citation_map", []),
            }
        yield {
            "type": "done",
            "content": "",
            "sources": None,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": retrieval_query or question,
                "pipeline_mode": PIPELINE_MODE_TRIPLE_PARALLEL,
                "pipeline_elapsed_ms": int((perf_counter() - pipeline_started) * 1000),
                "triple": triple_payload,
            },
        }

    def ask_stream(
        self,
        kb_id,
        question,
        pipeline_mode: str = PIPELINE_MODE_FULL,
        context: str | None = None,
        history=None,
    ):
        """
        统一入口：按 pipeline_mode 分流。
        - full: 检索 + 生成
        - retrieve_only: 仅检索
        - generate_only: 仅生成（使用请求体中的 context，不访问该知识库向量检索）
        """
        retrieval_query = question
        rewrite_enabled_modes = {
            PIPELINE_MODE_FULL,
            PIPELINE_MODE_RETRIEVE_ONLY,
            PIPELINE_MODE_VECTOR_GENERATE,
            PIPELINE_MODE_KEYWORD_GENERATE,
            PIPELINE_MODE_HYBRID_GENERATE,
            PIPELINE_MODE_TRIPLE_PARALLEL,
        }
        if pipeline_mode in rewrite_enabled_modes:
            settings = settings_service.get()
            if self._should_rewrite_query(question, history, settings):
                retrieval_query = self._rewrite_query_from_history(
                    question, history, settings
                )
                if retrieval_query != question:
                    logger.info(f"查询改写生效: {question} -> {retrieval_query}")

        if pipeline_mode == PIPELINE_MODE_RETRIEVE_ONLY:
            yield from self.retrieve_only_stream(
                kb_id, question, retrieval_query=retrieval_query
            )
        elif pipeline_mode == PIPELINE_MODE_GENERATE_ONLY:
            yield from self.generate_stream(question, context or "", history=history)
        elif pipeline_mode == PIPELINE_MODE_VECTOR_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="vector",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_KEYWORD_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="keyword",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_HYBRID_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="hybrid",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_TRIPLE_PARALLEL:
            yield from self.triple_parallel_stream(
                kb_id,
                question,
                history=history,
                retrieval_query=retrieval_query,
            )
        else:
            yield from self.full_rag_stream(
                kb_id, question, retrieval_query=retrieval_query, history=history
            )

    def _extract_citations(self, docs):
        sources = []

        def _to_percent_or_none(value):
            if value is None:
                return None
            try:
                return round(float(value) * 100, 2)
            except Exception:
                return None

        for doc in docs:
            metadata = doc.metadata
            retrieval_type = metadata.get("retrieval_type")
            rerank_score = _to_percent_or_none(metadata.get("rerank_score"))
            vector_score = _to_percent_or_none(metadata.get("vector_score"))
            keyword_score = _to_percent_or_none(metadata.get("keyword_score"))
            rrf_score = _to_percent_or_none(metadata.get("rrf_score"))
            chunk_id = metadata.get("chunk_id") or metadata.get("id")
            parent_id = metadata.get("parent_id")
            hit_child_id = metadata.get("hit_child_id")
            doc_id = metadata.get("doc_id")
            doc_name = metadata.get("doc_name")
            retrieval_rank = metadata.get("retrieval_rank")
            content = doc.page_content
            sources.append(
                {
                    "retrieval_type": retrieval_type,
                    "rerank_score": rerank_score,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "rrf_score": rrf_score,
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "hit_child_id": hit_child_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "retrieval_rank": retrieval_rank,
                    "content": content,
                }
            )
        return sources


rag_service = RAGService()
