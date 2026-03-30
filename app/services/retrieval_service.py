from app.utils.logger import get_logger
from app.services.vector_service import vector_service
from app.services.settings_service import settings_service
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from langchain_core.documents import Document
from app.utils.rerank_factory import RerankFactory
from collections import defaultdict
import re
from difflib import SequenceMatcher
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from app.config import Config
import pickle
import os
from app.models.parent_chunk import ParentChunk
from app.utils.db import db_session
import atexit

logger = get_logger(__name__)


class RetrievalService:
    def __init__(self):
        self._reranker_init_attempted = False
        self.reranker = None
        self._keyword_index_cache = {}
        self._keyword_cache_lock = threading.Lock()
        # 复用线程池，避免每次 hybrid_search 都创建/销毁线程池
        self._hybrid_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="retrieval-hybrid"
        )

    def shutdown(self):
        executor = getattr(self, "_hybrid_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)
            self._hybrid_executor = None

    def _get_settings(self) -> dict:
        # 每次检索都拉取最新设置，避免服务初始化后配置过期
        return settings_service.get() or {}

    def _ensure_reranker(self, settings: dict):
        # 默认启用重排；如果模型不可用则自动降级
        if self._reranker_init_attempted:
            return
        self._reranker_init_attempted = True
        use_rerank = str(settings.get("use_rerank", "1")).lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not use_rerank:
            logger.info("配置已关闭重排序（use_rerank=false）")
            return
        try:
            self.reranker = RerankFactory.create_reranker(settings)
            logger.info("重排序模型初始化成功")
        except Exception as e:
            self.reranker = None
            logger.warning(f"重排序模型初始化失败，自动降级为无重排: {e}")

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _should_apply_rerank(self, query: str, rerank: bool, settings: dict) -> bool:
        """
        按语言自动开关重排：
        - 英文/非CJK：启用（当前模型更适配）
        - 中文/CJK：默认关闭（避免英文重排模型误排）
        可通过 settings 覆盖：
        - rerank_language_mode: auto | always_on | always_off
        """
        if not rerank or not self.reranker:
            return False
        mode = str(settings.get("rerank_language_mode", "auto")).strip().lower()
        if mode == "always_off":
            return False
        if mode == "always_on":
            return True
        # auto
        is_cjk = self._contains_cjk(query)
        return not is_cjk

    @staticmethod
    def _get_rerank_candidate_k(settings: dict, top_k: int) -> int:
        try:
            candidate = int(settings.get("rerank_candidate_k", 24))
        except Exception:
            candidate = 24
        candidate = max(top_k, candidate)
        return max(top_k + 3, min(candidate, 200))

    @staticmethod
    def _doc_uid(doc: Document) -> str:
        metadata = doc.metadata or {}
        return str(
            metadata.get("chunk_id")
            or metadata.get("id")
            or f"{metadata.get('doc_id', '')}:{hash(doc.page_content)}"
        )

    def _dedup_docs(self, docs):
        seen = set()
        result = []
        for doc in docs or []:
            uid = self._doc_uid(doc)
            if uid in seen:
                continue
            seen.add(uid)
            result.append(doc)
        return result

    @staticmethod
    def _normalize_for_overlap(text: str) -> str:
        return re.sub(r"\s+", "", text or "")[:1200]

    @classmethod
    def _has_overlap(cls, a: str, b: str, ratio: float = 0.9) -> bool:
        if not a or not b:
            return False
        ta = cls._normalize_for_overlap(a)
        tb = cls._normalize_for_overlap(b)
        if not ta or not tb:
            return False
        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if len(shorter) >= 100 and shorter in longer:
            return True
        sim = SequenceMatcher(None, ta, tb).ratio()
        return sim >= ratio

    def _diversify_docs(self, docs, top_k: int):
        """
        轻量多样化：
        1) 尽量避免同文档连续霸榜（按 doc_id 轮转）
        2) 过滤高度近似片段，减少冗余上下文
        """
        if not docs:
            return []
        target_k = max(1, top_k)
        by_doc = defaultdict(list)
        for doc in docs:
            doc_id = str((doc.metadata or {}).get("doc_id") or "__unknown__")
            by_doc[doc_id].append(doc)

        ordered = []
        # 轮转抽取，提升跨文档覆盖
        max_pool = max(target_k * 3, target_k)
        while len(ordered) < max_pool and by_doc:
            remove_ids = []
            for doc_id, arr in by_doc.items():
                if not arr:
                    remove_ids.append(doc_id)
                    continue
                ordered.append(arr.pop(0))
                if len(ordered) >= max_pool:
                    break
                if not arr:
                    remove_ids.append(doc_id)
            for doc_id in remove_ids:
                by_doc.pop(doc_id, None)

        # 近似去重
        final_docs = []
        for doc in ordered:
            if any(
                self._has_overlap(doc.page_content or "", d.page_content or "")
                for d in final_docs
            ):
                continue
            final_docs.append(doc)
            if len(final_docs) >= target_k:
                break
        # 如果近似去重后过少，回填候选，避免来源数被误压缩到 1 条
        if len(final_docs) < target_k:
            seen = {self._doc_uid(d) for d in final_docs}
            for doc in ordered:
                uid = self._doc_uid(doc)
                if uid in seen:
                    continue
                seen.add(uid)
                final_docs.append(doc)
                if len(final_docs) >= target_k:
                    break
        return final_docs

    @staticmethod
    def _attach_retrieval_rank(docs):
        for idx, doc in enumerate(docs or [], start=1):
            metadata = doc.metadata or {}
            metadata["retrieval_rank"] = idx
            doc.metadata = metadata
        return docs

    @staticmethod
    def _attach_retrieval_debug(docs, debug: dict):
        for doc in docs or []:
            metadata = doc.metadata or {}
            metadata["retrieval_debug"] = debug
            doc.metadata = metadata
        return docs

    def _emit_retrieval_trace(self, mode: str, collection_name: str, query: str, debug: dict):
        payload = {
            "event": "retrieval_trace",
            "mode": mode,
            "collection": collection_name,
            "query_len": len(query or ""),
            "query_is_cjk": self._contains_cjk(query or ""),
            "debug": debug,
        }
        logger.info(f"RETRIEVAL_TRACE {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    def _promote_children_to_parents(self, docs, target_parent_k: int):
        """
        子块命中 -> 父块上下文：
        按 child 命中顺序聚合 parent_id，通过 parent_id 反查父块正文。
        """
        if not docs:
            return []
        parent_ids = []
        seen_for_query = set()
        for doc in docs:
            parent_id = (doc.metadata or {}).get("parent_id")
            if not parent_id:
                continue
            parent_id = str(parent_id)
            if parent_id in seen_for_query:
                continue
            seen_for_query.add(parent_id)
            parent_ids.append(parent_id)

        parent_content_map = {}
        if parent_ids:
            with db_session() as session:
                rows = (
                    session.query(ParentChunk.parent_id, ParentChunk.content)
                    .filter(ParentChunk.parent_id.in_(parent_ids))
                    .all()
                )
            parent_content_map = {str(pid): content for pid, content in rows}

        parent_docs = []
        parent_seen = set()
        for doc in docs:
            metadata = doc.metadata or {}
            parent_id = metadata.get("parent_id")
            if not parent_id:
                # 兼容老索引（无父子结构）：直接按当前块返回
                parent_docs.append(doc)
                if len(parent_docs) >= target_parent_k:
                    break
                continue
            parent_id = str(parent_id)
            parent_content = parent_content_map.get(parent_id)
            if not parent_content:
                # 若父块映射缺失，回退到当前子块，避免召回为空
                parent_docs.append(doc)
                if len(parent_docs) >= target_parent_k:
                    break
                continue
            if parent_id in parent_seen:
                continue
            parent_seen.add(parent_id)
            new_meta = dict(metadata)
            new_meta["node_type"] = "parent_context"
            new_meta["id"] = parent_id
            new_meta["chunk_id"] = parent_id
            new_meta["hit_child_id"] = metadata.get("chunk_id") or metadata.get("id")
            promoted = Document(page_content=parent_content, metadata=new_meta)
            parent_docs.append(promoted)
            if len(parent_docs) >= target_parent_k:
                break
        return parent_docs

    @staticmethod
    def _to_bool(value, default=True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _to_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _build_keyword_index(self, vector_store):
        results = vector_store._collection.get(include=["documents", "metadatas"])
        ids = results.get("ids") or []
        chroma_documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        langchain_docs = []
        for _id, chroma_document, meta in zip(ids, chroma_documents, metadatas):
            meta = meta or {}
            node_type = str(meta.get("node_type") or "")
            if node_type and node_type != "child":
                continue
            langchain_docs.append(Document(page_content=chroma_document, metadata=meta))
        if not langchain_docs:
            return {
                "bm25": None,
                "docs": [],
                "doc_count": 0,
                "total_count": len(ids),
                "built_at": time.monotonic(),
            }
        tokenized_docs = [self._tokenize_for_keyword(doc.page_content) for doc in langchain_docs]
        return {
            "bm25": BM25Okapi(tokenized_docs),
            "docs": langchain_docs,
            "doc_count": len(langchain_docs),
            "total_count": len(ids),
            "built_at": time.monotonic(),
        }

    @staticmethod
    def _keyword_index_file_path(collection_name: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(collection_name or "default"))
        return Path(Config.BASE_DIR) / "storages" / "keyword_index" / f"{safe_name}.pkl"

    def _load_keyword_index_from_disk(self, collection_name: str):
        index_path = self._keyword_index_file_path(collection_name)
        if not index_path.exists():
            return {
                "bm25": None,
                "docs": [],
                "doc_count": 0,
                "total_count": 0,
                "built_at": time.monotonic(),
                "source_mtime": None,
            }
        with open(index_path, "rb") as f:
            payload = pickle.load(f) or {}
        docs_payload = payload.get("docs") or []
        tokenized_docs = payload.get("tokenized_docs") or []
        langchain_docs = []
        for item in docs_payload:
            if not isinstance(item, dict):
                continue
            langchain_docs.append(
                Document(
                    page_content=item.get("page_content") or "",
                    metadata=item.get("metadata") or {},
                )
            )
        # 为保证分词规则升级后立即生效，加载时统一按当前 tokenizer 重算
        tokenized_docs = [self._tokenize_for_keyword(doc.page_content) for doc in langchain_docs]
        bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None
        return {
            "bm25": bm25,
            "docs": langchain_docs,
            "doc_count": len(langchain_docs),
            "total_count": int(payload.get("total_count", len(langchain_docs))),
            "built_at": time.monotonic(),
            "source_mtime": os.path.getmtime(index_path),
        }

    def _get_keyword_index(self, collection_name: str, settings: dict):
        ttl_sec = max(0, min(self._to_int(settings.get("keyword_index_ttl_sec", 300), 300), 3600))
        now = time.monotonic()
        source_mtime = None
        index_path = self._keyword_index_file_path(collection_name)
        if index_path.exists():
            try:
                source_mtime = os.path.getmtime(index_path)
            except OSError:
                source_mtime = None
        with self._keyword_cache_lock:
            cached = self._keyword_index_cache.get(collection_name)
            if (
                cached
                and cached.get("source_mtime") == source_mtime
                and (now - cached.get("built_at", 0.0)) <= ttl_sec
            ):
                return cached
        built = self._load_keyword_index_from_disk(collection_name)
        with self._keyword_cache_lock:
            self._keyword_index_cache[collection_name] = built
        return built

    def vector_search(self, collection_name, query, rerank=True, settings: dict | None = None):
        settings = settings or self._get_settings()
        self._ensure_reranker(settings)
        vector_store = vector_service.get_or_create_collection(collection_name)
        top_k = self._to_int(settings.get("top_k", "5"), 5)
        vector_threshold = float(settings.get("vector_threshold", "0.1"))
        # 把向量相似度的阈值限定在0到1之间
        vector_threshold = max(0.0, min(vector_threshold, 1.0))
        rerank_candidate_k = self._get_rerank_candidate_k(settings, top_k)
        candidate_k = max(top_k * 6, rerank_candidate_k)
        debug = {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "vector_threshold": vector_threshold,
            "rerank_candidate_k": rerank_candidate_k,
        }
        # 以相似度得分的方式检索，返回结果，这里先扩大k以以便后缀过滤
        results = vector_store.similarity_search_with_score(query=query, k=candidate_k)
        debug["raw_hits"] = len(results)
        docs_with_scores = []
        for doc, distance in results:
            # distance其实是一个相似度的距离，一般来距离越近，越小越相似
            # 对分数进行归一化处理并加入元数据，score取值范围 0-1，越大越相似
            # distance是0，score就是1 就是最相似
            # distance正无穷大，score无限接近0，最不相似
            # score越大越相似
            vector_score = 1.0 / (1.0 + float(distance))
            doc.metadata["vector_score"] = vector_score
            doc.metadata["retrieval_type"] = "vector"
            docs_with_scores.append((doc, vector_score))
        # 按相似度从高到底排序
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        # 根据阈值过滤掉低于阈值的文档
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores if score >= vector_threshold
        ]
        debug["after_threshold"] = len(filtered_docs)
        docs = [doc for doc, _ in filtered_docs]
        if len(docs) < top_k:
            # 阈值过严时兜底，避免召回过低
            debug["fallback_used"] = True
            docs = [doc for doc, _ in docs_with_scores[:candidate_k]]
        else:
            debug["fallback_used"] = False
        docs = self._dedup_docs(docs)
        debug["after_dedup"] = len(docs)
        apply_rerank = self._should_apply_rerank(query, rerank, settings)
        debug["rerank_applied"] = bool(apply_rerank)
        if apply_rerank:
            docs = self._apply_rerank(query, docs, rerank_candidate_k)
        debug["after_rerank"] = len(docs)
        parent_candidate_k = max(top_k * 3, rerank_candidate_k)
        docs = self._promote_children_to_parents(docs, parent_candidate_k)
        debug["after_parent_promote"] = len(docs)
        docs = self._diversify_docs(docs, top_k)
        debug["after_diversify"] = len(docs)
        docs = self._attach_retrieval_rank(docs)
        docs = self._attach_retrieval_debug(docs, debug)
        debug["final_count"] = len(docs)
        self._emit_retrieval_trace("vector", collection_name, query, debug)
        logger.info(f"向量检索 ：检索到{len(docs)}个文档")
        return docs

    def _apply_rerank(self, query, docs, top_k):
        if not self.reranker or not docs:
            if not self.reranker:
                logger.info(f"文档重排序实例不存在，跳过重排序")
            else:
                logger.info(f"检索到的文档为空")
            return docs
        try:
            reranked = self.reranker.rerank(query, docs, top_k=top_k)
            for doc, rerank_score in reranked:
                doc.metadata["rerank_score"] = rerank_score
            logger.info(f"已经应用了文档重排序:{len(reranked)}个文档进行了重排序")
            return [doc for doc, _ in reranked]
        except Exception as e:
            logger.error(f"应用重排序出现错误:{str(e)}")
            return docs

    def _tokenize_for_keyword(self, text: str):
        """
        关键词检索分词：
        - 英文：统一小写 + 正则抽取单词
        - CJK：兼容使用 jieba（兜底）
        """
        text = (text or "").strip()
        if not text:
            return []
        return self._bm25_tokenize(text)

    def _bm25_tokenize(self, text: str) -> list[str]:
        text = (text or "").lower()
        if not text:
            return []
        if self._contains_cjk(text):
            cjk_stopwords = {
                "的",
                "了",
                "在",
                "是",
                "和",
                "有",
                "与",
                "对",
                "等",
                "为",
                "也",
                "就",
                "都",
                "要",
                "可以",
                "会",
                "能",
                "而",
                "及",
                "或",
            }
            words = jieba.lcut(text)
            return [
                w.strip()
                for w in words
                if len(w.strip()) > 1 and w.strip() not in cjk_stopwords
            ]
        return re.findall(r"\b\w+\b", text)

    def keyword_search(self, collection_name, query, rerank=True, settings: dict | None = None):
        settings = settings or self._get_settings()
        self._ensure_reranker(settings)
        top_k = self._to_int(settings.get("top_k", "5"), 5)
        keyword_threshold = float(settings.get("keyword_threshold", "0.1"))
        rerank_candidate_k = self._get_rerank_candidate_k(settings, top_k)
        # 把关键字相似度的阈值限定在0到1之间
        keyword_threshold = max(0.0, min(keyword_threshold, 1.0))
        keyword_index = self._get_keyword_index(collection_name, settings)
        langchain_docs = keyword_index.get("docs") or []
        if not langchain_docs:
            self._emit_retrieval_trace(
                "keyword",
                collection_name,
                query,
                {
                    "top_k": top_k,
                    "candidate_k": 0,
                    "keyword_threshold": keyword_threshold,
                    "rerank_candidate_k": rerank_candidate_k,
                    "raw_hits": 0,
                    "after_threshold": 0,
                    "after_dedup": 0,
                    "after_rerank": 0,
                    "after_parent_promote": 0,
                    "after_diversify": 0,
                    "final_count": 0,
                    "fallback_used": False,
                    "rerank_applied": False,
                },
            )
            return []
        bm25 = keyword_index.get("bm25")
        if bm25 is None:
            return []
        # 对查询语句进行中文分词
        query_tokens = self._tokenize_for_keyword(query)
        # 获取每个文档与查询的BM25分数
        scores = bm25.get_scores(query_tokens)
        # 计算分数最大值，用于归一化分数到[0,1]之间
        max_score = (
            float(np.max(scores)) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        )
        # 归一化BM25分数 [1,2,3,4,5]  /5 = [0.2,,,,1]
        normalized_scores = scores / max_score if max_score > 0 else scores
        # 取分数最高的top_k*3个索引,以便于后续过滤
        candidate_k = max(top_k * 6, rerank_candidate_k)
        debug = {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "keyword_threshold": keyword_threshold,
            "rerank_candidate_k": rerank_candidate_k,
        }
        top_indices = np.argsort(normalized_scores)[::-1][:candidate_k]
        debug["raw_hits"] = len(top_indices)
        # 初始化结果列表
        docs_with_scores = []
        # 遍历候选索引列表
        for idx in top_indices:
            normalized_score = float(normalized_scores[idx])
            normalized_score = max(0.0, min(1.0, normalized_score))
            # 只保留分数高于阈值的文档
            if normalized_score >= keyword_threshold:
                # 取出对应的文档
                doc = langchain_docs[idx]
                doc.metadata["keyword_score"] = normalized_score
                doc.metadata["retrieval_type"] = "keyword"
                docs_with_scores.append((doc, normalized_score))
        debug["after_threshold"] = len(docs_with_scores)
        # 按相似度从高到底排序
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in docs_with_scores]
        if len(docs) < top_k:
            debug["fallback_used"] = True
            docs = [langchain_docs[idx] for idx in top_indices]
            for doc in docs:
                doc.metadata["retrieval_type"] = "keyword"
                doc.metadata.setdefault("keyword_score", 0.0)
        else:
            debug["fallback_used"] = False
        docs = self._dedup_docs(docs)
        debug["after_dedup"] = len(docs)
        apply_rerank = self._should_apply_rerank(query, rerank, settings)
        debug["rerank_applied"] = bool(apply_rerank)
        if apply_rerank:
            docs = self._apply_rerank(query, docs, rerank_candidate_k)
        debug["after_rerank"] = len(docs)
        parent_candidate_k = max(top_k * 3, rerank_candidate_k)
        docs = self._promote_children_to_parents(docs, parent_candidate_k)
        debug["after_parent_promote"] = len(docs)
        docs = self._diversify_docs(docs, top_k)
        debug["after_diversify"] = len(docs)
        docs = self._attach_retrieval_rank(docs)
        docs = self._attach_retrieval_debug(docs, debug)
        debug["final_count"] = len(docs)
        self._emit_retrieval_trace("keyword", collection_name, query, debug)
        logger.info(f"BM25关键词本文检索 ：检索到{len(docs)}个文档")
        return docs

    def hybrid_search(self, collection_name, query, rrf_k=60, settings: dict | None = None):
        """
        融合检索 使用RRF融合向量检索和全文检索
        """
        # 调用向量检索方法，得到向量检索结果
        settings = settings or self._get_settings()
        self._ensure_reranker(settings)
        top_k = self._to_int(settings.get("top_k", "5"), 5)
        rerank_candidate_k = self._get_rerank_candidate_k(settings, top_k)
        candidate_top_k = max(top_k * 2, rerank_candidate_k)
        debug = {
            "top_k": top_k,
            "candidate_top_k": candidate_top_k,
            "rrf_k": rrf_k,
            "rerank_candidate_k": rerank_candidate_k,
        }

        vector_future = self._hybrid_executor.submit(
            self.vector_search,
            collection_name=collection_name,
            query=query,
            rerank=False,
            settings=settings,
        )
        keyword_future = self._hybrid_executor.submit(
            self.keyword_search,
            collection_name=collection_name,
            query=query,
            rerank=False,
            settings=settings,
        )
        vector_results = vector_future.result()
        keyword_results = keyword_future.result()
        debug["vector_in"] = len(vector_results)
        debug["keyword_in"] = len(keyword_results)
        # 创建字典用于存储文本及其排名信息
        doc_ranks = {}
        # 遍历向量检索结果，记录排名及分数
        for rank, doc in enumerate(vector_results, start=1):
            # 其实是上传的文档document,用split分割后得到的文本分块存到向量库中的分块ID
            chunk_id = str(
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or self._doc_uid(doc)
            )
            # 如果文档ID不在字典中，则进行初始化
            if chunk_id not in doc_ranks:
                doc_ranks[chunk_id] = {"doc": doc}
            # 记录此doc_id对应的文档在向量结果列表中的排名
            doc_ranks[chunk_id]["vector_rank"] = rank
            # 再记录一下向量结果中此文档的对应的分数
            doc_ranks[chunk_id]["vector_score"] = doc.metadata.get("vector_score", 0)

            # 遍历向量检索结果，记录排名及分数
        for rank, doc in enumerate(keyword_results, start=1):
            # 其实是上传的文档document,用split分割后得到的文本分块存到向量库中的分块ID
            chunk_id = str(
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
                or self._doc_uid(doc)
            )
            # 如果文档ID不在字典中，则进行初始化
            if chunk_id not in doc_ranks:
                doc_ranks[chunk_id] = {"doc": doc}
            # 记录此doc_id对应的文档在向量结果列表中的排名
            doc_ranks[chunk_id]["keyword_rank"] = rank
            # 再记录一下向量结果中此文档的对应的分数
            doc_ranks[chunk_id]["keyword_score"] = doc.metadata.get("keyword_score", 0)
        # 从设置中读取向量权重，默认值为0.3
        vector_weight = float(settings.get("vector_weight", "0.7"))
        # 把关键字相似度的阈值限定在0到1之间
        vector_weight = max(0.0, min(vector_weight, 1.0))
        # 计算出关键词的权重
        keyword_weight = 1 - vector_weight
        # 遍历所有的文档，计算RRF融合分数
        for chunk_id, rank_info in doc_ranks.items():
            # 获取向量排名
            vector_rank = rank_info.get("vector_rank", rrf_k + 1)
            # 获取关键词排名
            keyword_rank = rank_info.get("keyword_rank", rrf_k + 1)
            # 初始化RRF分数
            rrf_score = 0.0
            # 融合分数 = 向量检索权重 / (rrk平滑常数 + 向量检索出结果的排名) + 关键字检索权重 / (rrk平滑常数 + 关键字检索出的结果的排名)
            rank_rrf = 0.0
            rank_rrf += vector_weight / (rrf_k + vector_rank)
            rank_rrf += keyword_weight / (rrf_k + keyword_rank)
            score_blend = (
                vector_weight * rank_info.get("vector_score", 0.0)
                + keyword_weight * rank_info.get("keyword_score", 0.0)
            )
            rrf_score = 0.6 * rank_rrf + 0.4 * score_blend
            doc_ranks[chunk_id]["rrf_score"] = rrf_score
        # 组装所有的文档以及其排名信息
        combined_results = [
            (chunk_id, rank_info) for chunk_id, rank_info in doc_ranks.items()
        ]
        debug["rrf_candidates"] = len(combined_results)
        # 最终的排序依据是RRF分数，从高到底进行排序
        combined_results.sort(key=lambda x: x[1].get("rrf_score", 0), reverse=True)
        docs = []
        for chunk_id, rank_info in combined_results[:candidate_top_k]:
            doc = rank_info["doc"]
            doc.metadata["vector_score"] = rank_info.get("vector_score", 0)
            doc.metadata["keyword_score"] = rank_info.get("keyword_score", 0)
            doc.metadata["rrf_score"] = rank_info.get("rrf_score", 0)
            doc.metadata["retrieval_type"] = "hybrid"
            docs.append(doc)
        debug["after_rrf_top"] = len(docs)
        logger.info(f"混合检索(RRF):检索到{len(docs)}个文档")
        docs = self._dedup_docs(docs)
        debug["after_dedup"] = len(docs)
        apply_rerank = self._should_apply_rerank(query, True, settings)
        debug["rerank_applied"] = bool(apply_rerank)
        if apply_rerank:
            docs = self._apply_rerank(query, docs, rerank_candidate_k)
        debug["after_rerank"] = len(docs)
        parent_candidate_k = max(top_k * 3, rerank_candidate_k)
        docs = self._promote_children_to_parents(docs, parent_candidate_k)
        debug["after_parent_promote"] = len(docs)
        docs = self._diversify_docs(docs, top_k)
        debug["after_diversify"] = len(docs)
        docs = self._attach_retrieval_rank(docs)
        docs = docs[:top_k]
        docs = self._attach_retrieval_debug(docs, debug)
        debug["final_count"] = len(docs)
        self._emit_retrieval_trace("hybrid", collection_name, query, debug)
        return docs


retrieval_service = RetrievalService()
atexit.register(retrieval_service.shutdown)
