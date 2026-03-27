import math
import re
from collections import Counter
from typing import Iterable

from app.utils.embedding_factory import EmbeddingFactory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationService:
    """
    SRD 评估框架（简化落地版）：
    1) Retrieval Quality: Recall@k / Precision@k / Hit@k
    2) Answer Quality: Exact Match / Token-level F1 / Semantic Similarity
    3) Relevance & Faithfulness: Citation Coverage / Citation Consistency
    4) Efficiency: latency / retrieval latency / generation latency
    """

    def __init__(self):
        self._embeddings = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _tokenize(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        if self._contains_cjk(text):
            chars = []
            for ch in text:
                if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", ch):
                    chars.append(ch)
            return chars
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text)
        return [tok for tok in cleaned.split() if tok]

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (n1 * n2)))

    def _semantic_similarity(self, pred: str, ref: str) -> float:
        pred = (pred or "").strip()
        ref = (ref or "").strip()
        if not pred or not ref:
            return 0.0
        try:
            if self._embeddings is None:
                self._embeddings = EmbeddingFactory.create_embeddings()
            v1 = self._embeddings.embed_query(pred)
            v2 = self._embeddings.embed_query(ref)
            return round(self._cosine_similarity(v1, v2), 4)
        except Exception as e:
            logger.warning(f"语义相似度计算失败，回退到token Jaccard: {e}")
            t1 = set(self._tokenize(pred))
            t2 = set(self._tokenize(ref))
            if not t1 or not t2:
                return 0.0
            return round(len(t1 & t2) / len(t1 | t2), 4)

    def _exact_match(self, pred: str, ref: str) -> float:
        return 1.0 if self._normalize_text(pred) == self._normalize_text(ref) else 0.0

    def _token_f1(self, pred: str, ref: str) -> float:
        p_tokens = self._tokenize(pred)
        r_tokens = self._tokenize(ref)
        if not p_tokens or not r_tokens:
            return 0.0
        c_pred, c_ref = Counter(p_tokens), Counter(r_tokens)
        overlap = sum((c_pred & c_ref).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(p_tokens)
        recall = overlap / len(r_tokens)
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"[。！？.!?\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _token_overlap_ratio(self, a: str, b: str) -> float:
        ta = set(self._tokenize(a))
        tb = set(self._tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta))

    def evaluate_faithfulness(self, answer: str, sources: list[dict] | None):
        source_contents = [
            (item or {}).get("content", "").strip() for item in (sources or [])
        ]
        source_contents = [s for s in source_contents if s]
        sentences = self._split_sentences(answer)
        if not sentences:
            return {"citation_coverage": 0.0, "citation_consistency": 0.0}
        if not source_contents:
            return {"citation_coverage": 0.0, "citation_consistency": 0.0}

        supported = 0
        for sent in sentences:
            if any(self._token_overlap_ratio(sent, src) >= 0.2 for src in source_contents):
                supported += 1
        coverage = supported / len(sentences)

        answer_text = "\n".join(sentences)
        consistent_sources = 0
        for src in source_contents:
            if self._token_overlap_ratio(src, answer_text) >= 0.1:
                consistent_sources += 1
        consistency = consistent_sources / len(source_contents)
        return {
            "citation_coverage": round(coverage, 4),
            "citation_consistency": round(consistency, 4),
        }

    def evaluate_retrieval(
        self,
        sources: list[dict] | None,
        gold_chunk_ids: Iterable[str] | None,
        k_values: Iterable[int] = (1, 3, 5),
    ):
        gold = [str(x) for x in (gold_chunk_ids or []) if str(x).strip()]
        if not gold:
            return {}
        retrieved = []
        for item in (sources or []):
            cid = (item or {}).get("chunk_id")
            if cid is not None:
                retrieved.append(str(cid))
        gold_set = set(gold)

        parsed_k_values = []
        for raw_k in (k_values or []):
            try:
                k = int(raw_k)
                if k > 0:
                    parsed_k_values.append(k)
            except Exception:
                continue
        if not parsed_k_values:
            parsed_k_values = [1, 3, 5]

        metrics = {}
        for k in sorted(set(parsed_k_values)):
            topk = retrieved[:k]
            hits = sum(1 for cid in topk if cid in gold_set)
            precision_k = hits / k
            recall_k = hits / len(gold_set)
            hit_k = 1.0 if hits > 0 else 0.0
            metrics[f"precision@{k}"] = round(precision_k, 4)
            metrics[f"recall@{k}"] = round(recall_k, 4)
            metrics[f"hit@{k}"] = round(hit_k, 4)
        return metrics

    def evaluate_answer_quality(self, answer: str, reference_answer: str | None):
        if not reference_answer:
            return {}
        em = self._exact_match(answer, reference_answer)
        f1 = self._token_f1(answer, reference_answer)
        sem = self._semantic_similarity(answer, reference_answer)
        return {
            "exact_match": round(em, 4),
            "token_f1": round(f1, 4),
            "semantic_similarity": round(sem, 4),
        }

    def evaluate_single_answer(
        self,
        *,
        answer: str,
        sources: list[dict] | None,
        reference_answer: str | None = None,
        gold_chunk_ids: Iterable[str] | None = None,
        k_values: Iterable[int] = (1, 3, 5),
        efficiency: dict | None = None,
    ):
        return {
            "retrieval_quality": self.evaluate_retrieval(sources, gold_chunk_ids, k_values),
            "answer_quality": self.evaluate_answer_quality(answer, reference_answer),
            "relevance_faithfulness": self.evaluate_faithfulness(answer, sources),
            "efficiency": efficiency or {},
        }

    def evaluate_triple_answers(
        self,
        triple_payload: dict,
        reference_answer: str | None = None,
        gold_chunk_ids: Iterable[str] | None = None,
        k_values: Iterable[int] = (1, 3, 5),
    ):
        if not triple_payload:
            return {}
        parsed_k_values = []
        for raw_k in (k_values or []):
            try:
                k = int(raw_k)
                if k > 0:
                    parsed_k_values.append(k)
            except Exception:
                continue
        if not parsed_k_values:
            parsed_k_values = [1, 3, 5]

        branch_scores = {}
        branch_details = {}
        for branch, data in triple_payload.items():
            answer = (data or {}).get("answer", "")
            sources = (data or {}).get("sources", [])
            efficiency = {
                "retrieval_elapsed_ms": (data or {}).get("retrieval_elapsed_ms", 0),
                "generation_elapsed_ms": (data or {}).get("generation_elapsed_ms", 0),
            }
            detail = self.evaluate_single_answer(
                answer=answer,
                sources=sources,
                reference_answer=reference_answer,
                gold_chunk_ids=gold_chunk_ids,
                k_values=parsed_k_values,
                efficiency=efficiency,
            )
            branch_details[branch] = detail

            aq = detail.get("answer_quality", {})
            rf = detail.get("relevance_faithfulness", {})
            rq = detail.get("retrieval_quality", {})
            best_hit = 0.0
            for k in parsed_k_values:
                best_hit = max(best_hit, float(rq.get(f"hit@{int(k)}", 0.0)))
            score = (
                0.45 * float(aq.get("semantic_similarity", 0.0))
                + 0.2 * float(aq.get("token_f1", 0.0))
                + 0.2 * float(rf.get("citation_coverage", 0.0))
                + 0.15 * best_hit
            )
            branch_scores[branch] = round(score, 4)

        recommended_branch = (
            max(branch_scores.items(), key=lambda x: x[1])[0] if branch_scores else None
        )
        return {
            "branch_metrics": branch_details,
            "branch_scores": branch_scores,
            "recommended_branch": recommended_branch,
        }


evaluation_service = EvaluationService()
