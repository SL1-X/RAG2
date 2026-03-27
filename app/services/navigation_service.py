from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import re

from app.models.document import Document as DocumentModel
from app.services.base_service import BaseService
from app.services.vector_service import vector_service


_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

_DATE_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"),
    re.compile(
        rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+(19|20)\d{{2}}\b", re.IGNORECASE
    ),
]

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "each",
    "from",
    "have",
    "into",
    "just",
    "many",
    "more",
    "most",
    "other",
    "over",
    "same",
    "some",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
    "from",
    "were",
    "been",
    "into",
    "used",
    "using",
    "will",
    "shall",
    "should",
    "than",
    "then",
    "them",
}


@dataclass
class _DocChunk:
    doc_id: str
    doc_name: str
    chunk_id: str
    chunk_index: int
    content: str


class NavigationService(BaseService[DocumentModel]):
    def build_knowledgebase_navigation(
        self,
        kb_id: str,
        *,
        max_docs: int = 50,
        max_chunks_per_doc: int = 1500,
    ) -> dict:
        docs = self._list_completed_docs(kb_id, max_docs=max_docs)
        if not docs:
            return {
                "toc": [],
                "timeline": [],
                "themes": [],
                "suggested_questions": [],
                "stats": {"docs": 0, "chunks": 0},
            }

        all_doc_chunks: list[_DocChunk] = []
        per_doc_chunks: dict[str, list[_DocChunk]] = {}
        for doc in docs:
            chunks = self._load_doc_chunks(
                kb_id=kb_id,
                doc_id=doc.id,
                doc_name=doc.name or "未知文档",
                max_k=max(max_chunks_per_doc, int(doc.chunk_count or 0) + 20),
            )
            if not chunks:
                continue
            all_doc_chunks.extend(chunks)
            per_doc_chunks[doc.id] = chunks

        toc = self._build_toc(per_doc_chunks)
        timeline = self._build_timeline(all_doc_chunks)
        themes = self._build_themes(all_doc_chunks)
        suggested_questions = self._build_questions(toc, timeline, themes, docs)

        return {
            "toc": toc,
            "timeline": timeline,
            "themes": themes,
            "suggested_questions": suggested_questions,
            "stats": {"docs": len(per_doc_chunks), "chunks": len(all_doc_chunks)},
        }

    def _list_completed_docs(self, kb_id: str, *, max_docs: int) -> list[DocumentModel]:
        with self.session() as session:
            return (
                session.query(DocumentModel)
                .filter(
                    DocumentModel.kb_id == kb_id,
                    DocumentModel.status == "completed",
                )
                .order_by(DocumentModel.updated_at.desc())
                .limit(max_docs)
                .all()
            )

    def _load_doc_chunks(
        self, *, kb_id: str, doc_id: str, doc_name: str, max_k: int
    ) -> list[_DocChunk]:
        collection_name = f"kb_{kb_id}"
        try:
            results = vector_service.similarity_search_with_score(
                collection_name=collection_name,
                query="",
                k=max_k,
                filter={"doc_id": doc_id},
            )
        except Exception as e:
            self.logger.warning(f"加载导航分块失败，doc_id={doc_id}: {e}")
            return []
        chunks = []
        for item, _score in results or []:
            md = item.metadata or {}
            chunks.append(
                _DocChunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunk_id=str(md.get("chunk_id") or md.get("id") or ""),
                    chunk_index=int(md.get("chunk_index") or 0),
                    content=(item.page_content or "").strip(),
                )
            )
        chunks.sort(key=lambda x: x.chunk_index)
        return chunks

    def _build_toc(self, per_doc_chunks: dict[str, list[_DocChunk]]) -> list[dict]:
        toc = []
        for doc_id, chunks in per_doc_chunks.items():
            if not chunks:
                continue
            doc_name = chunks[0].doc_name
            seed_text = "\n".join(c.content for c in chunks[:8])
            headings = self._extract_headings(seed_text)
            if not headings:
                headings = self._fallback_outline(seed_text)
            toc.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "headings": headings[:10],
                }
            )
        return toc

    @staticmethod
    def _extract_headings(text: str) -> list[str]:
        lines = [ln.strip() for ln in (text or "").splitlines()]
        result = []
        seen = set()
        patterns = [
            re.compile(r"^#{1,6}\s+(.+)$"),
            re.compile(r"^\d+(?:\.\d+)*\s+(.+)$"),
            re.compile(r"^[A-Z][A-Za-z0-9 ,:/()\-]{6,100}$"),
        ]
        for line in lines:
            if len(line) < 6 or len(line) > 120:
                continue
            if line.endswith((".", "!", "?", ";", ":")):
                continue
            candidate = None
            for p in patterns:
                m = p.match(line)
                if m:
                    candidate = (m.group(1) if m.groups() else line).strip()
                    break
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result

    @staticmethod
    def _fallback_outline(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text or "")
        output = []
        for s in sentences:
            s = " ".join((s or "").split()).strip()
            if len(s) < 25:
                continue
            output.append(s[:90] + ("..." if len(s) > 90 else ""))
            if len(output) >= 6:
                break
        return output

    def _build_timeline(self, chunks: list[_DocChunk]) -> list[dict]:
        events = []
        for chunk in chunks:
            for match in self._find_dates(chunk.content):
                sentence = self._extract_sentence(chunk.content, match["raw"])
                if not sentence:
                    continue
                events.append(
                    {
                        "date": match["date"],
                        "date_display": match["raw"],
                        "summary": sentence[:180] + ("..." if len(sentence) > 180 else ""),
                        "doc_id": chunk.doc_id,
                        "doc_name": chunk.doc_name,
                        "chunk_id": chunk.chunk_id,
                    }
                )
        dedup = {}
        for event in events:
            key = (event["date"], event["summary"][:120], event["doc_id"])
            dedup[key] = event
        ordered = sorted(dedup.values(), key=lambda x: x["date"])
        return ordered[:24]

    def _find_dates(self, text: str) -> list[dict]:
        found = []
        for pattern in _DATE_PATTERNS:
            for m in pattern.finditer(text or ""):
                raw = m.group(0)
                parsed = self._parse_date(raw)
                if not parsed:
                    continue
                found.append({"raw": raw, "date": parsed.strftime("%Y-%m-%d")})
        return found

    @staticmethod
    def _parse_date(raw: str) -> datetime | None:
        raw = (raw or "").strip()
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_sentence(text: str, needle: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        for s in sentences:
            if needle in s:
                return " ".join(s.split()).strip()
        return ""

    def _build_themes(self, chunks: list[_DocChunk]) -> list[dict]:
        tokens = []
        for chunk in chunks:
            tokens.extend(self._tokenize(chunk.content))
        counter = Counter(tokens)
        top_terms = [t for t, c in counter.most_common(16) if c >= 3][:8]
        themes = []
        used = set()
        for term in top_terms:
            if term in used:
                continue
            snippets = []
            keyword_pool = [term]
            co_terms = self._co_terms(term, counter)
            keyword_pool.extend(co_terms[:3])
            used.update(keyword_pool)
            for chunk in chunks:
                txt = chunk.content.lower()
                if term not in txt:
                    continue
                snippet = self._extract_snippet(chunk.content, term)
                if snippet:
                    snippets.append(
                        {
                            "text": snippet,
                            "doc_id": chunk.doc_id,
                            "doc_name": chunk.doc_name,
                            "chunk_id": chunk.chunk_id,
                        }
                    )
                if len(snippets) >= 3:
                    break
            if not snippets:
                continue
            themes.append(
                {
                    "title": term.title(),
                    "keywords": keyword_pool,
                    "snippets": snippets,
                }
            )
            if len(themes) >= 6:
                break
        return themes

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        words = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
        return [w for w in words if w not in _STOPWORDS]

    @staticmethod
    def _co_terms(term: str, counter: Counter) -> list[str]:
        related = []
        term_prefix = term[:4]
        for word, _freq in counter.most_common(64):
            if word == term:
                continue
            if word.startswith(term_prefix):
                continue
            related.append(word)
            if len(related) >= 6:
                break
        return related

    @staticmethod
    def _extract_snippet(text: str, keyword: str, max_len: int = 170) -> str:
        src = " ".join((text or "").split())
        if not src:
            return ""
        low = src.lower()
        idx = low.find(keyword.lower())
        if idx < 0:
            return ""
        left = max(0, idx - max_len // 2)
        right = min(len(src), idx + max_len // 2)
        snippet = src[left:right].strip()
        if left > 0:
            snippet = "..." + snippet
        if right < len(src):
            snippet = snippet + "..."
        return snippet

    @staticmethod
    def _build_questions(toc, timeline, themes, docs) -> list[str]:
        questions = []
        for item in toc[:3]:
            heading = (item.get("headings") or [""])[0]
            if heading:
                questions.append(f"What are the key takeaways in '{heading}'?")
        for theme in themes[:3]:
            title = theme.get("title")
            if title:
                questions.append(f"How does {title} evolve across the documents?")
                questions.append(f"What evidence supports the main claims about {title}?")
        for event in timeline[:2]:
            date_display = event.get("date_display")
            if date_display:
                questions.append(
                    f"What changed around {date_display} and why is it important?"
                )
        if len(docs) >= 2:
            questions.append("What are the major similarities and differences across documents?")
        deduped = []
        seen = set()
        for q in questions:
            key = q.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q.strip())
        return deduped[:10]


navigation_service = NavigationService()
