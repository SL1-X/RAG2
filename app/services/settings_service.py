from app.models.settings import Settings
from app.services.base_service import BaseService
from app.config import Config
from pathlib import Path
import json


class SettingsService(BaseService[Settings]):
    _EXTRA_SETTINGS_FILE = (
        Path(__file__).resolve().parents[2] / "storages" / "retrieval_tuning.json"
    )
    _MODULE_LLM_DEFAULTS = {
        "rewrite_llm_provider": "deepseek",
        "rewrite_llm_model_name": Config.DEEPSEEK_CHAT_MODEL,
        "rewrite_llm_api_key": Config.DEEPSEEK_API_KEY,
        "rewrite_llm_base_url": Config.DEEPSEEK_BASE_URL,
        "rewrite_llm_temperature": 0.0,
        "rewrite_llm_max_tokens": 256,
        "rewrite_llm_fallback_provider": "gemini",
        "rewrite_llm_fallback_model_name": Config.GEMINI_CHAT_MODEL,
        "rewrite_llm_fallback_api_key": Config.GEMINI_API_KEY,
        "rewrite_llm_fallback_base_url": Config.GEMINI_BASE_URL,
        "rag_llm_provider": "gemini",
        "rag_llm_model_name": Config.GEMINI_CHAT_MODEL,
        "rag_llm_api_key": Config.GEMINI_API_KEY,
        "rag_llm_base_url": Config.GEMINI_BASE_URL,
        "rag_llm_temperature": 0.7,
        "rag_llm_max_tokens": 768,
        "rag_llm_fallback_provider": "deepseek",
        "rag_llm_fallback_model_name": Config.DEEPSEEK_CHAT_MODEL,
        "rag_llm_fallback_api_key": Config.DEEPSEEK_API_KEY,
        "rag_llm_fallback_base_url": Config.DEEPSEEK_BASE_URL,
        "chat_llm_provider": "gemini",
        "chat_llm_model_name": Config.GEMINI_CHAT_MODEL,
        "chat_llm_api_key": Config.GEMINI_API_KEY,
        "chat_llm_base_url": Config.GEMINI_BASE_URL,
        "chat_llm_temperature": 0.7,
        "chat_llm_max_tokens": 1024,
        "chat_llm_fallback_provider": "deepseek",
        "chat_llm_fallback_model_name": Config.DEEPSEEK_CHAT_MODEL,
        "chat_llm_fallback_api_key": Config.DEEPSEEK_API_KEY,
        "chat_llm_fallback_base_url": Config.DEEPSEEK_BASE_URL,
    }
    _MODULE_LLM_NUMERIC_KEYS = {
        "rewrite_llm_temperature",
        "rewrite_llm_max_tokens",
        "rag_llm_temperature",
        "rag_llm_max_tokens",
        "chat_llm_temperature",
        "chat_llm_max_tokens",
    }
    _RETRIEVAL_OVERRIDE_DEFAULTS = {
        "retrieval_mode": "hybrid",
        "vector_threshold": 0.08,
        "keyword_threshold": 0.0,
        "vector_weight": 0.55,
        "top_k": 6,
        "enable_query_rewrite": True,
        "rewrite_only_when_needed": True,
        "keyword_index_ttl_sec": 300,
    }

    @staticmethod
    def _normalize_provider_name(value: str) -> str:
        provider = str(value or "").strip().lower()
        if provider == "openai":
            return "gemini"
        return provider

    @staticmethod
    def _as_bool(value, default=True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _normalize_provider_fields(self, payload: dict) -> dict:
        provider_keys = [
            "embedding_provider",
            "llm_provider",
            "rewrite_llm_provider",
            "rewrite_llm_fallback_provider",
            "rag_llm_provider",
            "rag_llm_fallback_provider",
            "chat_llm_provider",
            "chat_llm_fallback_provider",
        ]
        for key in provider_keys:
            if key in payload:
                payload[key] = self._normalize_provider_name(payload.get(key))
        return payload

    def _normalize_model_fields(self, payload: dict) -> dict:
        model_provider_pairs = [
            ("llm_provider", "llm_model_name"),
            ("rewrite_llm_provider", "rewrite_llm_model_name"),
            ("rewrite_llm_fallback_provider", "rewrite_llm_fallback_model_name"),
            ("rag_llm_provider", "rag_llm_model_name"),
            ("rag_llm_fallback_provider", "rag_llm_fallback_model_name"),
            ("chat_llm_provider", "chat_llm_model_name"),
            ("chat_llm_fallback_provider", "chat_llm_fallback_model_name"),
        ]
        for provider_key, model_key in model_provider_pairs:
            provider = self._normalize_provider_name(payload.get(provider_key))
            model_name = str(payload.get(model_key) or "").strip()
            if provider == "gemini" and model_name.lower().startswith(
                ("gpt-", "o1", "o3", "text-")
            ):
                payload[model_key] = Config.GEMINI_CHAT_MODEL
        return payload

    def _read_extra_settings(self) -> dict:
        file_path = self._EXTRA_SETTINGS_FILE
        try:
            if not file_path.exists():
                return {}
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_extra_settings(self, data: dict):
        file_path = self._EXTRA_SETTINGS_FILE
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _merge_module_llm_settings(self, payload: dict, extra: dict) -> dict:
        payload.update(self._MODULE_LLM_DEFAULTS)
        for key, default_value in self._MODULE_LLM_DEFAULTS.items():
            value = extra.get(key, default_value)
            if key in self._MODULE_LLM_NUMERIC_KEYS:
                if "temperature" in key:
                    try:
                        value = float(value)
                    except Exception:
                        value = float(default_value)
                else:
                    try:
                        value = int(value)
                    except Exception:
                        value = int(default_value)
            payload[key] = value
        return payload

    def _merge_retrieval_overrides(self, payload: dict, extra: dict) -> dict:
        # 允许通过 retrieval_tuning.json 统一覆盖检索参数，便于快速调优
        mode = str(
            extra.get(
                "retrieval_mode",
                payload.get(
                    "retrieval_mode",
                    self._RETRIEVAL_OVERRIDE_DEFAULTS.get("retrieval_mode", "hybrid"),
                ),
            )
        ).strip().lower()
        if mode not in {"vector", "keyword", "hybrid"}:
            mode = str(payload.get("retrieval_mode", "hybrid")).strip().lower()
            if mode not in {"vector", "keyword", "hybrid"}:
                mode = "hybrid"
        payload["retrieval_mode"] = mode

        try:
            vector_threshold = float(
                extra.get(
                    "vector_threshold",
                    payload.get(
                        "vector_threshold",
                        self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_threshold", 0.08),
                    ),
                )
            )
        except Exception:
            vector_threshold = float(
                self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_threshold", 0.08)
            )
        payload["vector_threshold"] = max(0.0, min(vector_threshold, 1.0))

        try:
            keyword_threshold = float(
                extra.get(
                    "keyword_threshold",
                    payload.get(
                        "keyword_threshold",
                        self._RETRIEVAL_OVERRIDE_DEFAULTS.get("keyword_threshold", 0.0),
                    ),
                )
            )
        except Exception:
            keyword_threshold = float(
                self._RETRIEVAL_OVERRIDE_DEFAULTS.get("keyword_threshold", 0.0)
            )
        payload["keyword_threshold"] = max(0.0, min(keyword_threshold, 1.0))

        try:
            vector_weight = float(
                extra.get(
                    "vector_weight",
                    payload.get(
                        "vector_weight",
                        self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_weight", 0.55),
                    ),
                )
            )
        except Exception:
            vector_weight = float(
                self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_weight", 0.55)
            )
        payload["vector_weight"] = max(0.0, min(vector_weight, 1.0))

        try:
            top_k = int(
                extra.get(
                    "top_k",
                    payload.get("top_k", self._RETRIEVAL_OVERRIDE_DEFAULTS.get("top_k", 8)),
                )
            )
        except Exception:
            top_k = int(self._RETRIEVAL_OVERRIDE_DEFAULTS.get("top_k", 8))
        payload["top_k"] = max(1, min(top_k, 50))

        payload["enable_query_rewrite"] = self._as_bool(
            extra.get(
                "enable_query_rewrite",
                payload.get(
                    "enable_query_rewrite",
                    self._RETRIEVAL_OVERRIDE_DEFAULTS.get("enable_query_rewrite", True),
                ),
            ),
            True,
        )
        payload["rewrite_only_when_needed"] = self._as_bool(
            extra.get(
                "rewrite_only_when_needed",
                payload.get(
                    "rewrite_only_when_needed",
                    self._RETRIEVAL_OVERRIDE_DEFAULTS.get("rewrite_only_when_needed", True),
                ),
            ),
            True,
        )
        try:
            keyword_index_ttl_sec = int(
                extra.get(
                    "keyword_index_ttl_sec",
                    payload.get(
                        "keyword_index_ttl_sec",
                        self._RETRIEVAL_OVERRIDE_DEFAULTS.get("keyword_index_ttl_sec", 300),
                    ),
                )
            )
        except Exception:
            keyword_index_ttl_sec = int(
                self._RETRIEVAL_OVERRIDE_DEFAULTS.get("keyword_index_ttl_sec", 300)
            )
        payload["keyword_index_ttl_sec"] = max(0, min(keyword_index_ttl_sec, 3600))
        return payload

    def get(self):
        extra = self._read_extra_settings()
        with self.session() as session:
            settings = session.query(Settings).filter_by(id="global").first()
            if settings:
                merged = settings.to_dict()
                merged.update(
                    {
                        "use_rerank": extra.get("use_rerank", True),
                        "rerank_candidate_k": extra.get("rerank_candidate_k", 24),
                        "rerank_language_mode": extra.get("rerank_language_mode", "auto"),
                    }
                )
                merged = self._merge_module_llm_settings(merged, extra)
                merged = self._merge_retrieval_overrides(merged, extra)
                merged = self._normalize_provider_fields(merged)
                return self._normalize_model_fields(merged)
            else:
                defaults = self._get_default_settings()
                defaults.update(
                    {
                        "use_rerank": extra.get("use_rerank", True),
                        "rerank_candidate_k": extra.get("rerank_candidate_k", 24),
                        "rerank_language_mode": extra.get("rerank_language_mode", "auto"),
                    }
                )
                defaults = self._merge_module_llm_settings(defaults, extra)
                defaults = self._merge_retrieval_overrides(defaults, extra)
                defaults = self._normalize_provider_fields(defaults)
                return self._normalize_model_fields(defaults)

    # 获取默认设置的方法
    def _get_default_settings(self) -> dict:
        """获取默认设置"""
        # 返回包含所有默认字段值的字典
        return {
            "id": "global",  # 设置主键
            "embedding_provider": "huggingface",  # 默认 embedding provider
            # "embedding_model_name": "C:/Users/lenovo/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",  # 默认 embedding 模型
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",  # 默认 embedding 模型
            "embedding_api_key": "",  # 默认无 embedding API key
            "embedding_base_url": "",  # 默认无 embedding base url
            "llm_provider": "gemini",  # 默认 LLM provider
            "llm_model_name": Config.GEMINI_CHAT_MODEL,  # 默认 LLM 模型
            "llm_api_key": Config.GEMINI_API_KEY,  # 配置里的默认 LLM API key
            "llm_base_url": Config.GEMINI_BASE_URL,  # 配置里的默认 LLM base url
            "llm_temperature": 0.7,  # 默认温度
            "chat_system_prompt": "你是一个严谨的智能文档问答助手。你的唯一任务是根据用户提供的【参考文档】来回答问题。 请你严格遵守以下三条铁律： 1. 绝对忠实：你的回答必须 100% 来源于【参考文档】，绝对不允许使用你的预训练知识、常识或进行任何主观捏造。 2. 拒绝推测：如果【参考文档】中没有直接包含回答该问题所需的信息，你必须直接回答：“抱歉，当前的知识库文档中没有找到与该问题相关的信息。” 不允许尝试给出一半的答案。 3. 必须引用：在你回答的每一句结论后，必须加上引用的原话或文档来源。",  # 聊天系统默认提示词
            "rag_system_prompt": "你是一个只允许基于检索文档作答的问答助手。你必须严格遵守：1) 只能使用提供的文档上下文，不得使用外部知识或常识补全；2) 每条关键结论后必须给出引用标记 [^n]；3) 若文档证据不足，必须原样回答：抱歉，当前检索到的文档中没有足够依据回答该问题。",  # RAG系统提示词
            "rag_query_prompt": "【文档上下文】\n{context}\n\n【用户问题】\n{question}\n\n请按以下要求输出：\n- 仅基于“文档上下文”回答；\n- 每条关键结论后添加引用标记 [^n]（n 对应文档片段编号）；\n- 如果无法从文档直接得到答案，只输出：抱歉，当前检索到的文档中没有足够依据回答该问题。",  # RAG查询提示词
            # "retrieval_mode": "vector",  # 默认检索模式
            "retrieval_mode": "hybrid",  # 默认检索模式
            "vector_threshold": 0.2,  # 向量检索阈值
            "keyword_threshold": 0.0,  # 关键词检索阈值
            "vector_weight": 0.7,  # 检索混合权重
            "top_k": 5,  # 返回结果数量
            "use_rerank": True,  # 是否启用重排
            "rerank_candidate_k": 24,  # 重排候选数量
            "rerank_language_mode": "auto",  # auto|always_on|always_off
        }

    def update(self, data):
        extra_payload = {}
        if "use_rerank" in data:
            value = data.get("use_rerank")
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            extra_payload["use_rerank"] = bool(value)
        if "rerank_candidate_k" in data:
            try:
                candidate_k = int(data.get("rerank_candidate_k"))
            except Exception:
                candidate_k = 24
            extra_payload["rerank_candidate_k"] = max(5, min(candidate_k, 200))
        if "rerank_language_mode" in data:
            mode = str(data.get("rerank_language_mode") or "auto").strip().lower()
            if mode not in {"auto", "always_on", "always_off"}:
                mode = "auto"
            extra_payload["rerank_language_mode"] = mode
        if "retrieval_mode" in data:
            mode = str(data.get("retrieval_mode") or "hybrid").strip().lower()
            if mode not in {"vector", "keyword", "hybrid"}:
                mode = "hybrid"
            extra_payload["retrieval_mode"] = mode
        if "vector_threshold" in data:
            try:
                value = float(data.get("vector_threshold"))
            except Exception:
                value = self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_threshold", 0.08)
            extra_payload["vector_threshold"] = max(0.0, min(float(value), 1.0))
        if "keyword_threshold" in data:
            try:
                value = float(data.get("keyword_threshold"))
            except Exception:
                value = self._RETRIEVAL_OVERRIDE_DEFAULTS.get("keyword_threshold", 0.0)
            extra_payload["keyword_threshold"] = max(0.0, min(float(value), 1.0))
        if "vector_weight" in data:
            try:
                value = float(data.get("vector_weight"))
            except Exception:
                value = self._RETRIEVAL_OVERRIDE_DEFAULTS.get("vector_weight", 0.55)
            extra_payload["vector_weight"] = max(0.0, min(float(value), 1.0))
        if "top_k" in data:
            try:
                value = int(float(data.get("top_k")))
            except Exception:
                value = self._RETRIEVAL_OVERRIDE_DEFAULTS.get("top_k", 8)
            extra_payload["top_k"] = max(1, min(int(value), 50))
        for key, default_value in self._MODULE_LLM_DEFAULTS.items():
            if key not in data:
                continue
            value = data.get(key)
            if key in self._MODULE_LLM_NUMERIC_KEYS:
                try:
                    value = (
                        float(value) if "temperature" in key else int(float(value))
                    )
                except Exception:
                    value = default_value
                if "temperature" in key:
                    value = max(0.0, min(float(value), 2.0))
                else:
                    value = max(64, min(int(value), 8192))
            elif value is None:
                value = default_value
            if key.endswith("_provider"):
                value = self._normalize_provider_name(value)
            extra_payload[key] = value

        if extra_payload:
            current_extra = self._read_extra_settings()
            current_extra.update(extra_payload)
            current_extra = self._normalize_provider_fields(current_extra)
            current_extra = self._normalize_model_fields(current_extra)
            self._write_extra_settings(current_extra)

        with self.transaction() as session:
            settings = session.query(Settings).filter_by(id="global").first()
            if not settings:
                settings = Settings(id="global")
                session.add(settings)
            for key, value in data.items():
                if hasattr(settings, key) and value is not None:
                    if key in {"embedding_provider", "llm_provider"}:
                        value = self._normalize_provider_name(value)
                    if key == "llm_model_name":
                        provider = self._normalize_provider_name(
                            data.get("llm_provider", settings.llm_provider)
                        )
                        if provider == "gemini" and str(value).strip().lower().startswith(
                            ("gpt-", "o1", "o3", "text-")
                        ):
                            value = Config.GEMINI_CHAT_MODEL
                    setattr(settings, key, value)
            session.flush()
            session.refresh(settings)
            result = settings.to_dict()
            result.update(self._read_extra_settings())
            return result


settings_service = SettingsService()
