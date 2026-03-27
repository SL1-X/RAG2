from app.utils.logger import get_logger
from app.config import Config
from app.services.settings_service import settings_service

logger = get_logger(__name__)


class LLMFactory:
    # 注册的LLM提供者，服务提供商，用于存储各个provider的构建函数
    _providers = {}

    @classmethod
    def register_provider(cls, provider_name, provider_factory):
        cls._providers[provider_name.lower()] = provider_factory
        logger.info(f"已经注册了LLM提供商:{provider_name}")

    @classmethod
    def _as_float(cls, value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    @classmethod
    def _as_int(cls, value, default):
        try:
            return int(value)
        except Exception:
            return int(default)

    @classmethod
    def _normalize_model_name_for_provider(cls, provider: str, model_name: str) -> str:
        provider = (provider or "").strip().lower()
        name = (model_name or "").strip()
        if provider != "gemini":
            return name
        if not name:
            return Config.GEMINI_CHAT_MODEL
        return name

    @classmethod
    def _build_llm_settings_for_role(cls, settings: dict, role: str) -> dict:
        role = (role or "default").strip().lower()
        if role == "default":
            return {
                "provider": settings.get("llm_provider", "deepseek"),
                "model_name": settings.get("llm_model_name", Config.DEEPSEEK_CHAT_MODEL),
                "api_key": settings.get("llm_api_key", Config.DEEPSEEK_API_KEY),
                "base_url": settings.get("llm_base_url", Config.DEEPSEEK_BASE_URL),
                "temperature": settings.get("llm_temperature", 0.7),
                "max_tokens": settings.get("llm_max_tokens", 1024),
            }

        # role 支持 rewrite / rag / chat
        role_prefix = f"{role}_llm"
        return {
            "provider": settings.get(
                f"{role_prefix}_provider", settings.get("llm_provider", "deepseek")
            ),
            "model_name": settings.get(
                f"{role_prefix}_model_name",
                settings.get("llm_model_name", Config.DEEPSEEK_CHAT_MODEL),
            ),
            "api_key": settings.get(
                f"{role_prefix}_api_key",
                settings.get("llm_api_key", Config.DEEPSEEK_API_KEY),
            ),
            "base_url": settings.get(
                f"{role_prefix}_base_url",
                settings.get("llm_base_url", Config.DEEPSEEK_BASE_URL),
            ),
            "temperature": settings.get(
                f"{role_prefix}_temperature", settings.get("llm_temperature", 0.7)
            ),
            "max_tokens": settings.get(f"{role_prefix}_max_tokens", 1024),
            "fallback_provider": settings.get(f"{role_prefix}_fallback_provider", ""),
            "fallback_model_name": settings.get(f"{role_prefix}_fallback_model_name", ""),
            "fallback_api_key": settings.get(f"{role_prefix}_fallback_api_key", ""),
            "fallback_base_url": settings.get(f"{role_prefix}_fallback_base_url", ""),
        }

    @classmethod
    def _create_by_provider(
        cls, provider: str, provider_settings: dict, temperature, max_tokens, streaming
    ):
        provider = (provider or "").lower()
        if provider == "deepseek":
            return cls._create_deepseek(
                provider_settings, temperature, max_tokens, streaming
            )
        if provider == "gemini":
            return cls._create_gemini(provider_settings, temperature, max_tokens, streaming)
        if provider == "ollama":
            return cls._create_ollama(provider_settings, temperature, max_tokens, streaming)
        raise ValueError(f"不支持LLM提供商{provider}")

    @classmethod
    def _provider_defaults(cls, provider: str) -> dict:
        provider = (provider or "").strip().lower()
        if provider == "gemini":
            return {
                "llm_model_name": Config.GEMINI_CHAT_MODEL,
                "llm_api_key": Config.GEMINI_API_KEY,
                "llm_base_url": Config.GEMINI_BASE_URL,
            }
        if provider == "ollama":
            return {
                "llm_model_name": Config.OLLAMA_CHAT_MODEL,
                "llm_api_key": Config.OLLAMA_API_KEY,
                "llm_base_url": Config.OLLAMA_BASE_URL,
            }
        return {
            "llm_model_name": Config.DEEPSEEK_CHAT_MODEL,
            "llm_api_key": Config.DEEPSEEK_API_KEY,
            "llm_base_url": Config.DEEPSEEK_BASE_URL,
        }

    @classmethod
    def _merge_with_provider_defaults(cls, provider: str, payload: dict) -> dict:
        defaults = cls._provider_defaults(provider)
        merged = dict(defaults)
        merged.update(payload or {})
        merged["llm_model_name"] = (merged.get("llm_model_name") or defaults["llm_model_name"]).strip()
        merged["llm_api_key"] = (merged.get("llm_api_key") or defaults["llm_api_key"]).strip()
        merged["llm_base_url"] = (merged.get("llm_base_url") or defaults["llm_base_url"]).strip()
        return merged

    @classmethod
    def create_llm(
        cls,
        settings=None,
        temperature=None,
        max_tokens=None,
        streaming=True,
        role: str = "default",
        use_fallback: bool = False,
    ):
        if settings is None:
            settings = settings_service.get()

        resolved = cls._build_llm_settings_for_role(settings, role)
        provider = (resolved.get("provider") or "deepseek").strip().lower()
        fallback_provider = (
            resolved.get("fallback_provider") or settings.get("llm_provider", "")
        ).strip().lower()
        fallback_model_name = (resolved.get("fallback_model_name") or "").strip()
        if use_fallback:
            if not fallback_provider or not fallback_model_name:
                raise ValueError(f"角色{role}未配置可用 fallback 模型")
            provider = fallback_provider
        final_temperature = (
            cls._as_float(temperature, 0.7)
            if temperature is not None
            else cls._as_float(resolved.get("temperature", 0.7), 0.7)
        )
        final_max_tokens = (
            cls._as_int(max_tokens, 1024)
            if max_tokens is not None
            else cls._as_int(resolved.get("max_tokens", 1024), 1024)
        )
        if use_fallback:
            primary_settings = {
                "llm_model_name": cls._normalize_model_name_for_provider(
                    provider, fallback_model_name
                ),
                "llm_api_key": resolved.get("fallback_api_key"),
                "llm_base_url": resolved.get("fallback_base_url"),
            }
        else:
            primary_settings = {
                "llm_model_name": cls._normalize_model_name_for_provider(
                    provider, resolved.get("model_name")
                ),
                "llm_api_key": resolved.get("api_key"),
                "llm_base_url": resolved.get("base_url"),
            }
        primary_settings = cls._merge_with_provider_defaults(provider, primary_settings)

        try:
            return cls._create_by_provider(
                provider, primary_settings, final_temperature, final_max_tokens, streaming
            )
        except Exception as primary_error:
            if use_fallback:
                raise primary_error
            if not fallback_provider or not fallback_model_name:
                raise primary_error
            logger.warning(
                f"角色{role}主模型初始化失败，回退到{fallback_provider}/{fallback_model_name}: {primary_error}"
            )
            fallback_settings = {
                "llm_model_name": cls._normalize_model_name_for_provider(
                    fallback_provider, fallback_model_name
                ),
                "llm_api_key": resolved.get("fallback_api_key"),
                "llm_base_url": resolved.get("fallback_base_url"),
            }
            fallback_settings = cls._merge_with_provider_defaults(
                fallback_provider, fallback_settings
            )
            return cls._create_by_provider(
                fallback_provider,
                fallback_settings,
                final_temperature,
                final_max_tokens,
                streaming,
            )

    @classmethod
    def _create_deepseek(cls, settings, temperature, max_tokens, streaming):
        from langchain_deepseek import ChatDeepSeek

        model_name = settings.get("llm_model_name", Config.DEEPSEEK_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.DEEPSEEK_API_KEY)
        base_url = settings.get("llm_base_url", Config.DEEPSEEK_BASE_URL)
        llm = ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        logger.info(f"已经创建DeepSeek LLM:{model_name}")
        return llm

    @classmethod
    def _create_gemini(cls, settings, temperature, max_tokens, streaming):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = settings.get("llm_model_name", Config.GEMINI_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.GEMINI_API_KEY)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            disable_streaming=not streaming,
        )
        logger.info(f"已经创建Gemini LLM:{model_name}")
        return llm

    @classmethod
    def _create_ollama(cls, settings, temperature, max_tokens, streaming):
        from langchain_community.chat_models import ChatOllama

        model_name = settings.get("llm_model_name", Config.OLLAMA_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.OLLAMA_API_KEY)
        base_url = settings.get("llm_base_url", Config.OLLAMA_BASE_URL)
        llm = ChatOllama(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        logger.info(f"已经创建Ollama LLM:{model_name}")
        return llm


LLMFactory.register_provider("deepseek", LLMFactory._create_deepseek)
LLMFactory.register_provider("gemini", LLMFactory._create_gemini)
LLMFactory.register_provider("ollama", LLMFactory._create_ollama)
