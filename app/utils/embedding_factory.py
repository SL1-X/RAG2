from app.services.settings_service import settings_service
from app.utils.logger import get_logger
from app.config import Config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

logger = get_logger(__name__)


class EmbeddingFactory:
    @staticmethod
    def _normalize_embedding_model(provider: str, model_name: str) -> str:
        provider = (provider or "").strip().lower()
        name = (model_name or "").strip()
        if provider == "openai":
            provider = "gemini"
        if provider != "gemini":
            return name
        if not name:
            return Config.GEMINI_EMBEDDING_MODEL
        if name.lower().startswith(("text-embedding-", "text-")):
            return Config.GEMINI_EMBEDDING_MODEL
        return name

    @staticmethod
    def create_embeddings():
        settings = settings_service.get()
        embedding_provider = settings.get("embedding_provider")
        embedding_model_name = EmbeddingFactory._normalize_embedding_model(
            embedding_provider, settings.get("embedding_model_name")
        )
        embedding_api_key = settings.get("embedding_api_key")
        embedding_base_url = settings.get("embedding_base_url")
        try:
            if embedding_provider == "huggingface":
                embeddings = HuggingFaceEmbeddings(  # 这是一个本地模型 模型文件是在本地的
                    model_name=embedding_model_name,  # 不需要baseurl，也不需要apikey
                    model_kwargs={"device": "cpu"},
                    # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"创建HuggingFaceEmbeddings:{embedding_model_name}")
            elif embedding_provider in (
                "gemini",
                "openai",
            ):  # openai 作为历史配置兼容，统一走 Gemini
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embedding_model_name, google_api_key=embedding_api_key
                )
                logger.info(f"创建GeminiEmbeddings:{embedding_model_name}")
            elif (
                embedding_provider == "ollama"
            ):  # 调用的是本地服务 baseURL，但不需要apikey
                embeddings = OllamaEmbeddings(
                    model_name=embedding_model_name, base_url=embedding_base_url
                )
                logger.info(f"创建HuggingFaceEmbeddings:{embedding_model_name}")
            else:
                logger.warning(f"未知的Embedding提供商,默认使用huggingface")
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                    encode_kwargs={"normalize_embeddings": True},
                )
            return embeddings
        except Exception as e:
            logger.info(f"创建向量模型失败:{e}", exc_info=True)
            return HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                encode_kwargs={"normalize_embeddings": True},
            )
