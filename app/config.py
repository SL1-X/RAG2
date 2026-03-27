import os
from pathlib import Path
from dotenv import load_dotenv

# 固定从项目根目录加载 .env，并覆盖空的外部环境变量，避免出现 DB_PASSWORD 被意外置空
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)


# 项目根目录的路径
class Config:
    BASE_DIR = Path(__file__).parent.parent
    # 加载环境变量中配置的密钥
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # 应用配置
    # 应用监听的主机地址
    APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
    # 服务器监听的端口号
    APP_PORT = os.environ.get("APP_PORT", 5001)
    # 是否启动调用模式
    APP_DEBUG = os.environ.get("APP_DEBUG", "false").lower() == "true"
    # 上传的文件的最大文件大小
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 104857600))  # 100M
    # 允许 上传的文件
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
    # 允许 上传的图片的扩展名
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
    # 允许 上传的图片的最大大小，默认为5M
    MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", 5242880))

    # 日志配置
    # 日志存放目录
    LOG_DIR = os.environ.get("LOG_DIR", "./logs")
    # 日志文件
    LOG_FILE = os.environ.get("LOG_FILE", "rag_lite.log")
    # 日志级别
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    # 是否启用文件日志
    LOG_ENABLE_FILE = os.environ.get("LOG_ENABLE_FILE", "true").lower() == "true"
    # 是否启用控制台
    LOG_ENABLE_CONSOLE = os.environ.get("LOG_ENABLE_CONSOLE", "true").lower() == "true"

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", 3306)
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME = os.environ.get("DB_NAME", "rag")
    DB_CHARSET = os.environ.get("DB_CHARSET", "utf8mb4")

    # 存储的类型
    STORAGE_TYPE = os.environ.get("STORAGE_TYPE", "local")  # local / minio
    # 本地文件的存储目录
    STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storages")

    # MinIO 配置（当 STORAGE_TYPE='minio' 时使用）
    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
    MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "rag-lite")
    MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    MINIO_REGION = os.environ.get("MINIO_REGION", None)

    DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_API_KEY = os.environ.get(
        "DEEPSEEK_API_KEY", ""
    )
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    GEMINI_FAST_MODEL = os.environ.get("GEMINI_FAST_MODEL", "gemini-2.0-flash")
    GEMINI_EMBEDDING_MODEL = os.environ.get(
        "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_BASE_URL = os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
    )

    OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    # 指定向量数据库的类型
    VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma")  # chroma 或 milvus
    # 指定 chroma向量数据库的本地存储目录
    CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
