# 模型配置
# 定义可用的 Embedding 模型和 LLM 模型列表
"""
模型配置
定义可用的 Embedding 模型和 LLM 模型列表
"""

# 定义向量嵌入模型（Embedding Models）的配置字典
EMBEDDING_MODELS = {
    # HuggingFace 嵌入模型
    "huggingface": {
        # 名称
        "name": "HuggingFace Embeddings",
        # 描述说明
        "description": "本地 HuggingFace 模型",
        # 可用模型列表
        "models": [
            # 第一个模型：all-MiniLM-L6-v2
            {
                # 模型名称
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                # 模型路径
                # "path": "C:/Users/lenovo/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",
                "path": "sentence-transformers/all-MiniLM-L6-v2",
                # 向量维度
                "dimension": "384",
                # 描述
                "description": "轻量级多语言模型，速度快",
            },
            # 第二个模型：paraphrase-multilingual-MiniLM-L12-v2
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": "384",
                "description": "多语言模型，支持中文",
            },
            # 第三个模型：bge-small-zh-v1.5
            {
                "name": "BAAI/bge-small-zh-v1.5",
                "path": "BAAI/bge-small-zh-v1.5",
                "dimension": "512",
                "description": "中文优化模型",
            },
        ],
        # 是否需要 API Key
        "requires_api_key": True,
        # 是否需要 Base URL
        "requires_base_url": True,
    },
    # Gemini 嵌入模型
    "gemini": {
        "name": "Gemini Embeddings",
        "description": "Google Gemini 官方嵌入模型",
        # Gemini 可用模型
        "models": [
            {
                "name": "models/text-embedding-004",
                "dimension": "768",
                "description": "通用文本嵌入模型，推荐默认",
            },
            {
                "name": "models/embedding-001",
                "dimension": "768",
                "description": "兼容模型",
            },
        ],
        "requires_api_key": True,
        "requires_base_url": False,
    },
    # 本地 Ollama 嵌入模型
    "ollama": {
        "name": "Ollama Embeddings",
        "description": "本地 Ollama 模型",
        # Ollama 可用嵌入模型
        "models": [
            {
                "name": "nomic-embed-text",
                "dimension": "768",
                "description": "通用文本嵌入模型",
            }
        ],
        "requires_api_key": False,
        "requires_base_url": True,
    },
}

# 定义 LLM（大模型，推理/对话模型）配置字典
LLM_MODELS = {
    # DeepSeek 模型配置
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek API",
        # DeepSeek 可用模型列表
        "models": [
            # DeepSeek 对话模型
            {"name": "deepseek-chat", "description": "对话模型"},
            # DeepSeek 代码模型
            {"name": "deepseek-coder", "description": "代码模型"},
        ],
        "requires_api_key": True,
        "requires_base_url": True,
    },
    # Gemini 大模型配置
    "gemini": {
        "name": "Gemini",
        "description": "Google Gemini API",
        # Gemini 可用大模型
        "models": [
            {"name": "gemini-2.5-pro", "description": "高质量推理模型"},
            {"name": "gemini-2.5-flash", "description": "高性价比快速模型"},
            {"name": "gemini-2.0-flash", "description": "低延迟模型"},
        ],
        "requires_api_key": True,
        "requires_base_url": False,
    },
    # 本地 Ollama 大模型配置
    "ollama": {
        "name": "Ollama",
        "description": "本地 Ollama 模型",
        # Ollama 可用大模型
        "models": [
            {"name": "qwen2.5:7b", "description": "英文问答性价比模型"},
            {"name": "mistral:7b", "description": "轻量英文模型"},
            {"name": "llama3.1:8b", "description": "通用英文模型"},
        ],
        "requires_api_key": False,
        "requires_base_url": True,
    },
}
