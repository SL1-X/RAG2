from sqlalchemy import Column, String, DateTime, Text, Float, Integer
from sqlalchemy.sql import func
from app.models.base import BaseModel


class Settings(BaseModel):
    # 指定数据库表名为settings
    __tablename__ = "settings"
    # 指定__repr__显示的字段
    __repr_fields__ = ["id"]
    id = Column(String(32), primary_key=True, default="global")
    # 向量模型提供商
    embedding_provider = Column(
        String(64), nullable=False, default="huggingface"
    )  # huggingface gemini ollama
    # 向量嵌入模型的名称
    embedding_model_name = Column(String(256), nullable=False)
    # 向量嵌入模型的基础地址
    embedding_base_url = Column(String(64), nullable=True)
    # 向量嵌入模型的密钥
    embedding_api_key = Column(String(64), nullable=True)

    # LLM模型提供商
    llm_provider = Column(
        String(64), nullable=False, default="deepseek"
    )  # deepseek gemini ollama
    # LLM模型的名称
    llm_model_name = Column(String(64), nullable=True)
    # LLM模型的基础地址
    llm_base_url = Column(String(64), nullable=True)
    # LLM模型的密钥
    llm_api_key = Column(String(64), nullable=True)
    # LLM模型温度
    llm_temperature = Column(String(64), nullable=True, default="0.7")

    # 提示词配置
    # 普通聊天系统提示词
    chat_system_prompt = Column(Text, nullable=True)
    # 知识库聊天系统提示词
    rag_system_prompt = Column(Text, nullable=True)
    # 知识库查询提示词
    rag_query_prompt = Column(Text, nullable=True)

    # 检索设置
    # 检索模式 可选值 vector(向量检索就是常说的稠密检索) keyword(关键字检索就是常说的稀疏检索) hybird(混合检索)
    retrieval_mode = Column(
        String(32),
        nullable=False,
        default="vector",
        comment="检索模型：vector(向量检索) keyword(关键字检索) hybird(混合检索)",
    )
    # 向量检索阈值
    vector_threshold = Column(Float, nullable=True, default=0.2, comment="向量检索阈值")
    # 关键字检索阈值
    keyword_threshold = Column(
        Float, nullable=True, default=0.2, comment="关键字检索阈值"
    )
    # 向量检索权重 在混合检索的时候，需要设置
    vector_weight = Column(
        Float,
        nullable=True,
        default=0.5,
        comment="向量检索的权重(只会在混合检索的时候使用)",
    )
    # 结果数量 ，可为空，默认值为5
    top_k = Column(Integer, nullable=True, default=5, comment="返回结果的数量")
    # 创建时间 默认为当前时间 创建索引
    created_at = Column(DateTime, default=func.now(), index=True)
    # 更新时间 默认为当前时间，在数据更新的自动更新为当前最新的时间
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
