from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, BigInteger
from sqlalchemy.sql import func
import uuid
from app.models.base import BaseModel


class Document(BaseModel):
    # 指定数据库表名为document
    __tablename__ = "document"
    # 指定__repr__显示的字段
    __repr_fields__ = ["id", "name"]
    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:32])
    # 知识库的主键
    kb_id = Column(
        String(32),
        ForeignKey("knowledgebase.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 文档的名称
    name = Column(String(255), nullable=False)
    # 文件路径
    file_path = Column(String(512), nullable=False)
    # 文件类型
    file_type = Column(String(32), nullable=False)
    # 文件大小
    file_size = Column(BigInteger, nullable=False)
    # 文档状态 刚开始的是pending 等待处理
    status = Column(String(32), nullable=False)
    # 文件分块数量
    chunk_count = Column(Integer, nullable=True)
    # 处理错误消息
    error_message = Column(Text, nullable=True)
    # 创建时间 默认为当前时间 创建索引
    created_at = Column(DateTime, default=func.now(), index=True)
    # 更新时间 默认为当前时间，在数据更新的自动更新为当前最新的时间
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    # 复合索引
    # __table_args———— = (Index("name"), Index("kb_id"))


#   index=True, 普通 索引
# unique=True 建立唯一索引
