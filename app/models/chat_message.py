from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
import uuid
import json
from app.models.base import BaseModel


class ChatMessage(BaseModel):
    # 指定数据库表名为user
    __tablename__ = "chat_message"
    # 指定__repr__显示的字段
    __repr_fields__ = ["id", "session_id", "role"]
    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:32])
    # 此消息是属于哪个会话
    session_id = Column(
        String(32),
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 角色 区分user assistant
    role = Column(String(64), nullable=True)
    # 消息的内容
    content = Column(Text, nullable=True)
    # 定义引的来源,JSON类型 当使用知识回答的时候,会把引用的知识库的文本片段放在sources里
    sources = Column(Text, nullable=True)
    # 创建时间 默认为当前时间 创建索引
    created_at = Column(DateTime, default=func.now(), index=True)
    # 更新时间 默认为当前时间，在数据更新的自动更新为当前最新的时间
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def to_dict(self, exclude=[], **kwargs):
        result = super().to_dict(exclude=exclude, **kwargs)
        if result.get("sources") and isinstance(result["sources"], str):
            try:
                result["sources"] = json.loads(result["sources"])
            except json.JSONDecodeError:
                result["sources"] = None
        return result
