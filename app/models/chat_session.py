from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
import uuid
from app.models.base import BaseModel


class ChatSession(BaseModel):
    # 指定数据库表名为user
    __tablename__ = "chat_session"
    # 指定__repr__显示的字段
    __repr_fields__ = ["id", "user_id", "title"]
    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:32])
    # 定义用户ID，外键关联到user表的id,删除用户时级联删除，不能为空，并且建有索引
    user_id = Column(
        String(32),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 知识库的主键
    kb_id = Column(
        String(32),
        ForeignKey("knowledgebase.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    # 会话的标题
    title = Column(String(128), nullable=True)
    # 创建时间 默认为当前时间 创建索引
    created_at = Column(DateTime, default=func.now(), index=True)
    # 更新时间 默认为当前时间，在数据更新的自动更新为当前最新的时间
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
