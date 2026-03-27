from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
import uuid
from app.models.base import BaseModel


class Knowledgebase(BaseModel):
    # 指定数据库表名为knowledgebase
    __tablename__ = "knowledgebase"
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_knowledgebase_user_id_name"),
    )
    # 指定__repr__显示的字段
    __repr_fields__ = ["id", "name"]
    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:32])
    # 定义用户ID，外键关联到user表的id,删除用户时级联删除，不能为空，并且建有索引
    user_id = Column(
        String(32),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 知识库的名称
    name = Column(String(128), nullable=False, index=True)
    # 知识库描述
    description = Column(Text, nullable=True)
    # 知识库封面图片路径 可以存储到本地硬盘上，也可以保存到minio等对象存储中去
    cover_image = Column(String(512), nullable=True, comment="封面图片")
    # 分块大小
    chunk_size = Column(Integer, nullable=False, comment="分块大小")
    # 分块重叠大小
    chunk_overlap = Column(Integer, nullable=False, comment="分块重叠大小")
    # 创建时间 默认为当前时间 创建索引
    created_at = Column(DateTime, default=func.now(), index=True)
    # 更新时间 默认为当前时间，在数据更新的自动更新为当前最新的时间
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
