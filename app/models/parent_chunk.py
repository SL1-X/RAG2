from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.sql import func
from app.models.base import BaseModel


class ParentChunk(BaseModel):
    __tablename__ = "parent_chunk"
    __repr_fields__ = ["parent_id", "doc_id"]

    # 使用 parent_id 作为主键，直接支持检索阶段按 parent_id 反查
    parent_id = Column(String(128), primary_key=True)
    kb_id = Column(
        String(32),
        ForeignKey("knowledgebase.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_id = Column(
        String(32),
        ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_parent_chunk_doc_parent", "doc_id", "parent_id"),
    )
