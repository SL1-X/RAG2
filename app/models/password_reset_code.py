import uuid
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.models.base import BaseModel


class PasswordResetCode(BaseModel):
    __tablename__ = "password_reset_code"
    __repr_fields__ = ["id", "email", "is_used"]

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:32])
    user_id = Column(String(32), nullable=False, index=True)
    email = Column(String(128), nullable=False, index=True)
    code_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    is_used = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
