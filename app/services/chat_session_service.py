from app.services.base_service import BaseService
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.utils.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger
from sqlalchemy import desc
from datetime import datetime
import json

logger = get_logger(__name__)


class ChatSessionService(BaseService[ChatSession]):
    def create_session(self, user_id, kb_id=None, title=None):
        with self.transaction() as session:
            if not title:
                title = "新对话"
            chat_session = ChatSession(user_id=user_id, title=title, kb_id=kb_id)
            session.add(chat_session)
            session.flush()
            session.refresh(chat_session)
            self.logger.info(f"已经创建聊天会话:{chat_session.id},用户ID:{user_id}")
            return chat_session.to_dict()

    def list_sessions(self, user_id, page=1, page_size=100):
        with self.transaction() as session:
            query = session.query(ChatSession).filter_by(user_id=user_id)
            return self.paginate_query(
                query,
                page=page,
                page_size=page_size,
                order_by=desc(ChatSession.updated_at),
            )

    def delete_session(self, session_id, user_id):
        with self.transaction() as session:
            # TODO 应该先删除会话下面的消息再删除此会话
            chat_session = (
                session.query(ChatSession)
                .filter_by(id=session_id, user_id=user_id)
                .first()
            )
            if not chat_session:
                return False
            session.delete(chat_session)
            self.logger.info(f"删除会话{session_id}成功")
            return True

    def delete_all_session(self, user_id):
        with self.transaction() as session:
            count = session.query(ChatSession).filter_by(user_id=user_id).delete()
            self.logger.info(f"已经删除了用户{user_id}的个{count}聊天会话")
            return count

    def get_messages(self, session_id, user_id):
        with self.session() as session:
            if user_id:
                chat_session = (
                    session.query(ChatSession)
                    .filter_by(id=session_id, user_id=user_id)
                    .first()
                )
                if not chat_session:
                    return []
            # 查询此对话的历史消息
            messages = (
                session.query(ChatMessage)
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            return [message.to_dict() for message in messages]

    def add_message(self, session_id, role, content, sources=None):
        with self.transaction() as session:
            # 构建新的消息对象
            sources_str = json.dumps(sources) if sources is not None else None
            message = ChatMessage(
                session_id=session_id, role=role, content=content, sources=sources_str
            )
            session.add(message)
            # 查询出当前的会话对象
            chat_session = session.query(ChatSession).filter_by(id=session_id).first()
            if chat_session:
                # 更新会话的更新时间
                chat_session.updated_at = datetime.now()
                # 如果是此消息的用户角色是用户的话,并且此对话标题为空或者是新对话,说明还没有设置标题
                if role == "user" and (
                    not chat_session.title or chat_session.title == "新对话"
                ):
                    title = content[:30] + ("..." if len(content) > 30 else "")
                    chat_session.title = title
            session.flush()
            session.refresh(message)
            return message.to_dict()

    def get_session_by_id(self, session_id, user_id):
        with self.session() as session:
            query = session.query(ChatSession).filter_by(id=session_id)
            if user_id:
                query = query.filter_by(user_id=user_id)
            chat_session = query.first()
            if chat_session:
                return chat_session.to_dict()
            else:
                return None

    def bind_kb_if_missing(self, session_id, user_id, kb_id):
        """
        将会话绑定到知识库（仅当当前会话还未绑定 kb_id 时）。
        返回更新后的会话字典；会话不存在时返回 None。
        """
        with self.transaction() as session:
            chat_session = (
                session.query(ChatSession)
                .filter_by(id=session_id, user_id=user_id)
                .first()
            )
            if not chat_session:
                return None
            if not chat_session.kb_id:
                chat_session.kb_id = kb_id
                session.flush()
                session.refresh(chat_session)
            return chat_session.to_dict()


session_service = ChatSessionService()
