from app.services.settings_service import settings_service
from app.utils.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger
from app.services.rag_service import PIPELINE_MODE_FULL, rag_service

logger = get_logger(__name__)


class ChatService:
    def __init__(self):
        pass

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _has_role_fallback(settings: dict, role: str) -> bool:
        prefix = f"{role}_llm"
        provider = str(settings.get(f"{prefix}_fallback_provider", "")).strip()
        model_name = str(settings.get(f"{prefix}_fallback_model_name", "")).strip()
        return bool(provider and model_name)

    def _normalize_history_messages(self, history):
        """将历史消息转换为 ChatPromptTemplate 可识别的消息格式。"""
        if not history:
            return []
        role_map = {"user": "human", "assistant": "ai", "human": "human", "ai": "ai"}
        normalized = []
        for item in history:
            role = role_map.get((item.get("role") or "").lower())
            content = (item.get("content") or "").strip()
            if role and content:
                normalized.append((role, content))
        return normalized

    def chat_stream(self, question, history=None):
        settings = settings_service.get()
        temperature = self._as_float(
            settings.get("chat_llm_temperature", settings.get("llm_temperature", "0.7")),
            0.7,
        )
        temperature = max(0.0, min(temperature, 2.0))
        chat_system_prompt = settings.get("chat_system_prompt")
        if not chat_system_prompt:
            chat_system_prompt = (
                "You are a professional AI assistant. "
                "Always reply in the same language as the user's latest question. "
                "Be clear, accurate, and helpful."
            )
        language_guard = (
            "Language policy: You MUST reply in the same language as the user's latest message. "
            "If the user asks in English, reply in English. "
            "Do not switch language unless the user explicitly asks you to."
        )
        messages = [("system", chat_system_prompt), ("system", language_guard)]
        messages.extend(self._normalize_history_messages(history))
        messages.append(("human", question))
        prompt = ChatPromptTemplate.from_messages(messages)
        def _stream(use_fallback: bool):
            llm = LLMFactory.create_llm(
                settings,
                temperature=temperature,
                max_tokens=self._as_int(settings.get("chat_llm_max_tokens", 1024), 1024),
                role="chat",
                use_fallback=use_fallback,
            )
            chain = prompt | llm
            for chunk in chain.stream({}):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        # 服务器准备开始向客户端发送消息
        yield {"type": "start", "content": ""}
        full_answer = ""
        try:
            # 遍历大模型生成的每一段代码
            for content in _stream(use_fallback=False):
                full_answer += content
                yield {"type": "content", "content": content}
        except Exception as e:
            if self._has_role_fallback(settings, "chat"):
                logger.warning(f"chat 主模型流式失败，尝试 fallback: {e}")
                try:
                    for content in _stream(use_fallback=True):
                        full_answer += content
                        yield {"type": "content", "content": content}
                except Exception as fallback_error:
                    logger.error(f"fallback 流式生成时出错:{fallback_error}")
                    yield {"type": "error", "content": f"流式生成时出错:{fallback_error}"}
                    return
            else:
                logger.error(f"流式生成时出错:{e}")
                yield {"type": "error", "content": f"流式生成时出错:{e}"}
                return
        yield {"type": "done", "content": "", "metadata": {"question": question}}

    # 流式知识库问答：支持 full / retrieve_only / generate_only
    def ask_stream(
        self,
        kb_id,
        question,
        pipeline_mode: str = PIPELINE_MODE_FULL,
        context: str | None = None,
        history=None,
    ):
        return rag_service.ask_stream(
            kb_id=kb_id,
            question=question,
            pipeline_mode=pipeline_mode,
            context=context,
            history=history,
        )


chat_service = ChatService()
