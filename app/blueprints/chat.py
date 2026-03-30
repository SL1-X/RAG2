"""
聊天相关的路由
"""

from flask import (
    Blueprint,
    request,
    flash,
    render_template,
    redirect,
    url_for,
    Response,
    stream_with_context,
)
import os
from time import perf_counter
from app.blueprints.utils import (
    success_response,
    error_response,
    handle_api_error,
    get_current_user_or_error,
    get_pagination_params,
    check_ownership,
)
from app.utils.logger import get_logger
from app.utils.auth import login_required, api_login_required, get_current_user
import json
from app.services.chat_service import chat_service
from app.services.rag_service import (
    PIPELINE_MODE_FULL,
    PIPELINE_MODE_RETRIEVE_ONLY,
    VALID_PIPELINE_MODES,
)

_TRIPLE_BRANCH_LABELS = {
    "vector": "向量检索",
    "keyword": "关键词检索",
    "hybrid": "混合检索",
}
from app.services.knowledgebase_service import kb_service
from app.services.chat_session_service import session_service
from app.services.evaluation_service import evaluation_service

logger = get_logger(__name__)

bp = Blueprint("chat", __name__)


@bp.route("/chat")
@login_required
def chat_view():
    current_user = get_current_user()
    result = kb_service.list(user_id=current_user["id"], page=1, page_size=100)
    return render_template("chat.html", knowledgebases=result["items"])


@bp.route("/api/v1/chat", methods=["POST"])
@handle_api_error
@api_login_required
def common_chat():
    # 现在第一步只实现普通聊天，不支持知识库
    current_user, err = get_current_user_or_error()
    if err:
        return err
    # 获取请求体JSON数据
    data = request.get_json()
    question = data["question"].strip()
    if not question:
        return error_response(f"用户的提问内容为空", 400)
    session_id = data.get("session_id")
    # 初始历史消息
    history = None
    if session_id:
        session_obj = session_service.get_session_by_id(session_id, current_user["id"])
        if not session_obj:
            return error_response("会话不存在或无权限访问", 404)
        # 获取当前用户的此会话的历史消息
        history_messages = session_service.get_messages(session_id, current_user["id"])
        # 将历史消息转换为对话格式,仅保留最近的10条消息
        history = [
            {"role": message.get("role"), "content": message.get("content")}
            for message in history_messages[-10:]  # 只取最近的10条
        ]
    else:
        chat_session = session_service.create_session(user_id=current_user["id"])
        session_id = chat_session["id"]
    # 将用户的问题消息保存到当前会话的消息表中
    session_service.add_message(session_id, "user", question)

    @stream_with_context
    def generate():
        try:
            # 用于缓存完整的答案内容
            full_answer = ""
            for chunk in chat_service.chat_stream(question=question, history=history):
                if chunk.get("type") == "content":
                    full_answer += chunk.get("content")
                yield f"data: {json.dumps(chunk,ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if full_answer:
                session_service.add_message(session_id, "assistant", full_answer)
        except Exception as e:
            logger.error(f"流式输出出错:{e}")
            error_chunk = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_chunk,ensure_ascii=False)}\n\n"

    response = Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )
    return response


@bp.route("/api/v1/sessions", methods=["POST"])
@api_login_required
@handle_api_error
def api_create_session():
    # 现在第一步只实现普通聊天，不支持知识库
    current_user, err = get_current_user_or_error()
    if err:
        return err
    # 获取请求体JSON数据
    data = request.get_json()
    # 获取会话的标题
    title = data.get("title", "")

    session_dict = session_service.create_session(
        user_id=current_user["id"], title=title
    )
    return success_response(session_dict)


@bp.route("/api/v1/sessions", methods=["GET"])
@api_login_required
@handle_api_error
def api_list_sessions():
    # 现在第一步只实现普通聊天，不支持知识库
    current_user, err = get_current_user_or_error()
    if err:
        return err
    page, page_size = get_pagination_params(max_page_size=1000)
    result = session_service.list_sessions(
        current_user["id"], page=page, page_size=page_size
    )
    return success_response(result)


@bp.route("/api/v1/sessions/<session_id>", methods=["DELETE"])
@api_login_required
@handle_api_error
def api_delete_session(session_id):
    current_user, err = get_current_user_or_error()
    if err:
        return err
    success = session_service.delete_session(session_id, current_user["id"])
    if success:
        return success_response(None, "会话删除成功")
    else:
        return error_response("会话删除失败", 404)


@bp.route("/api/v1/sessions", methods=["DELETE"])
@api_login_required
@handle_api_error
def api_delete_all_session():
    current_user, err = get_current_user_or_error()
    if err:
        return err
    success = session_service.delete_all_session(current_user["id"])
    if success:
        return success_response(None, "会话全部删除成功")
    else:
        return error_response("会话全部删除失败", 404)


@bp.route("/api/v1/sessions/<session_id>", methods=["GET"])
@api_login_required
@handle_api_error
def api_get_session(session_id):
    current_user, err = get_current_user_or_error()
    if err:
        return err
    session_obj = session_service.get_session_by_id(session_id, current_user["id"])
    if not session_obj:
        return error_response("会话不存在", 404)
    messages = session_service.get_messages(session_id, current_user["id"])
    return success_response({"session": session_obj, "messages": messages})


@bp.route("/api/v1/knowledgebases/<kb_id>/chat", methods=["POST"])
@handle_api_error
@api_login_required
def rag_chat(kb_id):
    # 支持知识库检索聊天
    current_user, err = get_current_user_or_error()
    if err:
        return err
    kb = kb_service.get_by_id(kb_id)
    # 判断此知识库是否是用户自己的知识库
    has_permission, err = check_ownership(kb["user_id"], current_user["id"], "知识库")
    if not has_permission:
        return err
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return error_response("用户的提问内容为空", 400)
    session_id = data.get("session_id")
    pipeline_mode = (data.get("pipeline_mode") or PIPELINE_MODE_FULL).strip()
    extra_context = (data.get("context") or "").strip()
    if pipeline_mode not in VALID_PIPELINE_MODES:
        return error_response(
            f"无效的 pipeline_mode，可选: {', '.join(sorted(VALID_PIPELINE_MODES))}",
            400,
        )
    max_tokens = int(data.get("max_tokens", 1024))
    max_tokens = max(1, min(max_tokens, 10240))
    eval_cfg = data.get("evaluation") or {}
    reference_answer = (eval_cfg.get("reference_answer") or "").strip() or None
    gold_chunk_ids = eval_cfg.get("gold_chunk_ids") or []
    k_values = eval_cfg.get("k_values") or [1, 3, 5]
    req_started = perf_counter()
    history = None
    # 创建或校验会话，并确保会话绑定知识库，方便历史记录回切
    if not session_id:
        chat_session = session_service.create_session(
            user_id=current_user["id"], kb_id=kb_id
        )
        session_id = chat_session["id"]
    else:
        session_obj = session_service.get_session_by_id(session_id, current_user["id"])
        if not session_obj:
            return error_response("会话不存在或无权限访问", 404)
        existing_kb_id = session_obj.get("kb_id")
        if existing_kb_id and existing_kb_id != kb_id:
            # 同一会话已经绑定其他知识库时，自动新建会话，避免上下文与知识库串线
            chat_session = session_service.create_session(
                user_id=current_user["id"], kb_id=kb_id
            )
            session_id = chat_session["id"]
        else:
            session_service.bind_kb_if_missing(session_id, current_user["id"], kb_id)
            history_messages = session_service.get_messages(session_id, current_user["id"])
            history = [
                {"role": message.get("role"), "content": message.get("content")}
                for message in history_messages[-10:]
            ]
    # 保存用户的问题到消息列表中
    session_service.add_message(session_id, "user", question)

    @stream_with_context
    def generate():
        try:
            full_answer = ""
            sources = None
            final_eval = None
            final_meta = None
            for chunk in chat_service.ask_stream(
                kb_id,
                question=question,
                pipeline_mode=pipeline_mode,
                context=extra_context,
                history=history,
                settings_override={
                    "rag_llm_max_tokens": max_tokens,
                    # 总结场景的 reduce/repair 阶段同样放宽，减少“只出半段”概率
                    "summary_reduce_max_tokens": max(1280, max_tokens),
                    "summary_repair_max_tokens": max(1280, max_tokens),
                },
            ):
                ct = chunk.get("type")
                if ct == "branch_start":
                    b = chunk.get("branch")
                    full_answer += f"\n\n### {_TRIPLE_BRANCH_LABELS.get(b, b)}\n\n"
                elif ct == "content":
                    full_answer += chunk.get("content", "")
                elif ct == "done":
                    meta = chunk.get("metadata") or {}
                    if meta.get("triple"):
                        sources = meta["triple"]
                        triple_eval = evaluation_service.evaluate_triple_answers(
                            sources,
                            reference_answer=reference_answer,
                            gold_chunk_ids=gold_chunk_ids,
                            k_values=k_values,
                        )
                        if triple_eval:
                            meta["evaluation"] = triple_eval
                    else:
                        sources = chunk.get("sources")
                        if meta.get("answer_with_citations"):
                            full_answer = meta.get("answer_with_citations", full_answer)
                        single_eval = evaluation_service.evaluate_single_answer(
                            answer=full_answer.strip(),
                            sources=sources,
                            reference_answer=reference_answer,
                            gold_chunk_ids=gold_chunk_ids,
                            k_values=k_values,
                            efficiency={
                                "pipeline_elapsed_ms": meta.get("pipeline_elapsed_ms", 0),
                                "retrieval_elapsed_ms": meta.get("retrieval_elapsed_ms", 0),
                                "generation_elapsed_ms": meta.get("generation_elapsed_ms", 0),
                            },
                        )
                        meta["evaluation"] = single_eval
                    meta["efficiency"] = {
                        "request_elapsed_ms": int((perf_counter() - req_started) * 1000),
                        "pipeline_elapsed_ms": meta.get("pipeline_elapsed_ms", 0),
                    }
                    final_eval = meta.get("evaluation")
                    final_meta = {
                        "pipeline_mode": meta.get("pipeline_mode"),
                        "retrieval_mode": meta.get("retrieval_mode"),
                        "retrieval_debug": meta.get("retrieval_debug"),
                        "pipeline_elapsed_ms": meta.get("pipeline_elapsed_ms", 0),
                        "retrieval_elapsed_ms": meta.get("retrieval_elapsed_ms", 0),
                        "generation_elapsed_ms": meta.get("generation_elapsed_ms", 0),
                        "efficiency": meta.get("efficiency", {}),
                    }
                    chunk["metadata"] = meta
                yield f"data: {json.dumps(chunk,ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if full_answer:
                stored_payload = {
                    "__rag_payload_v1": True,
                    "sources": sources,
                    "evaluation": final_eval,
                    "metadata": final_meta,
                }
                session_service.add_message(
                    session_id, "assistant", full_answer.strip(), stored_payload
                )
            elif pipeline_mode == PIPELINE_MODE_RETRIEVE_ONLY and sources:
                stored_payload = {
                    "__rag_payload_v1": True,
                    "sources": sources,
                    "evaluation": final_eval,
                    "metadata": final_meta,
                }
                session_service.add_message(
                    session_id,
                    "assistant",
                    "（仅检索模式：未调用大模型，见引用来源）",
                    stored_payload,
                )

        except Exception as e:
            logger.error(f"流式输出时出错:{e}")
            error_chunk = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_chunk,ensure_ascii=False)}\n\n"

    response = Response(
        generate(),
        mimetype="text/event-stream",  ## 响应的内容类型
        headers={  # 响应头的类型
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )
    return response
