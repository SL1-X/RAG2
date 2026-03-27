from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    abort,
    send_file,
)
from io import BytesIO
import os
from app.utils.logger import get_logger
from app.services.knowledgebase_service import kb_service
from app.blueprints.utils import (
    success_response,
    error_response,
    handle_api_error,
    get_pagination_params,
    get_current_user_or_error,
    check_ownership,
)
from app.utils.auth import get_current_user, login_required, api_login_required
from app.services.storage_service import storage_service
from app.services.document_service import document_service
from app.services.navigation_service import navigation_service

logger = get_logger(__name__)

bp = Blueprint("knowledgebase", __name__)


@bp.route("/api/v1/kb", methods=["POST"])
@handle_api_error
def api_create():
    current_user, err = get_current_user_or_error()
    # 如果返回错误，说明用户未登录，直接返回错误消息
    if err:
        return err
    name = request.form.get("name")
    description = request.form.get("description")
    chunk_size = request.form.get("chunk_size", 512)
    chunk_overlap = request.form.get("chunk_overlap", 50)
    cover_image_data = None
    cover_image_filename = None
    # 判断请求传过来的请求文件中是否有cover_image
    if "cover_image" in request.files:
        cover_file = request.files["cover_image"]
        if cover_file and cover_file.filename:
            # 读取文件的内容为二进制数据
            cover_image_data = cover_file.read()
            # 读取上传的文件的文件名
            cover_image_filename = cover_file.filename
            logger.info(
                f"收到新的知识库的封面图片：文件名:{cover_image_filename},大小:{len(cover_image_data)},内容类型:{cover_file.content_type}"
            )
    kb_dict = kb_service.create(
        name=name,
        user_id=current_user["id"],
        description=description,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cover_image_data=cover_image_data,
        cover_image_filename=cover_image_filename,
    )
    return success_response(kb_dict)


@bp.route("/kb")
@login_required
def kb_list():
    # 获取当前登录的用户
    current_user = get_current_user()
    # 获取分页参数
    page, page_size = get_pagination_params(max_page_size=100)
    search = request.args.get("search", "").strip()
    sort_by = request.args.get("sort_by", "created_at").strip()
    sort_order = request.args.get("sort_order", "desc").strip()
    result = kb_service.list(
        user_id=current_user["id"],  # 当前登录的用户ID
        page=page,  # 页码
        page_size=page_size,  # 每页的大小
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return render_template(
        "kb_list.html",
        kbs=result["items"],
        pagination=result,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@bp.route("/api/v1/kb/<kb_id>", methods=["DELETE"])
@api_login_required
@handle_api_error
def api_delete(kb_id):
    current_user, error = get_current_user_or_error()
    if error:
        return error
    kb_dict = kb_service.get_by_id(kb_id)
    if not kb_dict:
        return error_response("知识库未找到", 404)
    # 验证用户是否有删除此知识库的权限
    has_permission, err = check_ownership(
        kb_dict["user_id"], current_user["id"], "knowledgebase"
    )
    if not has_permission:
        return err
    success = kb_service.delete(kb_id)
    if not success:
        return error_response("知识库删除失败", 500)
    return success_response()


@bp.route("/api/v1/kb/<kb_id>", methods=["PUT"])
@api_login_required
@handle_api_error
def api_update(kb_id):
    current_user, error = get_current_user_or_error()
    if error:
        return error
    kb_dict = kb_service.get_by_id(kb_id)
    if not kb_dict:
        return error_response("知识库未找到", 404)
    # 验证用户是否有删除此知识库的权限
    has_permission, err = check_ownership(
        kb_dict["user_id"], current_user["id"], "knowledgebase"
    )
    if not has_permission:
        return err
    name = request.form.get("name")
    description = request.form.get("description")
    chunk_size = request.form.get("chunk_size", 512)
    chunk_overlap = request.form.get("chunk_overlap", 50)
    cover_image_data = None
    cover_image_filename = None
    delete_cover = request.form.get("delete_cover") == "true"
    # 判断请求传过来的请求文件中是否有cover_image
    if "cover_image" in request.files:
        cover_file = request.files["cover_image"]
        if cover_file and cover_file.filename:
            # 读取文件的内容为二进制数据
            cover_image_data = cover_file.read()
            # 读取上传的文件的文件名
            cover_image_filename = cover_file.filename
            logger.info(
                f"收到知识库的新的封面图片：文件名:{cover_image_filename},大小:{len(cover_image_data)},内容类型:{cover_file.content_type}"
            )
    update_data = {}
    if name:
        update_data["name"] = name
    if description:
        update_data["description"] = description
    if chunk_size:
        update_data["chunk_size"] = chunk_size
    if chunk_overlap:
        update_data["chunk_overlap"] = chunk_overlap
    updated_kb = kb_service.update(
        kb_id=kb_id,
        cover_image_data=cover_image_data,
        cover_image_filename=cover_image_filename,
        delete_cover=delete_cover,
        **update_data,
    )
    return success_response(updated_kb, "更新知识库成功")


@bp.route("/kb/<kb_id>/cover")
@login_required
def kb_cover(kb_id):
    current_user, error = get_current_user_or_error()
    if error:
        return error
    kb_dict = kb_service.get_by_id(kb_id)
    if not kb_dict:
        return error_response("知识库未找到", 404)
    # 验证用户是否有删除此知识库的权限
    has_permission, err = check_ownership(
        kb_dict["user_id"], current_user["id"], "knowledgebase"
    )
    if not has_permission:
        return err
    # covers/2324bcd410ae433790de1b63eae9aba8.png
    cover_path = kb_dict.get("cover_image")
    if not cover_path:
        logger.info(f"知识库没有设置封面图片")
        abort(404)
    try:
        image_data = storage_service.download_file(cover_path)
        if not image_data:
            logger.error(f"读取封面图片失败:{cover_path}")
            abort(404)
        file_ext = os.path.splitext(cover_path)[1].lower()
        # 自定义映射，优先根据文件扩展名判断图片MIME类型
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_type_map.get(file_ext)
        return send_file(
            BytesIO(image_data),  # 图片数据
            mimetype=mime_type,  # 服务器返回响应体的内容类型
            as_attachment=False,  # 不以附件的形式传递
        )

    except FileNotFoundError as e:
        logger.error(f"封面图片未找到:{cover_path},错误：{e}")
        abort(404)
    except Exception as e:
        logger.error(f"访问知识库图片路径时出错:{cover_path},错误：{e}")
        abort(404)


@bp.route("/kb/<kb_id>")
@login_required
def kb_detail(kb_id):
    kb = kb_service.get_by_id(kb_id)
    if not kb:
        return redirect(url_for("knowledgebase.kb_list"))
    # 获取分页参数
    page, page_size = get_pagination_params(max_page_size=100)
    result = document_service.list_by_kb(kb_id, page=page, page_size=page_size)
    return render_template(
        "kb_detail.html",
        kb=kb,
        documents=result["items"],
        pagination=result["pagination"],
    )


@bp.route("/api/v1/knowledgebases/<kb_id>/navigation", methods=["GET"])
@api_login_required
@handle_api_error
def api_navigation(kb_id):
    current_user, error = get_current_user_or_error()
    if error:
        return error
    kb_dict = kb_service.get_by_id(kb_id)
    if not kb_dict:
        return error_response("知识库未找到", 404)
    has_permission, err = check_ownership(
        kb_dict["user_id"], current_user["id"], "knowledgebase"
    )
    if not has_permission:
        return err
    data = navigation_service.build_knowledgebase_navigation(kb_id)
    return success_response(data)
