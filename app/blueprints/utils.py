from flask import jsonify, request
from functools import wraps
from app.utils.logger import get_logger
from app.utils.auth import get_current_user

logger = get_logger(__name__)


# 定义成功时的响应函数
def success_response(data=None, message="success"):
    return jsonify({"code": 200, "message": message, "data": data}), 200


# 定义失败时的响应函数
def error_response(message: str, code: int = 500):
    return jsonify({"code": code, "message": message, "data": None}), code


def handle_api_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"ValueError in {func.__name__}:{e}")
            return error_response(str(e), 400)
        except Exception as e:
            logger.warning(f"ValueError in {func.__name__}:{e}")
            return error_response(str(e), 500)

    return wrapper


def get_pagination_params(max_page_size=100):
    # 获取请求中的查询参数page参数的值，表示当前查询第几页，默认为1，转为整数
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 10))
    page = max(1, page)
    page_size = max(1, min(page_size, max_page_size))
    return page, page_size


def get_current_user_or_error():
    current_user = get_current_user()
    if not current_user:
        return None, error_response("用户无权访问", 401)
    return current_user, None


def check_ownership(entity_user_id, current_user_id, entity_name):
    if entity_user_id != current_user_id:
        return False, error_response(f"未授权访问此{entity_name}", 403)
    return True, None


def require_json_body():
    data = request.get_json()
    if not data:
        return None, error_response("请求体不能为空", 400)
    return data, None
