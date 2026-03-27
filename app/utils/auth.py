from flask import session, g, url_for, redirect, request, jsonify
from app.utils.logger import get_logger
from app.services.user_service import user_service
from functools import wraps

logger = get_logger(__name__)


# 获取当前登录的用户
def get_current_user():
    if not hasattr(g, "current_user"):
        #  session["user_id"]
        if "user_id" in session:  # session["user_id"]
            g.current_user = user_service.get_by_id(session["user_id"])
        else:
            g.current_user = None
    return g.current_user


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 当用户访问request.url的时候，比如说访问/kb的时候，要判断用户是否已经登录过
        # 如果没有登录，则重定向到登录页登录，并且携带next参数为/kb
        # 未来当登录成功后还要自动跳回/kb
        if "user_id" not in session:
            return redirect(url_for("auth.login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# 定义 API 登录认证装饰器
def api_login_required(f):
    """
    API 登录装饰器
    用于 API 端点，返回 JSON 错误响应而不是重定向
    """

    # 保持被装饰函数的元信息
    @wraps(f)
    # 定义装饰后的函数
    def decorated_function(*args, **kwargs):
        # 判断 session 中是否没有 user_id，未登录状态
        if "user_id" not in session:
            # 返回未授权的 JSON 错误响应和 401 状态码
            return jsonify({"code": 401, "message": "未授权", "data": None}), 401
        # 如果已登录，正常执行原函数
        return f(*args, **kwargs)

    # 返回包装后的函数
    return decorated_function
