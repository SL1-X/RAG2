from flask import Flask, request, jsonify, session, redirect, url_for
from functools import wraps

app = Flask(__name__)


# 会在每次请求前执行
@app.before_request
def authenticate():
    # 定义需要权限认证的路径列表
    auth_required_paths = ["/kb", "/api/v1/kb"]
    # 获取当前请求的路径
    current_path = request.path
    # 检查当前的路径是否需要认证
    if current_path in auth_required_paths:
        # 如果需要认证，但当前用户又没有登录，则需要让用户去登录
        if "user_id" not in session:
            return redirect(url_for("auth.login", next=request.url))
    # 如果不需要权限认证，或者 需要认证但是用户已经登录了，则继续向下执行
    return None
