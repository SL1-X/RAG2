from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app.utils.logger import get_logger
from app.services.user_service import user_service

logger = get_logger(__name__)

bp = Blueprint("auth", __name__)


@bp.route("/")
def home():
    return render_template("home.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")
        email = request.form.get("email", "").strip()
        if password != password_confirm:
            flash("两次输入的密码不一致", "error")
            return render_template("register.html")
        try:
            user_service.register(username, password, email)
            flash("用户注册成功", "success")
            return redirect(url_for("auth.home"))
        except ValueError as e:
            logger.error(f"注册失败:{str(e)}")
            flash(str(e), "error")
        except Exception as e:
            logger.error(f"注册失败:{str(e)}")
            flash("注册失败,请稍后重试", "error")

    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        # 获取登录后重定向的目标页面(优先是读取表单，其次是读URL参数)
        next_url = request.form.get("next") or request.args.get("next")
        try:
            user = user_service.login(username, password)
            # 这个session是flask里的会话字典，和数据库里的会话没关系
            # 当登录成功后，把用户ID放在了会话对象里
            session["user_id"] = user["id"]
            # 设置会话为永久有效，默认是31天
            session.permanent = True
            flash("登录成功", "success")
            return redirect(next_url or url_for("auth.home"))
        except ValueError as e:
            logger.error(f"登录失败:{str(e)}")
            flash(str(e), "error")
        except Exception as e:
            logger.error(f"登录失败:{str(e)}")
            flash("登录失败,请稍后重试", "error")
    # 如果是Get请求或者登录失败，获取URL参数中的next
    next_url = request.args.get("next")
    return render_template("login.html", next_url=next_url)


@bp.route("/logout")
def logout():
    # 清除会话
    session.clear()
    flash("已成功退出", "success")
    return redirect(url_for("auth.home"))


@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        try:
            user_service.request_password_reset_code(email)
            flash("验证码已发送到邮箱，请查收", "success")
            return redirect(url_for("auth.reset_password", email=email))
        except ValueError as e:
            logger.error(f"发送重置密码验证码失败: {str(e)}")
            flash(str(e), "error")
        except Exception as e:
            logger.error(f"发送重置密码验证码失败: {str(e)}")
            flash("发送失败，请稍后重试", "error")

    return render_template("forgot_password.html")


@bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        code = request.form.get("code", "").strip()
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")

        if password != password_confirm:
            flash("两次输入的密码不一致", "error")
            return render_template("reset_password.html", email=email)

        try:
            user_service.reset_password_by_code(email, code, password)
            flash("密码重置成功，请重新登录", "success")
            return redirect(url_for("auth.login"))
        except ValueError as e:
            logger.error(f"重置密码失败: {str(e)}")
            flash(str(e), "error")
        except Exception as e:
            logger.error(f"重置密码失败: {str(e)}")
            flash("重置失败，请稍后重试", "error")

        return render_template("reset_password.html", email=email)

    email = request.args.get("email", "").strip()
    return render_template("reset_password.html", email=email)
