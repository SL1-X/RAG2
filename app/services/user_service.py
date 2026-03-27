import hashlib
import re
import secrets
from datetime import datetime, timedelta
from app.models.user import User
from app.models.password_reset_code import PasswordResetCode
from app.services.base_service import BaseService
from app.services.email_service import email_service


class UserService(BaseService[User]):
    PASSWORD_RESET_CODE_EXPIRE_MINUTES = 10
    EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def hash_password(self, password):
        # 使用SHA256哈希算法进行哈希计算，
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def normalize_email(self, email):
        return (email or "").strip().lower()

    def validate_email(self, email):
        normalized_email = self.normalize_email(email)
        if not normalized_email:
            raise ValueError("邮箱不能为空")
        if not self.EMAIL_PATTERN.match(normalized_email):
            raise ValueError("邮箱格式不正确")
        return normalized_email

    def register(self, username, password, email):
        email = self.validate_email(email)
        if not username or not password:
            raise ValueError("用户名、邮箱和密码不能为空")
        if len(username) < 3:
            raise ValueError("用户名至少需要3个字符")
        if len(password) < 6:
            raise ValueError("密码至少需要6个字符")
        with self.transaction() as session:
            # 检查用户名是否存在
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                raise ValueError("用户名已经被占用")
            existing_email = session.query(User).filter_by(email=email).first()
            if existing_email:
                raise ValueError("邮件已经被占用")
            password_hash = self.hash_password(password)
            # .1 创建模型的实例
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                is_active=True,
            )
            # 用户的ID是在插入数据库的时候生成的
            # 添加新的用户到会话中
            session.add(user)
            # 刷新用户实例，得到用户ID
            session.flush()
            self.logger.info(f"用户{username}注册成功")
            return user.to_dict()

    def request_password_reset_code(self, email):
        email = self.validate_email(email)
        code = f"{secrets.randbelow(1000000):06d}"
        code_hash = self.hash_password(code)
        expire_at = datetime.utcnow() + timedelta(
            minutes=self.PASSWORD_RESET_CODE_EXPIRE_MINUTES
        )

        with self.transaction() as session:
            user = session.query(User).filter_by(email=email).first()
            if not user:
                raise ValueError("该邮箱未注册")

            # 避免一个邮箱并存多个可用验证码，旧验证码全部作废
            session.query(PasswordResetCode).filter_by(email=email, is_used=False).update(
                {"is_used": True}, synchronize_session=False
            )

            session.add(
                PasswordResetCode(
                    user_id=user.id,
                    email=email,
                    code_hash=code_hash,
                    expires_at=expire_at,
                    is_used=False,
                )
            )

        try:
            email_service.send_password_reset_code(
                to_email=email,
                code=code,
                expire_minutes=self.PASSWORD_RESET_CODE_EXPIRE_MINUTES,
            )
        except Exception as e:
            with self.transaction() as session:
                (
                    session.query(PasswordResetCode)
                    .filter_by(email=email, is_used=False, code_hash=code_hash)
                    .update({"is_used": True}, synchronize_session=False)
                )
            self.logger.error(f"发送重置密码验证码失败: {str(e)}")
            raise ValueError("验证码发送失败，请检查邮件配置后重试")

    def reset_password_by_code(self, email, code, new_password):
        email = self.validate_email(email)
        code = (code or "").strip()
        if not code:
            raise ValueError("验证码不能为空")
        if len(new_password or "") < 6:
            raise ValueError("新密码至少需要6个字符")

        with self.transaction() as session:
            latest_code = (
                session.query(PasswordResetCode)
                .filter_by(email=email, is_used=False)
                .order_by(PasswordResetCode.created_at.desc())
                .first()
            )
            if not latest_code:
                raise ValueError("请先获取验证码")

            if latest_code.expires_at < datetime.utcnow():
                latest_code.is_used = True
                raise ValueError("验证码已过期，请重新获取")

            if not self.verify_password(code, latest_code.code_hash):
                raise ValueError("验证码错误")

            user = session.query(User).filter_by(email=email).first()
            if not user:
                raise ValueError("用户不存在")

            user.password_hash = self.hash_password(new_password)
            latest_code.is_used = True
            self.logger.info(f"用户{user.username}通过邮箱验证码重置密码成功")

    def verify_password(self, password, password_hash):
        return self.hash_password(password) == password_hash

    def login(self, username, password):
        if not username or not password:
            raise ValueError("用户名和密码不能为空")
        with self.session() as db_session:
            existing_user = db_session.query(User).filter_by(username=username).first()
            if not existing_user:
                raise ValueError("此用户不存在")
            if not existing_user.is_active:
                raise ValueError("此用户已经被封禁")
            if not self.verify_password(password, existing_user.password_hash):
                raise ValueError("密码错误")
            self.logger.info(f"用户{username}登录成功")
            return existing_user.to_dict()

    def get_by_id(self, user_id):
        user = super().get_by_id(User, user_id)
        if user:
            return user.to_dict()
        else:
            return None


user_service = UserService()
