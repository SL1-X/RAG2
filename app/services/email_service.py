import smtplib
from email.message import EmailMessage
from app.config import Config
from app.utils.logger import get_logger


class EmailService:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def _check_config(self):
        if not Config.SMTP_HOST:
            raise ValueError("邮件服务未配置: 缺少 SMTP_HOST")
        if not Config.SMTP_FROM:
            raise ValueError("邮件服务未配置: 缺少 SMTP_FROM")

    def send_password_reset_code(self, to_email: str, code: str, expire_minutes: int = 10):
        self._check_config()

        msg = EmailMessage()
        msg["Subject"] = "Smart QA System - 密码重置验证码"
        msg["From"] = Config.SMTP_FROM
        msg["To"] = to_email
        msg.set_content(
            f"您好，\n\n您的密码重置验证码是: {code}\n"
            f"该验证码将在 {expire_minutes} 分钟后过期。\n"
            "如果这不是您的操作，请忽略此邮件。"
        )

        if Config.SMTP_USE_SSL:
            with smtplib.SMTP_SSL(
                Config.SMTP_HOST, Config.SMTP_PORT, timeout=Config.SMTP_TIMEOUT
            ) as server:
                if Config.SMTP_USER:
                    server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
                server.send_message(msg)
            return

        with smtplib.SMTP(
            Config.SMTP_HOST, Config.SMTP_PORT, timeout=Config.SMTP_TIMEOUT
        ) as server:
            if Config.SMTP_USE_TLS:
                server.starttls()
            if Config.SMTP_USER:
                server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.send_message(msg)


email_service = EmailService()
