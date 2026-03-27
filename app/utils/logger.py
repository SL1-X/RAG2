import logging
import sys

# 用于日志文件轮转
from logging.handlers import RotatingFileHandler

from pathlib import Path
from typing import Optional
from app.config import Config


class LoggerManager:
    """日志管理器"""

    # 日志格式化字符串
    FORMAT_STRING = (
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    #  最大日志文件大小 10M
    MAX_BYTES = 10 * 1024 * 1024
    # 日志文件保留份数
    BACKUP_COUNT = 5

    def __init__(self):
        self.log_dir = Path(Config.LOG_DIR)
        self.log_file = Path(Config.LOG_FILE)
        self.level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
        self.enable_file = Config.LOG_ENABLE_FILE
        self.enable_console = Config.LOG_ENABLE_CONSOLE
        self._initialize()

    def _initialize(self):
        # 获取根日志记录器
        root_logger = logging.getLogger()
        # 移除原有的所有处理器，防止重复添加
        root_logger.handlers.clear()
        # 设置日志级别
        root_logger.setLevel(self.level)
        # 创建日志格式器
        formatter = logging.Formatter(self.FORMAT_STRING)
        # 如果启动控制台日志的话
        if self.enable_console:
            # 创建控制台日志处理器 sys.stdout就是指的标准输出
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        # 如果启动文件日志的话
        if self.enable_file:
            # mkdir创建目录，parents=True如果父目录不存在，则自动创建所有的父目录，exist_ok如果目录已经存在，不抛出错误
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            # 创建日志文件路径 1.不优雅 2.还得考虑操作系统兼容 \ /
            log_path = self.log_dir / self.log_file
            # 创建控制台日志处理器 sys.stdout就是指的标准输出
            file_handler = RotatingFileHandler(
                str(log_path),  # 日志文件路径
                maxBytes=self.MAX_BYTES,  # 最大文件大小
                backupCount=self.BACKUP_COUNT,  # 最多保留几份最新的日志文件
                encoding="utf-8",  # 编码
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        # 捕获warning模块的警告作为文件
        logging.captureWarnings(True)

    def get_logger(self, name):
        if name is None:
            return logging.getLogger()
        # 返回指定的名称的日志处理器
        return logging.getLogger(name)


loggerManager = LoggerManager()


def get_logger(name):
    return loggerManager.get_logger(name)
