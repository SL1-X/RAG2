import os
from app import create_app

# 应用配置类
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    app = create_app()
    logger.info(f"正在启动RAGLite服务器在{Config.APP_HOST}:{Config.APP_PORT}")
    app.run(debug=Config.APP_DEBUG, host=Config.APP_HOST, port=Config.APP_PORT)
