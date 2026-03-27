from sqlalchemy import create_engine
from app.config import Config
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from app.models import *
from app.models.base import Base

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_database_url():
    return (
        f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}"
        #f"mysql+pymysql://{Config.DB_USER}"
        f"@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}?charset={Config.DB_CHARSET}"
    )


print(f"数据库连接URL: {get_database_url()}")

# 创建数据库的连接引擎
engine = create_engine(
    get_database_url(),  # 数据库的连接地址URL
    poolclass=QueuePool,  # 数据库连接池
    pool_size=10,  # 数据库连接池中的最大连接数
    max_overflow=20,  # 允许 最大溢出连接 数量为20
    pool_pre_ping=True,  # 连接每次获取 前先检查可用性
    pool_recycle=3600,  # 3600秒如果不使用回收连接
    echo=False,  # 不输出SQL日志
)

# 创建会话工厂，用于生成会话session对象
Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def db_session():
    # 创建会话的实例
    session = Session()
    try:
        # 将session交给调用方使用
        yield session
    except Exception as e:
        logger.error(f"数据库会话错误:{e}")
        raise
    finally:
        session.close()


@contextmanager
def db_transaction():
    # 创建会话的实例
    session = Session()
    try:
        # 将session交给调用方使用
        yield session
        # 事务正常结束可以自动提交
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"数据库事务错误:{e}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"数据库会话错误:{e}")
        raise
    finally:
        session.close()


def init_db():
    try:
        # 使用引擎来创建数据库的表结构
        Base.metadata.create_all(engine)
    except Exception as e:
        logger.error(f"初始化数据库失败:{e}")
        raise
