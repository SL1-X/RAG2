from sqlalchemy import create_engine
from sqlalchemy import text
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
        _migrate_knowledgebase_name_unique_scope()
    except Exception as e:
        logger.error(f"初始化数据库失败:{e}")
        raise


def _migrate_knowledgebase_name_unique_scope():
    """把 knowledgebase.name 的全局唯一迁移为 (user_id, name) 联合唯一。"""
    if engine.dialect.name != "mysql":
        return

    with engine.begin() as conn:
        current_db = conn.execute(text("SELECT DATABASE()")).scalar()
        if not current_db:
            return

        rows = (
            conn.execute(
                text(
                    """
                    SELECT
                        INDEX_NAME,
                        NON_UNIQUE,
                        GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS columns_csv
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = :schema_name
                      AND TABLE_NAME = 'knowledgebase'
                    GROUP BY INDEX_NAME, NON_UNIQUE
                    """
                ),
                {"schema_name": current_db},
            )
            .mappings()
            .all()
        )

        unique_name_only_indexes = [
            row["INDEX_NAME"]
            for row in rows
            if row["NON_UNIQUE"] == 0
            and row["columns_csv"] == "name"
            and row["INDEX_NAME"] != "PRIMARY"
        ]
        has_user_name_unique = any(
            row["NON_UNIQUE"] == 0 and row["columns_csv"] == "user_id,name"
            for row in rows
        )
        has_non_unique_name_index = any(
            row["INDEX_NAME"] == "ix_knowledgebase_name"
            and row["columns_csv"] == "name"
            and row["NON_UNIQUE"] == 1
            for row in rows
        )

        for index_name in unique_name_only_indexes:
            conn.execute(text(f"ALTER TABLE knowledgebase DROP INDEX `{index_name}`"))
            logger.info(f"已删除 knowledgebase 旧唯一索引: {index_name}")

        if not has_user_name_unique:
            conn.execute(
                text(
                    """
                    ALTER TABLE knowledgebase
                    ADD CONSTRAINT uq_knowledgebase_user_id_name UNIQUE (user_id, name)
                    """
                )
            )
            logger.info("已新增 knowledgebase 联合唯一约束: uq_knowledgebase_user_id_name")

        if not has_non_unique_name_index:
            conn.execute(
                text("CREATE INDEX ix_knowledgebase_name ON knowledgebase (name)")
            )
            logger.info("已新增 knowledgebase 普通索引: ix_knowledgebase_name")
