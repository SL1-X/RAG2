from app.utils.db import db_session, db_transaction
from typing import Optional, TypeVar, Generic, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)
# 定义泛型的类型变量T
T = TypeVar("T")


# 定义基础服务器，支持泛型
class BaseService(Generic[T]):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def session(self):
        return db_session()

    def transaction(self):
        return db_transaction()

    def get_by_id(self, model_class: T, entity_id: str):
        with self.session() as db_session:
            try:
                return (
                    db_session.query(model_class)
                    .filter(model_class.id == entity_id)
                    .first()
                )
            except Exception as e:
                self.logger.error("获取ID对应的对象失败:{e}")
                return None

    def paginate_query(self, query, page, page_size, order_by):
        # 如果传入了排序字段
        if order_by is not None:
            query = query.order_by(order_by)
        # 获取结果集中的条数
        total = query.count()
        # 计算偏移量
        offset = (page - 1) * page_size
        # 获取当前页的数据
        items = query.offset(offset).limit(page_size).all()
        return {
            "items": [
                item.to_dict() if hasattr(item, "to_dict") else item for item in items
            ],
            "pagination": {"total": total, "page": page, "page_size": page_size},
        }
