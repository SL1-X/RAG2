from app.utils.logger import get_logger
from app.config import Config
from app.services.vectordb.chroma import ChromaVectorDB
from app.services.vectordb.milvus import MilvusVectorDB

logger = get_logger(__name__)


class VectorDBFactory:
    _instance = None

    @classmethod
    def create_vector_db(cls):
        vector_db_type = Config.VECTOR_DB_TYPE
        if vector_db_type == "chroma":
            return ChromaVectorDB()
        elif vector_db_type == "milvus":
            return MilvusVectorDB()
        else:
            raise ValueError(f"不支持向量数据库类型:{vector_db_type}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.create_vector_db()
        return cls._instance


def get_vector_db_service():
    return VectorDBFactory.get_instance()
