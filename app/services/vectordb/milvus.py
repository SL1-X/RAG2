from app.services.vectordb.base import VectorDBInterface
from app.config import Config
from app.utils.logger import get_logger
from app.utils.embedding_factory import EmbeddingFactory
from langchain_milvus import Milvus

logger = get_logger(__name__)


class MilvusVectorDB(VectorDBInterface):
    def __init__(
        self,
    ):
        self.connection_args = {"host": Config.MILVUS_HOST, "port": Config.MILVUS_PORT}
        self.embeddings = EmbeddingFactory.create_embeddings()
        logger.info(f"Milvus已经初始化，连接参数{self.connection_args}")

    def get_or_create_collection(self, collection_name):
        # 获取或者创建向量存储对象
        vectorstore = Milvus(
            collection_name=collection_name,  # 集合的名称
            embedding_function=self.embeddings,  # 向量函数
            connection_args=self.connection_args,  # 传递连接字符串
        )
        # vectorstore.similarity_search
        # vectorstore.similarity_search_with_score
        # 如果集合对象有_collection属性，尝试加载已有集合
        if hasattr(vectorstore, "_collection"):
            try:
                vectorstore._collection.load()
                logger.info(f"已经加载集合{collection_name}")
            except Exception as e:
                logger.info(f"集合可能不存在：{e}")
        return vectorstore

    def add_documents(self, collection_name, documents, ids):
        vectorstore = self.get_or_create_collection(collection_name)
        if ids:
            result_ids = vectorstore.add_documents(documents=documents, ids=ids)
        else:
            result_ids = vectorstore.add_documents(documents=documents)
        if hasattr(vectorstore, "_collection"):
            # 刷新集合
            vectorstore._collection.flush()
        logger.info(f"已经向ChromDB集合{collection_name}添加了{len(documents)}个文档")
        return result_ids

    def delete_documents(self, collection_name, ids=None, filter=None):
        vectorstore = self.get_or_create_collection(collection_name)
        if ids:
            vectorstore.delete(ids=ids)
        elif filter:
            expr = f'doc_id=="{filter["doc_id"]}"'
            vectorstore.delete(expr=expr)
        else:
            raise ValueError(f"你既没有传ids,也没有传filter")
        if hasattr(vectorstore, "_collection"):
            # 刷新集合
            vectorstore._collection.flush()
        logger.info(f"已经从ChromDB集合{collection_name}删除文档")

    def similarity_search_with_score(self, collection_name, query, k, filter):
        vectorstore = self.get_or_create_collection(collection_name)
        if hasattr(vectorstore, "_collection"):
            try:
                vectorstore._collection.load()
                logger.info(f"已经加载集合{collection_name}")
            except Exception as e:
                logger.info(f"集合可能不存在：{e}")
        if filter:
            expr = f'doc_id=="{filter["doc_id"]}"'
            results = vectorstore.similarity_search_with_score(
                query=query, k=k, expr=expr
            )
            print("filter_results", len(results))
        else:
            results = vectorstore.similarity_search_with_score(query=query, k=k)
        return results
