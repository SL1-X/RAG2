from app.services.vectordb.base import VectorDBInterface
from app.config import Config
from app.utils.logger import get_logger
from app.utils.embedding_factory import EmbeddingFactory
from langchain_chroma import Chroma
import chromadb

logger = get_logger(__name__)


class ChromaVectorDB(VectorDBInterface):
    def __init__(
        self,
    ):
        self.persist_directory = Config.CHROMA_PERSIST_DIRECTORY
        self.embeddings = EmbeddingFactory.create_embeddings()
        logger.info(f"ChromaDB已经初始化，持久化目录为:{ self.persist_directory}")

    def get_or_create_collection(self, collection_name):
        # 获取或者创建向量存储对象
        vectorstore = Chroma(
            collection_name=collection_name,  # 集合的名称
            embedding_function=self.embeddings,  # 向量函数
            persist_directory=self.persist_directory,  # 向量数据存储目录
            collection_metadata={"hnsw:space": "cosine"},
        )
        return vectorstore

    def add_documents(self, collection_name, documents, ids):
        vectorstore = self.get_or_create_collection(collection_name)
        if ids:
            result_ids = vectorstore.add_documents(documents=documents, ids=ids)
        else:
            result_ids = vectorstore.add_documents(documents=documents)
        logger.info(f"已经向ChromDB集合{collection_name}添加了{len(documents)}个文档")
        return result_ids

    def delete_documents(self, collection_name, ids=None, filter=None):
        vectorstore = self.get_or_create_collection(collection_name)
        if ids:
            vectorstore.delete(ids=ids)
        elif filter:
            # ChromaDB 不支持直接使用 filter 参数删除
            # 需要先查询出符合条件的文档 IDs，然后删除
            try:
                # 方法1: 尝试通过 vectorstore 的 _collection 属性访问
                if hasattr(vectorstore, "_collection"):
                    collection = vectorstore._collection
                    # 使用 where 条件查询匹配的文档
                    # filter 格式: {"doc_id": "xxx"}
                    where = filter
                    results = collection.get(where=where)
                    if results and "ids" in results and results["ids"]:
                        matched_ids = results["ids"]
                        vectorstore.delete(ids=matched_ids)
                        logger.info(f"已通过filter条件删除{len(matched_ids)}个文档")
                    else:
                        logger.info(f"未找到匹配filter条件的文档，无需删除")
                else:
                    # 方法2: 直接使用 ChromaDB 客户端访问集合
                    client = chromadb.PersistentClient(path=self.persist_directory)
                    try:
                        collection = client.get_collection(name=collection_name)
                        # 使用 where 条件查询匹配的文档
                        where = filter
                        results = collection.get(where=where)
                        if results and "ids" in results and results["ids"]:
                            matched_ids = results["ids"]
                            vectorstore.delete(ids=matched_ids)
                            logger.info(f"已通过filter条件删除{len(matched_ids)}个文档")
                        else:
                            logger.info(f"未找到匹配filter条件的文档，无需删除")
                    except Exception as client_error:
                        logger.warning(
                            f"通过ChromaDB客户端访问集合失败: {client_error}"
                        )
                        # 如果集合不存在，说明没有数据需要删除
                        logger.info(f"集合{collection_name}不存在，无需删除")
            except Exception as e:
                logger.error(f"使用filter删除文档时出错: {e}", exc_info=True)
                raise
        else:
            raise ValueError(f"你既没有传ids,也没有传filter")
        logger.info(f"已经从ChromDB集合{collection_name}删除文档")

    def similarity_search_with_score(self, collection_name, query, k, filter):
        vectorstore = self.get_or_create_collection(collection_name)
        if filter:
            results = vectorstore.similarity_search_with_score(
                query=query, k=k, filter=filter
            )
        else:
            results = vectorstore.similarity_search_with_score(query=query, k=k)
        return results
