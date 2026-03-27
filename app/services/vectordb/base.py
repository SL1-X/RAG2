from abc import ABC, abstractmethod
from langchain_core.documents import Document


class VectorDBInterface(ABC):
    @abstractmethod
    def get_or_create_collection(self, collection_name):
        """
         获取或创建集合

        :param collection_name: 集合名称
        """
        pass

    @abstractmethod
    def add_documents(self, collection_name, documents, ids):
        """
         添加文档到向量存储

        :param collection_name: 集合名称
        """
        pass

    @abstractmethod
    def delete_documents(self, collection_name, ids=None, filter=None):
        """
         获取或创建集合

        :param collection_name: 集合名称
        """
        pass

    @abstractmethod
    def similarity_search_with_score(self, collection_name, query, k, filter):
        pass


# 为什么service有的模块抽象出package,有的以单个文件形式
