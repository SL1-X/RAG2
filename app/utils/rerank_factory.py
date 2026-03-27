from app.utils.logger import get_logger
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = get_logger(__name__)


class BaseReranker:
    def rerank(self, query, documents, top_k):
        raise NotImplementedError


class LocalReranker(BaseReranker):
    def __init__(self):
        # https://www.modelscope.cn/models/cross-encoder/ms-marco-MiniLM-L6-v2
        # 可下载完整模型库到本地
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    def rerank(self, query, documents, top_k):
        if not documents:
            return []
        top_k = top_k or len(documents)
        try:
            # 构建输入对，每个文档和查询组成一组
            pairs = [[query, doc.page_content] for doc in documents]
            # 用cross-encoder这个模型计算每个对的相关性分数
            scores = self.reranker.predict(pairs)
            # 把分数转成列表
            scores = list(scores)
            scores_float = [float(score) for score in scores]
            # 计算分数是最小值
            min_score = min(scores_float) if scores_float else 0.0
            # 计算分数中的最大值
            max_score = max(scores_float) if scores_float else 1.0
            # 对分数进行归一化
            normalized_scores = [
                (
                    (score - min_score) / (max_score - min_score)
                    if max_score > min_score
                    else 0.5
                )
                for score in scores_float
            ]
            doc_scores = [
                (doc, max(0.0, min(1.0, score)))
                for doc, score in zip(documents, normalized_scores)
            ]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"CrossEncoder重排序：已经重排序了{len(doc_scores)}个文档")
            return doc_scores[:top_k]
        except Exception as e:
            logger.error(f"CrossEncoder重排序出错:{str(e)}")
            return [(doc, 0.5) for doc in documents[:top_k]]


class RerankFactory:

    @staticmethod
    def create_reranker(settings):
        return LocalReranker()
