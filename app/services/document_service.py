from app.models.document import Document as DocumentModel
from app.services.base_service import BaseService
import os
from app.services.storage_service import storage_service
from app.services.parser_service import parser_service
from app.services.vector_service import vector_service
from app.config import Config
from app.models.knowledgebase import Knowledgebase
import uuid
from app.utils.text_splitter import TextSplitter
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentService(BaseService[DocumentModel]):
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def upload(self, kb_id, file_data, filename):
        file_uploaded = False
        file_path = None
        try:
            with self.session() as session:
                kb = (
                    session.query(Knowledgebase)
                    .filter(Knowledgebase.id == kb_id)
                    .first()
                )
                if not kb:
                    raise ValueError(f"知识库{kb_id}不存在")
            # 获取文件的扩展名
            file_ext = os.path.splitext(filename)[1]
            if file_ext:
                file_ext = file_ext[1:].lower()
            else:
                raise ValueError(f"文件名必须包含扩展名:{filename}")
            # 生成文档ID，用于标识唯一的文档
            doc_id = uuid.uuid4().hex[:32]
            # 构建文档存储的路径，以便于后续操作
            file_path = f"documents/{kb_id}/{doc_id}/{filename}"
            try:
                storage_service.upload_file(file_path, file_data)
                file_uploaded = True
            except Exception as storage_error:
                self.logger.error(f"上传文件到存储时发生了错误:{storage_error}")
                raise ValueError(f"文件上传失败:{str(storage_error)}")
            with self.transaction() as session:
                doc = DocumentModel(
                    id=doc_id,
                    kb_id=kb_id,
                    name=filename,
                    file_path=file_path,
                    file_type=file_ext,
                    file_size=len(file_data),
                    status="pending",  # 文档状态默认值为正在处理中
                )
                session.add(doc)
                session.flush()
                session.refresh(doc)
                doc_dict = doc.to_dict()
                self.logger.info(f"文档上传成功:{doc_id}")
            return doc_dict
        except Exception as e:
            if file_uploaded and file_path:
                try:
                    storage_service.delete_file(file_path)
                except Exception as delete_error:
                    self.logger.warning(f"删除已经上传的文件时出错:{delete_error}")
            raise

    def list_by_kb(self, kb_id, page, page_size, status=None):
        with self.session() as session:
            query = session.query(DocumentModel).filter(DocumentModel.kb_id == kb_id)
            if status:
                query.filter(DocumentModel.status == status)
            return self.paginate_query(
                query,  # 查询条件
                page=page,  # 当前 页码
                page_size=page_size,  # 返回多少条
                order_by=DocumentModel.created_at.desc(),  # 排序字段
            )

    def process(self, doc_id):
        with self.session() as session:
            doc = (
                session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            )
            if not doc:
                raise ValueError("文档{doc_id}不存在")
        self.logger.info(f"提交文档处理任务:{doc_id}")
        # 在线程池中异步提交处理任务
        future = self.executor.submit(self._process_document, doc_id)

        def exception_callback(future):
            try:
                # 获取异步线程的执行结果，发生异常的时候会在此抛出异常
                future.result()
            except Exception as e:
                self.logger.error(f"文档处理任务异常:{doc_id},错误:{e}", exc_info=True)

        future.add_done_callback(exception_callback)

    def _process_document(self, doc_id):
        try:
            self.logger.info(f"开始处理文档:{doc_id}")
            with self.transaction() as session:
                doc = (
                    session.query(DocumentModel)
                    .filter(DocumentModel.id == doc_id)
                    .first()
                )
                if not doc:
                    self.logger.error(f"未找到文档:{doc_id}")
                    return
                # 如果文档已经处理过了，需要重置为原始的状态
                need_cleanup = doc.status in ["completed", "failed"]
                # 如果要恢复为原始状态的话，需要重置 状态，分块大小，错误处理清空
                if need_cleanup:
                    doc.chunk_count = 0
                    doc.error_message = ""
                # 把状态更新为处理中
                doc.status = "processing"
                # 刷新到数据库，这个时候写入未提交的修改
                session.flush()
                # 提前取出相关的数据
                kb_id = doc.kb_id
                file_path = doc.file_path
                file_type = doc.file_type
                doc_name = doc.name
                # 一个知识库对应一个chromadb的向量数据库的集合
                collection_name = f"kb_{doc.kb_id}"
                kb = (
                    session.query(Knowledgebase)
                    .filter(Knowledgebase.id == kb_id)
                    .first()
                )
                if not kb:
                    raise ValueError(f"知识库不存在")
                kb_chunk_size = kb.chunk_size
                kb_chunk_overlap = kb.chunk_overlap
            # 如果要清理旧数据
            if need_cleanup:
                try:
                    # 调用向量服务，删除指定集合中指定向档ID下面的所有的向量数据
                    vector_service.delete_documents(
                        collection_name=collection_name, filter={"doc_id": doc_id}
                    )
                except Exception as e:
                    self.logger.warning(f"删除向量数据库失败:{e}")

            self.logger.info(f"文档{doc_id}状态已经更新为processing状态了")
            # 从存储中下载文件内容
            file_data = storage_service.download_file(file_path)
            # 解析文件，根据文件类型按不同的方法得到文本内容
            langchain_docs = parser_service.parse(file_data, file_type)
            self.logger.info(f"加载到{len(langchain_docs)}个文档")
            if not langchain_docs:
                raise ValueError(f"未能抽取到任何文本内容")
            if not self._is_english_document(langchain_docs):
                raise ValueError("当前系统仅支持英文文档，请上传英文内容后重试。")
            # 创建文本的分块器，指定知识库参数
            parent_splitter = TextSplitter(
                chunk_size=kb_chunk_size, chunk_overlap=kb_chunk_overlap
            )
            # 第一层：父分块（章节/大段）
            parent_chunks = parent_splitter.split_documents(langchain_docs, doc_id=doc_id)
            if not parent_chunks:
                raise ValueError(f"文档{doc_id}未能成功分层切分（父分块为空）")
            child_chunk_size = self._calc_child_chunk_size(kb_chunk_size)
            child_chunk_overlap = self._calc_child_chunk_overlap(child_chunk_size, kb_chunk_overlap)
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=child_chunk_size,
                chunk_overlap=child_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            )
            self.logger.info(
                f"父子分层切分参数: parent_size={kb_chunk_size}, parent_overlap={kb_chunk_overlap}, "
                f"child_size={child_chunk_size}, child_overlap={child_chunk_overlap}"
            )
            # 初始化一个列表用于存放默认换后的langchain document对象
            documents = []
            chunk_ids = []
            child_counter = 0
            for parent_idx, parent in enumerate(parent_chunks):
                parent_text = (parent.get("text") or "").strip()
                if not parent_text:
                    continue
                parent_id = f"{doc_id}_p_{parent_idx}"
                child_texts = child_splitter.split_text(parent_text)
                for child_idx, child_text in enumerate(child_texts):
                    child_text = (child_text or "").strip()
                    if not child_text:
                        continue
                    child_id = f"{parent_id}_c_{child_idx}"
                    doc_obj = Document(
                        page_content=child_text,
                        metadata={
                            "doc_id": doc_id,  # 文档ID
                            "doc_name": doc_name,  # 文档名称
                            "chunk_index": child_counter,  # 兼容旧字段：全局子分块序号
                            "id": child_id,  # 子分块ID
                            "chunk_id": child_id,  # 子分块ID
                            "node_type": "child",  # 节点类型
                            "parent_id": parent_id,  # 父分块ID（关联键）
                            "parent_chunk_index": parent_idx,  # 父分块序号
                            "child_chunk_index": child_idx,  # 父分块内子序号
                            "parent_content": parent_text,  # 父分块正文（命中后回填上下文）
                            "doc_lang": "en",  # 文档语言
                        },
                    )
                    documents.append(doc_obj)
                    chunk_ids.append(child_id)
                    child_counter += 1
            if not documents:
                raise ValueError(f"文档{doc_id}未能成功分层切分（子分块为空）")
            self.logger.info(f"父分块{len(parent_chunks)}个，子分块{len(documents)}个")
            # 调用向量服务，将分块后的文档对象写入向量数据库
            vector_service.add_documents(
                collection_name=collection_name, documents=documents, ids=chunk_ids
            )
            with self.transaction() as session:
                doc = (
                    session.query(DocumentModel)
                    .filter(DocumentModel.id == doc_id)
                    .first()
                )
                if doc:
                    doc.status = "completed"
                    doc.chunk_count = len(documents)
            self.logger.info(f"文档{doc_id}处理完成,子分块数量为{len(documents)}")

        except Exception as e:
            # 如果文档处理了，则需要更新文档的状态为失败，并且记录错误信息
            with self.transaction() as session:
                doc = (
                    session.query(DocumentModel)
                    .filter(DocumentModel.id == doc_id)
                    .first()
                )
                if doc:
                    doc.status = "failed"
                    doc.error_message = str(e)[:500]
                    session.flush()
                    session.refresh(doc)
            self.logger.error(f"处理文档{doc_id}时发生了错误:{e}")

    @staticmethod
    def _is_english_document(langchain_docs, min_letters: int = 80) -> bool:
        """
        仅允许英文文档入库：
        - 英文字母数量需要达到最小阈值；
        - CJK 字符占比不能过高。
        """
        text = " ".join((d.page_content or "") for d in (langchain_docs or []))
        if not text.strip():
            return False
        sample = text[:20000]
        letters = len(re.findall(r"[A-Za-z]", sample))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
        total = max(1, len(sample))
        cjk_ratio = cjk / total
        return letters >= min_letters and cjk_ratio <= 0.08

    @staticmethod
    def _calc_child_chunk_size(parent_chunk_size: int) -> int:
        parent_chunk_size = max(200, int(parent_chunk_size or 0))
        # 英文场景：子分块偏小，提升向量召回精度
        return max(120, min(320, parent_chunk_size // 4))

    @staticmethod
    def _calc_child_chunk_overlap(child_chunk_size: int, parent_overlap: int) -> int:
        base = max(20, int(parent_overlap or 0) // 4)
        return max(20, min(child_chunk_size // 3, base))

    def delete(self, doc_id):
        """
        删除文档的时候，要删除向量数据库中的向量数据，上传的文件，删除数据库里的文档数据
        """
        with self.session() as session:
            doc = (
                session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            )
            if not doc:
                raise ValueError(f"文档{doc_id}不存在")
            kb_id = doc.kb_id
            file_path = doc.file_path
            collection_name = f"kb_{kb_id}"
        try:
            # 1.删除向量数据库中的相关向量数据
            vector_service.delete_documents(
                collection_name=collection_name, filter={"doc_id": doc_id}
            )
            self.logger.info(f"已经删除文档{doc_id}的向量数据")
        except Exception as e:
            self.logger.warning(f"删除向量数据库失败:{str(e)}")

        # 2.删除存储的文件
        storage_service.delete_file(file_path)
        self.logger.info(f"已经删除文档{doc_id}的存储文件:{file_path}")
        # 3.删除数据库的记录
        with self.transaction() as session:
            doc = (
                session.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            )
            if doc:
                session.delete(doc)
                session.flush()
                self.logger.info(f"已经删除文档{doc_id}的数据记录")


document_service = DocumentService()
