from app.utils.logger import get_logger
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from tempfile import NamedTemporaryFile
import os
import chardet
import pdfplumber
from langchain_core.documents import Document

logger = get_logger(__name__)


class DocumentLoader:
    @staticmethod
    def load_pdf(file_data):
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_data)
                tmp_path = tmp_file.name
            # try:  # 其实在内部会根据页面进行分割，一个页面对应一个Document
            #     loader = PyPDFLoader(tmp_path)
            #     # documents会在加载的时候自动设置metadata
            #     documents = loader.load()
            #     # print('documents', documents)
            #     for doc in documents:
            #         print(doc.page_content)
            #     return documents
            try:
                documents = []
                # pdfplumber解析PDF文档对中文支持较好
                with pdfplumber.open(tmp_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        # 可根据需要处理编码
                        documents.append(Document(page_content=text, metadata={"page": i+1}))
                        print(text)
                # for doc in documents:
                #     print(doc.page_content)
                return documents
            finally:
                # 最后手动删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"加载PDF时出错:{e}")
            raise ValueError(f"加载PDF时出错:{e}")

    @staticmethod
    def load_docx(file_data):
        try:
            with NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file_data)
                tmp_path = tmp_file.name
            try:
                loader = Docx2txtLoader(tmp_path)
                documents = loader.load()
                return documents
            finally:
                # 最后手动删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"加载docx时出错:{e}")
            raise ValueError(f"加载docx时出错:{e}")

    @staticmethod
    def load_text(file_data):
        try:
            with NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp_file:
                tmp_file.write(file_data)
                tmp_path = tmp_file.name
            try:
                print(f"临时文件路径: {tmp_path}")
                # 检测文件编码
                with open(tmp_path, 'rb') as f:
                    raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected.get('encoding', 'utf-8')
                print(f"检测到的编码: {encoding}")
                loader = TextLoader(tmp_path, encoding=encoding)
                documents = loader.load()
                print(f"加载的文档内容: {documents}")
                return documents
            finally:
                # 最后手动删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"加载text时出错:{e}")
            raise ValueError(f"加载text时出错:{e}")

    @staticmethod
    def load(file_data, file_type):
        file_type = file_type.lower()
        if file_type == "pdf":
            return DocumentLoader.load_pdf(file_data)
        if file_type == "docx":
            return DocumentLoader.load_docx(file_data)
        if file_type in ["txt", "md"]:
            return DocumentLoader.load_text(file_data)
        else:
            raise ValueError(f"不支持的文件类型:{file_type}")
