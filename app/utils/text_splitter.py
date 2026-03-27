from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

    def split_documents(self, documents, doc_id):
        if not documents:
            return []
        # 使用分割器分割文档
        chunks = self.splitter.split_documents(documents)
        results = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"  # 这是真正的分块的ID,就是文档ID_分块索引
            results.append(
                {
                    "id": chunk_id,
                    "text": chunk.page_content,
                    "chunk_index": i,
                    "metadata": chunk.metadata,  # 把
                }
            )
        return results
