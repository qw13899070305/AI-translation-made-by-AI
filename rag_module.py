import os
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from config import Config

cfg = Config()

class RAGModule:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self.vectorstore = None
        self.bm25_retriever = None
        self.use_bm25 = True
        self.use_graph = False  # GraphRAG 暂未实现，可后续扩展
        self.persist_dir = cfg.vector_db_path
        os.makedirs(self.persist_dir, exist_ok=True)
        # 尝试加载已有向量库
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                print("✅ Loaded existing vector database.")
            except Exception as e:
                print(f"⚠️ Failed to load vector DB: {e}")

    def add_documents(self, file_paths: List[str]):
        """添加文档到向量库（去重）"""
        existing_fingerprints = set()
        if self.vectorstore is not None:
            existing_docs = self.vectorstore.get()['documents']
            existing_fingerprints = {doc[:100] for doc in existing_docs if doc}

        documents = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"⚠️ File not found: {path}")
                continue
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            for doc in loader.load():
                fingerprint = doc.page_content[:100]
                if fingerprint not in existing_fingerprints:
                    documents.append(doc)
                    existing_fingerprints.add(fingerprint)
                else:
                    print(f"Skipping duplicate document: {path} (fingerprint match)")

        if not documents:
            print("No new documents to add.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(splits)
        self.vectorstore.persist()

        # 重建 BM25 索引
        if self.use_bm25:
            all_docs = list(self.vectorstore.get()['documents'])
            if all_docs:
                self.bm25_retriever = BM25Retriever.from_texts(
                    all_docs,
                    preprocess_func=lambda text: text.lower()
                )
                self.bm25_retriever.k = 5

        print(f"✅ Added {len(splits)} new text chunks.")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """混合检索：语义检索 + BM25（如果启用）"""
        if self.vectorstore is None:
            return []

        # 语义检索
        semantic_docs = self.vectorstore.similarity_search(query, k=k)
        semantic_texts = [doc.page_content for doc in semantic_docs]

        # BM25 检索
        bm25_texts = []
        if self.use_bm25 and self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            bm25_texts = [doc.page_content for doc in bm25_docs[:k]]

        # 合并去重（保留顺序：先语义，后 BM25）
        all_texts = semantic_texts + bm25_texts
        unique_texts = []
        seen = set()
        for t in all_texts:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)
        return unique_texts[:k]

    def clear(self):
        """清空向量库（慎用）"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None
        self.bm25_retriever = None
        print("🗑️ RAG storage cleared.")