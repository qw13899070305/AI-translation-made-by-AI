import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import Config

cfg = Config()
os.makedirs(cfg.vector_db_path, exist_ok=True)

class RAGModule:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self.vectorstore = None
        self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(f"{cfg.vector_db_path}/chroma.sqlite3"):
            self.vectorstore = Chroma(persist_directory=cfg.vector_db_path, embedding_function=self.embeddings)
            print("Loaded existing vector database.")
        else:
            self.vectorstore = None
            print("No vector database found. Please upload documents first.")

    def add_documents(self, file_paths):
        documents = []
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        splits = text_splitter.split_documents(documents)
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings, persist_directory=cfg.vector_db_path)
        else:
            self.vectorstore.add_documents(splits)
        self.vectorstore.persist()
        print(f"Added {len(splits)} text chunks to the vector database.")

    def retrieve(self, query, k=3):
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]