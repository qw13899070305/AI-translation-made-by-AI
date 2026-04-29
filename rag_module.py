import os
import re
import numpy as np
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
import networkx as nx

from config import Config

cfg = Config()
os.makedirs(cfg.vector_db_path, exist_ok=True)

class RAGModule:
    def __init__(self, use_bm25=True, use_iterative=True, use_rerank=True,
                 use_graph=True, use_hyde=True, use_self_correction=True):
        self.embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        self.vectorstore = None
        self.bm25_retriever = None
        self.use_bm25 = use_bm25
        self.use_iterative = use_iterative
        self.use_rerank = use_rerank
        self.use_graph = use_graph
        self.use_hyde = use_hyde
        self.use_self_correction = use_self_correction

        # 知识图谱相关
        self.graph = nx.Graph()
        self.entity_index = defaultdict(set)  # 实体 -> 文档ID集合

        self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(f"{cfg.vector_db_path}/chroma.sqlite3"):
            self.vectorstore = Chroma(
                persist_directory=cfg.vector_db_path,
                embedding_function=self.embeddings
            )
            print("Loaded existing vector database.")
            # 如果之前有图索引，也可以考虑重建，但为了简单，这里只保留向量库
        else:
            self.vectorstore = None
            print("No vector database found. Please upload documents first.")

    # -------------------- 文档添加 & 图谱构建 --------------------
    def add_documents(self, file_paths):
        documents = []
        for path in file_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)

        # 向量库更新
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=cfg.vector_db_path
            )
        else:
            self.vectorstore.add_documents(splits)
        self.vectorstore.persist()

        # BM25 索引更新
        if self.use_bm25:
            all_docs = list(self.vectorstore.get()['documents'])
            self.bm25_retriever = BM25Retriever.from_texts(
                all_docs,
                preprocess_func=lambda text: text.lower()
            )
            self.bm25_retriever.k = 5

        # 知识图谱构建
        if self.use_graph:
            self._build_graph(splits)

        print(f"Added {len(splits)} text chunks to the vector database.")

    def _extract_entities(self, text):
        """简单的中英文实体提取：英文=大写词序列，中文=连续汉字的二元组"""
        entities = set()
        # 英文单词实体
        english_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.update(english_words)
        # 中文词组：连续汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        entities.update(chinese_chars)
        return list(entities)

    def _build_graph(self, documents):
        """构建文档实体共现图"""
        doc_texts = [doc.page_content for doc in documents]
        # 为每个文档分配 ID
        for i, text in enumerate(doc_texts):
            entities = self._extract_entities(text)
            for ent in entities:
                self.entity_index[ent].add(i)
        # 构建实体-实体共现图
        for i, text in enumerate(doc_texts):
            entities = self._extract_entities(text)
            for j in range(len(entities)):
                for k in range(j+1, len(entities)):
                    e1, e2 = entities[j], entities[k]
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]['weight'] += 1
                    else:
                        self.graph.add_edge(e1, e2, weight=1)

    # -------------------- 检索核心 --------------------
    def _get_retriever(self, k=5, weight_semantic=0.7):
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        if self.use_bm25 and self.bm25_retriever is not None:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, self.bm25_retriever],
                weights=[weight_semantic, 1 - weight_semantic]
            )
            return ensemble_retriever
        return semantic_retriever

    def _rerank_documents(self, query, docs, top_k=3):
        """重排序：用嵌入相似度计算"""
        if not docs:
            return docs
        query_emb = self.embeddings.embed_query(query)
        scored = []
        for doc in docs:
            doc_emb = self.embeddings.embed_query(doc.page_content)
            sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8)
            scored.append((doc, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def _generate_hyde_document(self, query):
        """生成假设性文档：用简单规则生成包含关键实体的句子，模拟 LLM 生成，但我们没有 LLM，所以用模板"""
        # 抽取查询中的关键实体
        entities = self._extract_entities(query)
        # 生成一个假设文档，包含这些实体
        template = " ".join([f"{ent} is relevant to the question." for ent in entities if len(ent) > 2])
        return template if template else query

    def _graph_retrieve(self, query_doc_ids, k=3):
        """基于知识图谱检索相关文档：找出与查询实体的邻居实体所关联的文档"""
        entities = self._extract_entities(query_doc_ids[0] if query_doc_ids else "")
        related_docs = set()
        for ent in entities:
            if ent in self.graph:
                # 找该实体的一跳邻居
                neighbors = list(self.graph.neighbors(ent))
                for neigh in neighbors:
                    if neigh in self.entity_index:
                        related_docs.update(self.entity_index[neigh])
        # 返回文档 ID 对应的文本
        docs = list(self.vectorstore.get()['documents'])
        retrieved = [docs[i] for i in related_docs if i < len(docs)]
        return retrieved[:k]

    def _iterative_retrieve(self, query, k=3):
        """自适应迭代检索 + HYDE + Graph"""
        # HYDE 增强查询
        if self.use_hyde:
            hyde_doc = self._generate_hyde_document(query)
            # 把假设文档加到查询中作为补充检索词
            enhanced_query = f"{query} {hyde_doc}"
        else:
            enhanced_query = query

        retriever = self._get_retriever(k=k)
        docs = retriever.get_relevant_documents(enhanced_query)

        # 如果检索结果少，使用迭代扩展
        if len(docs) < k:
            # 分解查询词
            words = [w for w in re.findall(r'\w+', query) if len(w) > 3]
            for w in words[:3]:
                extra_docs = retriever.get_relevant_documents(w)
                exist_texts = {d.page_content for d in docs}
                for d in extra_docs:
                    if d.page_content not in exist_texts:
                        docs.append(d)
                        exist_texts.add(d.page_content)

        # GraphRAG 增强
        if self.use_graph:
            # 从已有文档中提取实体，然后走图谱邻居
            graph_docs = self._graph_retrieve([d.page_content for d in docs], k=k)
            # 转换为 Document 对象
            for gdoc_text in graph_docs:
                if gdoc_text not in [d.page_content for d in docs]:
                    docs.append(Document(page_content=gdoc_text))

        return docs

    def _self_correction(self, query, retrieved_docs, response):
        """自校正：检查检索到的文档与潜在冲突"""
        if not retrieved_docs:
            return response
        # 简单检查：如果回答中包含明显与检索文档矛盾的词，尝试纠正
        # 这里只做示例：如果回答中出现“不知道”，但文档充分，提示补充
        if "不知道" in response and len(retrieved_docs) >= 2:
            correction = "根据已有资料，可以补充如下：" + retrieved_docs[0][:100]
            return response + "\n\n【补充信息】" + correction
        return response

    # -------------------- 公开接口 --------------------
    def retrieve(self, query, k=3):
        if self.vectorstore is None:
            return []

        if self.use_iterative:
            docs = self._iterative_retrieve(query, k=k)
        else:
            retriever = self._get_retriever(k=k)
            docs = retriever.get_relevant_documents(query)

        if self.use_rerank and docs:
            docs = self._rerank_documents(query, docs, top_k=k)

        result_texts = [doc.page_content for doc in docs[:k]]
        return result_texts

    def augmented_prompt(self, query, k=3):
        """返回增强后的 prompt 片段"""
        retrieved = self.retrieve(query, k)
        context = "\n".join(retrieved)
        if context:
            return f"基于以下信息：\n{context}\n\n问题：{query}"
        return query