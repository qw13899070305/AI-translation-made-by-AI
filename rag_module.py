def add_documents(self, file_paths):
    # 获取现有文档指纹
    existing_fingerprints = set()
    if self.vectorstore is not None:
        existing_docs = self.vectorstore.get()['documents']
        existing_fingerprints = {doc[:100] for doc in existing_docs if doc}

    documents = []
    for path in file_paths:
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
            persist_directory=cfg.vector_db_path
        )
    else:
        self.vectorstore.add_documents(splits)
    self.vectorstore.persist()

    if self.use_bm25:
        all_docs = list(self.vectorstore.get()['documents'])
        self.bm25_retriever = BM25Retriever.from_texts(
            all_docs,
            preprocess_func=lambda text: text.lower()
        )
        self.bm25_retriever.k = 5

    if self.use_graph:
        self._build_graph(splits)

    print(f"Added {len(splits)} new text chunks.")