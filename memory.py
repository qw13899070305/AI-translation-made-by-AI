import chromadb
from chromadb.config import Settings

class LongTermMemory:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./memory_db"
        ))
        self.collection = self.client.get_or_create_collection("conversations")
    
    def add(self, user_msg, assistant_msg):
        self.collection.add(
            documents=[f"用户: {user_msg}\n助手: {assistant_msg}"],
            ids=[str(hash(user_msg))[:16]]
        )
    
    def recall(self, query, k=3):
        results = self.collection.query(query_texts=[query], n_results=k)
        return results['documents'][0] if results['documents'] else []