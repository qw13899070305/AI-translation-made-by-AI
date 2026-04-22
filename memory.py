import chromadb
from chromadb.config import Settings
import uuid

class LongTermMemory:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./memory_db"
        ))
        self.collection = self.client.get_or_create_collection("conversations")
    
    def add(self, user_msg, assistant_msg):
        # 使用 UUID 代替截断的哈希值，彻底避免碰撞
        unique_id = str(uuid.uuid4())
        self.collection.add(
            documents=[f"用户: {user_msg}\n助手: {assistant_msg}"],
            ids=[unique_id]
        )
    
    def recall(self, query, k=3):
        results = self.collection.query(query_texts=[query], n_results=k)
        return results['documents'][0] if results['documents'] else []