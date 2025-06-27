import os
from chromadb import Client
from chromadb.config import Settings
from typing import List, Dict
from chromadb import PersistentClient


# Set absolute path for chroma_db folder relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "../../chroma_db")
CHROMA_DB_DIR = os.path.abspath(CHROMA_DB_DIR)

print(f"📂 Chroma DB will be stored at: {CHROMA_DB_DIR}")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)  # Create folder if it doesn't exist

# Initialize Chroma client with persistence
client = PersistentClient(path=CHROMA_DB_DIR)
# Create or get the collection
collection = client.get_or_create_collection(name="documents")

def log_stored_documents():
    try:
        docs = collection.get()
        print("✅ Stored Document IDs:", docs.get("ids", []))
        print("🧠 Metadata:", docs.get("metadatas", []))
    except Exception as e:
        print("⚠️ Failed to retrieve stored documents:", e)

def add_to_vectorstore(
    doc_id: str,
    embedding: List[float],
    metadata: Dict,
    document_text: str = ""
):
    try:
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document_text]
        )
        print(f"✅ Added to ChromaDB: {doc_id}")
        log_stored_documents()
    except Exception as e:
        print(f"❌ Failed to add {doc_id} to ChromaDB:", e)

def search_vectorstore(query_embedding: List[float], top_k: int = 5):
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        print("🔍 Query Results:", results)
        return results
    except Exception as e:
        print("❌ Vector search failed:", e)
        return {}
