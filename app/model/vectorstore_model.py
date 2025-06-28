# === File 1: vectorstore_model.py ===

import os
from chromadb import PersistentClient
from typing import List, Dict

# Set absolute path for Chroma DB folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../chroma_db"))
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

print(f"📂 Chroma DB will be stored at: {CHROMA_DB_DIR}")

# Initialize persistent ChromaDB client
client = PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="documents")

def log_stored_documents():
    try:
        docs = collection.get()
        print("✅ Stored Document IDs:", docs.get("ids", []))
        cleaned_metadatas = []

        for meta in docs.get("metadatas", []):
            clean_meta = dict(meta)
            clean_meta.pop("s3_path", None)
            clean_meta.pop("path", None)
            if clean_meta.get("type") in ["CDR", "document"]:
                original_file = clean_meta.get("original_file", "")
                ext = os.path.splitext(original_file)[1].lstrip(".").lower()
                clean_meta["type"] = ext or "unknown"
            cleaned_metadatas.append(clean_meta)

        print("🧠 Metadata:", cleaned_metadatas)
    except Exception as e:
        print("⚠️ Failed to retrieve stored documents:", e)

def add_to_vectorstore(
    doc_id: str,
    embedding: List[float],
    metadata: Dict,
    document_text: str = ""
):
    try:
        metadata.pop("s3_path", None)
        metadata.pop("path", None)
        if metadata.get("type") in ["CDR", "document"]:
            original_file = metadata.get("original_file", "")
            ext = os.path.splitext(original_file)[1].lstrip(".").lower()
            metadata["type"] = ext or "unknown"

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
        # print("🔍 Query Results:", results)
        return results
    except Exception as e:
        print("❌ Vector search failed:", e)
        return {}
