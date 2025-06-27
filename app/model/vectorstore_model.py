import chromadb
from typing import List, Dict, Optional
from chromadb.config import Settings

# Initialize Chroma DB client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="whatsapp_chats")

def add_to_vectorstore(message_id: str, embedding: List[float], metadata: Dict, text: str) -> None:
    """
    Adds an embedding to Chroma DB with metadata and text.

    Parameters:
    - message_id (str): Unique ID for the message.
    - embedding (List[float]): Embedding vector.
    - metadata (Dict): Metadata (file_id, group_id, sender_name, sender_role, timestamp).
    - text (str): Original message text.
    """
    try:
        collection.add(
            ids=[message_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )
        print(f"✅ Added embedding for message ID: {message_id}")
    except Exception as e:
        print(f"❌ Error adding to Chroma DB: {type(e).__name__} - {e}")

def search_vectorstore(query_embedding: List[float], top_k: int = 5) -> Dict:
    """
    Searches Chroma DB for similar embeddings.

    Parameters:
    - query_embedding (List[float]): Embedding of the query.
    - top_k (int): Number of results to return.

    Returns:
    - Dict: Search results with documents, metadatas, and distances.
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return {
            "documents": results.get("documents", [[]]),
            "metadatas": results.get("metadatas", [[]]),
            "distances": results.get("distances", [[]])
        }
    except Exception as e:
        print(f"❌ Error searching Chroma DB: {type(e).__name__} - {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}