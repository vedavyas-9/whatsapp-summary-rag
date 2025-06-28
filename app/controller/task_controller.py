from typing import List
from langsmith import traceable

from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

@traceable(name="Task Agent")
def task_query(query: str, top_k: int = 5) -> str:
    try:
        # Step 1: Generate Titan embedding for the user query
        query_embedding = get_embedding(query)

        # Step 2: Search ChromaDB for similar documents
        search_results = search_vectorstore(query_embedding, top_k=top_k)

        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            return "⚠️ No relevant results found to answer the query."

        # Step 3: Combine context + tags from metadata
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            tag = f"(Document Type: {metadatas[idx].get('type', 'Unknown')}, Source: {metadatas[idx].get('s3_path', 'N/A')})"
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        # Step 4: Compose Claude prompt
        prompt = f"""
You are an AI assistant analyzing AP Police whatsapp chat datasets.
The chats contains task that a senior officer assigns to junior officer. Extract those tasks from the chats(e.g: Situation reports from all circles must be submitted by 11:00 AM,evacuation drill to start in VZM, Confirm PC deployment near Zone-2).
When extracting task extarct which user assigned the task to whom.
---

📄 Document Context:
{context}

❓ Question:
{query}

---

💬 Final Answer:
"""

        # Step 5: Get answer from Claude with LangSmith trace
        return run_traced_claude_task(prompt, agent_name="Task Agent")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"
