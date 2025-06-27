from typing import List
from langsmith import traceable

from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

@traceable(name="Chat Agent")
def answer_query(query: str, top_k: int = 5) -> str:
    try:
        # Step 1: Generate Titan embedding for the user query
        query_embedding = get_embedding(query)

        # Step 2: Search ChromaDB for similar documents
        search_results = search_vectorstore(query_embedding, top_k=top_k)

        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            return "⚠️ No relevant documents found to answer the query."

        # Step 3: Combine context + tags from metadata
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            tag = f"(Document Type: {metadatas[idx].get('type', 'Unknown')}, Source: {metadatas[idx].get('s3_path', 'N/A')})"
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        # Step 4: Compose Claude prompt
        prompt = f"""
You are an AI assistant analyzing law enforcement datasets such as Call Detail Records (CDRs), IPDRs, FIRs, CAFs, and Bank Statements.

You are given document excerpts related to police investigations. Your job is to:
1. Extract relevant facts and entities (phone numbers, names, times, call durations, locations, etc.).
2. Link relationships between these entities (e.g., who called whom, mentioned in which FIR, appeared in which bank statement).
3. Detect unusual patterns or anomalies (e.g., frequent calls to a suspect, overlapping session times, suspicious call timings).
4. Answer the user’s query based ONLY on the context provided.

Respond in **clear and concise natural language**, and support your answer with specific evidence from the context where applicable.

---

📄 Document Context:
{context}

❓ Question:
{query}

---

💬 Final Answer:
"""

        # Step 5: Get answer from Claude with LangSmith trace
        return run_traced_claude_task(prompt, agent_name="Chat Agent")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"
