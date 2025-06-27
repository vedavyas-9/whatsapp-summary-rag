from typing import List
from langsmith import traceable
from model.vectorstore_model import search_vectorstore
from service.langstream_service import run_traced_claude_task
from model.embedding_model import get_embedding

@traceable(name="WhatsApp Chat Agent")
def answer_query(query: str, top_k: int = 5) -> str:
    """
    Answers queries on WhatsApp chat data using Chroma DB vector search.

    Parameters:
    - query (str): User query.
    - top_k (int): Number of results.

    Returns:
    - str: Answer to the query.
    """
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return "⚠️ Failed to generate query embedding."

        search_results = search_vectorstore(query_embedding, top_k=top_k)

        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            return "⚠️ No relevant messages found."

        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            metadata = metadatas[idx]
            tag = (
                f"(Group: {metadata.get('group_id', 'Unknown')}, "
                f"Sender: {metadata.get('sender_name', 'Unknown')} - {metadata.get('sender_role', 'Unknown')}, "
                f"Timestamp: {metadata.get('timestamp', 'N/A')})"
            )
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are an AI assistant analyzing WhatsApp chat data.

You are given message excerpts with metadata. Your job is to:
1. Extract facts and entities (e.g., sender names, timestamps, tasks).
2. Summarize conversations by topic if requested.
3. Identify tasks (e.g., "assign teams") with assignee and due date.
4. Detect urgent alerts (e.g., "urgent," 🚨).
5. Answer the query based ONLY on the context.

Respond in **clear and concise natural language**, supporting your answer with evidence.

---

📄 Message Context:
{context}

❓ Question:
{query}

---

💬 Final Answer:
"""
        return run_traced_claude_task(prompt, agent_name="WhatsApp Chat Agent", task_type="summarization")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"