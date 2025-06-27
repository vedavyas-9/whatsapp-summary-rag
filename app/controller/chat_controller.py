from typing import List
from langsmith import traceable
from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

@traceable(name="WhatsApp Chat Agent")
def answer_query(query: str, top_k: int = 5) -> str:
    """
    Answers queries on WhatsApp chat data by searching the vector store and generating
    responses based on message content, sender details, and metadata.

    Parameters:
    - query (str): User query (e.g., "summarize jewelry heist discussions").
    - top_k (int): Number of top results to retrieve from vector store.

    Returns:
    - str: Concise answer to the query with evidence from chat data.
    """
    try:
        # Step 1: Generate embedding for the user query
        query_embedding = get_embedding(query)

        # Step 2: Search ChromaDB for similar messages
        search_results = search_vectorstore(query_embedding, top_k=top_k)

        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            return "⚠️ No relevant messages found to answer the query."

        # Step 3: Combine context + metadata (group, sender, role, timestamp)
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            metadata = metadatas[idx]
            tag = (
    f"(Group: {metadata.get('group_name', 'Unknown')}, "
    f"Sender: {metadata.get('sender_name', 'Unknown')} - {metadata.get('sender_role', 'Unknown')}, "
    f"Timestamp: {metadata.get('timestamp', 'N/A')}, "
    f"File: {metadata.get('file_path', 'N/A')})"
)
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        # Step 4: Compose Claude prompt
        prompt = f"""
You are an AI assistant analyzing WhatsApp chat data from law enforcement groups.

You are given message excerpts with metadata (group, sender name, role, timestamp). Your job is to:
1. Extract relevant facts and entities (e.g., sender names, roles, timestamps, tasks, locations).
2. Summarize conversations by topic (e.g., drug cases, jewelry heists) if requested.
3. Identify tasks (e.g., "assign teams," "update by 3 PM") with assignee and due date.
4. Detect urgent alerts (e.g., messages with "urgent," "alert," or emojis like 🚨).
5. Answer the user’s query based ONLY on the context provided.

Respond in **clear and concise natural language**, supporting your answer with specific evidence from the context.

---

📄 Message Context:
{context}

❓ Question:
{query}

---

💬 Final Answer:
"""

        # Step 5: Get answer from Claude with LangSmith trace
        return run_traced_claude_task(prompt, agent_name="WhatsApp Chat Agent")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"