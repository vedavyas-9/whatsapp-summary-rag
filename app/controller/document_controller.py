from typing import List
from langsmith import traceable
from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

@traceable(name="WhatsApp Document Agent")
def process_chat_logs(query: str, top_k: int = 5) -> str:
    """
    Processes WhatsApp chat logs (e.g., multiple files like Part1.txt, Part2.txt) to
    extract structured insights such as topic summaries, tasks, and early warnings.

    Parameters:
    - query (str): User query (e.g., "extract tasks from all chat logs").
    - top_k (int): Number of top results to retrieve from vector store.

    Returns:
    - str: Structured output with summaries, tasks, and alerts.
    """
    try:
        # Step 1: Generate embedding for the user query
        query_embedding = get_embedding(query)

        # Step 2: Search ChromaDB for relevant chat messages
        search_results = search_vectorstore(query_embedding, top_k=top_k)

        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            return "⚠️ No relevant chat messages found to process."

        # Step 3: Combine context + metadata (group, sender, role, timestamp, file)
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            metadata = metadatas[idx]
            tag = (
                f"(File: {metadata.get('file_name', 'Unknown')}, "
                f"Group: {metadata.get('group_name', 'Unknown')}, "
                f"Sender: {metadata.get('sender_name', 'Unknown')} - {metadata.get('sender_role', 'Unknown')}, "
                f"Timestamp: {metadata.get('timestamp', 'N/A')})"
            )
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        # Step 4: Compose Claude prompt
        prompt = f"""
You are an AI assistant processing WhatsApp chat logs from law enforcement groups.

You are given message excerpts from chat logs with metadata (file, group, sender name, role, timestamp). Your job is to:
1. Summarize conversations by topic (e.g., drug cases, jewelry heists, cyber scams).
2. Extract tasks and assignments (e.g., "assign teams," "update by 3 PM") with details like assignee, due date, and priority.
3. Identify early warnings (e.g., messages with "urgent," "alert," or emojis like 🚨).
4. Correlate entities (e.g., link senders to roles, map task assignments to hierarchy).
5. Answer the user’s query based ONLY on the context provided, providing structured insights.

Respond in **clear and concise bullet points** using these symbols:
- 📌 for topic-based summaries
- ✅ for extracted tasks (include assignee, task description, due date if available)
- ⚠️ for early warnings or urgent alerts
- 🔗 for correlations between entities (e.g., sender roles, task assignments)

---

📄 Chat Log Context:
{context}

❓ Query:
{query}

---

💬 Final Answer:
"""

        # Step 5: Get structured insights from Claude
        return run_traced_claude_task(prompt, agent_name="WhatsApp Document Agent")

    except Exception as e:
        return f"❌ Error during chat log processing: {type(e).__name__} - {e}"