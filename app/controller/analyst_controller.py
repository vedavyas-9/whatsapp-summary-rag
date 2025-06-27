from typing import List, Dict
from langsmith import traceable
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError
from service.langstream_service import run_traced_claude_task
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB client with explicit SSL settings
client = MongoClient(
    os.getenv("MONGODB_URI"),
    tls=True,
    tlsAllowInvalidCertificates=False,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    retryWrites=True,
    maxPoolSize=50
)
db = client["hackathon"]
messages_col = db["messages"]
results_col = db["results"]

@traceable(name="WhatsApp Analyst Agent")
def run_analyst_agent(file_id: str) -> str:
    """
    Analyzes WhatsApp chat data for summaries, tasks, and alerts.

    Parameters:
    - file_id (str): MongoDB file ID of the chat log.

    Returns:
    - str: Analysis results in markdown format.
    """
    try:
        # Fetch messages from MongoDB
        messages = messages_col.find({"file_id": file_id})
        message_list: List[Dict] = list(messages)

        if not message_list:
            return f"⚠️ No messages found for file_id: {file_id}"

        # Prepare context for Claude
        context_blocks = []
        for msg in message_list:
            tag = (
                f"(Sender: {msg.get('sender_name', 'Unknown')} - {msg.get('sender_role', 'Unknown')}, "
                f"Timestamp: {msg.get('timestamp', 'N/A')})"
            )
            context_blocks.append(f"{tag}\n{msg.get('message', '').strip()}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
You are an AI assistant analyzing WhatsApp chat data for a police department.

You are given message excerpts with metadata. Your job is to:
1. Summarize key topics discussed in the chat (e.g., drug cases, jewelry heist).
2. Identify tasks assigned, including assignees and due dates if mentioned.
3. Detect urgent alerts (e.g., phrases with "urgent," 🚨, or critical actions).
4. Provide a concise analysis in markdown format with sections for Summary, Tasks, and Alerts.

Use ONLY the provided context. If no relevant information is found, state so clearly.

Example Output:
```markdown
## Summary
- Drug case discussions focused on suspect "Raja" and interrogations.

## Tasks
- DSP Lovekik: Form a special team for jewelry heist (Due: Immediate).
- DSP Koushik: Ensure drug case interrogation yields info.

## Alerts
- 🚨 Urgent: Jewelry heist reported at Balaji Jewellers.
```

---

📄 Message Context:
{context}

---

💬 Analysis:
"""
        response = run_traced_claude_task(
            prompt,
            agent_name="WhatsApp Analyst Agent",
            task_type="analysis",
            file_id=file_id
        )

        # Store results in MongoDB
        if response:
            try:
                results_col.insert_one({
                    "file_id": file_id,
                    "task_type": "analysis",
                    "output": response,
                    "created_at": datetime.now().isoformat()
                })
            except (AutoReconnect, ServerSelectionTimeoutError) as e:
                return f"❌ MongoDB connection error during result storage: {type(e).__name__} - {e}"

        return response if response else "⚠️ No analysis generated."

    except (AutoReconnect, ServerSelectionTimeoutError) as e:
        return f"❌ MongoDB connection error: {type(e).__name__} - {e}"
    except Exception as e:
        return f"❌ Error in analyst agent: {type(e).__name__} - {e}"