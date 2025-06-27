from langsmith import traceable
from app.service.langstream_service import run_traced_claude_task

@traceable(name="WhatsApp Analyst Agent")
def run_analyst_agent(structured_json: str) -> str:
    """
    Stage 2 WhatsApp Analyst Agent
    Analyzes structured JSON from WhatsApp chat data to generate topic-based summaries,
    extract tasks, and flag early warnings for law enforcement investigations.

    Parameters:
    - structured_json (str): Pre-parsed JSON containing chat messages, sender details, and metadata.

    Returns:
    - str: Bullet-point summary of insights, tasks, and alerts for further processing or UI display.
    """

    prompt = f"""
You are an Investigative Analyst AI working with WhatsApp chat data from law enforcement groups.

You are given the following structured data:
{structured_json}

Your responsibilities:
1. Summarize conversations by topic (e.g., drug cases, jewelry heists, cyber scams) based on group purpose and message content.
2. Extract tasks and assignments (e.g., "assign teams," "update by 3 PM") with details like assignee, due date, and priority.
3. Identify early warnings (e.g., messages with urgent keywords like "urgent," "alert," or emojis like 🚨).
4. Correlate entities (e.g., link senders to roles from metadata, map task assignments to hierarchy).

---

🎯 Output Format:
Respond using clear, concise bullet points using these symbols:
- 📌 for topic-based summaries
- ✅ for extracted tasks (include assignee, task description, due date if available)
- ⚠️ for early warnings or urgent alerts
- 🔗 for correlations between entities (e.g., sender roles, task assignments)

Avoid repeating input. Focus on actionable insights.
"""

    return run_traced_claude_task(prompt, agent_name="WhatsApp Analyst Agent")