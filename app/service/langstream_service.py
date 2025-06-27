from langsmith import traceable
from model.llm_model import run_claude_task
from typing import Optional

@traceable(name="Claude Task", run_type="llm")
def run_traced_claude_task(
    prompt: str,
    agent_name: str = "Generic Agent",
    task_type: str = "generic",
    file_id: Optional[str] = None,
    max_tokens: int = 1000
) -> str:
    """
    Runs a Claude task with tracing, handling LLM interactions.

    Parameters:
    - prompt (str): Input prompt for Claude.
    - agent_name (str): Name of the agent.
    - task_type (str): Type of task (e.g., summarization, graph_extraction).
    - file_id (Optional[str]): Associated file ID.
    - max_tokens (int): Maximum tokens for Claude response.

    Returns:
    - str: Claude's response.
    """
    try:
        metadata = {
            "agent_name": agent_name,
            "task_type": task_type,
            "file_id": file_id
        }
        response = run_claude_task(prompt, max_tokens=max_tokens, metadata=metadata)
        return response
    except Exception as e:
        print(f"❌ Error in Claude task: {type(e).__name__} - {e}")
        return f"Error: {str(e)}"