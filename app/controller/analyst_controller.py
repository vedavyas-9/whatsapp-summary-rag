# app/controller/analyst_controller.py

from langsmith import traceable
from app.service.langstream_service import run_traced_claude_task

@traceable(name="Analyst Agent")
def run_analyst_agent(structured_json: str) -> str:
    """
    Stage 2 Analyst Agent
    Analyzes the dataset of the AP Police and extract the meaningful.
    """

    prompt = f"""
You are an Investigative Analyst AI working with AP Police chats and analyzing the chats and extracting the users, group details, member roles and which user assigned to whom.

You are given the following structured data:

{structured_json}


Give the Outout in correct format
"""

    return run_traced_claude_task(prompt, agent_name="Analyst Agent")
