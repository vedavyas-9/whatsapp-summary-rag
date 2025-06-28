from typing import List
from langsmith import traceable
import json
import logging

from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def try_fix_json_response(response: str) -> str:
    """Attempt to extract a valid JSON array from a possibly incomplete Claude response."""
    try:
        start = response.index("[")
        last_brace = response.rfind("}")
        if last_brace == -1 or last_brace < start:
            return "[]"
        trimmed = response[start:last_brace + 1]
        if not trimmed.endswith("]"):
            trimmed += "]"
        return trimmed
    except ValueError:
        return "[]"

@traceable(name="Task Agent")
def group_task(group_id: str, top_k: int = 20) -> str:
    """
    Extract all tasks from the given group using vector search and Claude.
    """
    try:
        query = f"Give me all tasks from group {group_id}"
        logger.debug(f"Processing query: {query}")

        # Step 1: Generate Titan embedding
        query_embedding = get_embedding(query)

        # Step 2: Vector search in ChromaDB
        search_results = search_vectorstore(query_embedding, top_k=top_k)
        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            logger.warning("No matched documents found.")
            return json.dumps({"status": "error", "message": "⚠️ No relevant documents found."})

        # Step 3: Combine context
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            tag = f"(Document Type: {metadatas[idx].get('type', 'Unknown')}, Source: {metadatas[idx].get('s3_path', 'N/A')})"
            context_blocks.append(f"{tag}\n{doc.strip()}")
        context = "\n\n---\n\n".join(context_blocks)

        logger.debug(f"Combined context length: {len(context)} characters")

        # Step 4: Prompt to Claude with actual group_id injected
        prompt = f"""
Strict JSON Output Required.

You are an AI assistant analyzing AP Police WhatsApp chat datasets. Extract *all* tasks assigned by senior officers to junior officers in group **{group_id}**. Output must be a valid JSON array with the following fields:

- user, name, task_name, assigned_by, priority, deadline, status, group_id ("{group_id}"), date, timestamp, jurisdiction_name

Instructions:
1. Use `members_info.json` to resolve personnel info.
2. Verify ranks from `hierarchy.json`.
3. Analyze chats from Part1.txt and Part2.txt to detect all task-related content.
4. When messages like "@All CIs" or "@SI" appear, generate tasks for all matching roles.
5. Infer implied tasks where needed.
6. Ensure each output object contains all required fields.
7. Do not include any markdown or explanation. Return ONLY a valid JSON array. If no task, return [].

Document Context:
{context}

Question:
Give me all tasks from group {group_id}

Final Answer:
Return only a valid JSON array.
"""

        # Step 5: Claude generation
        claude_response = run_traced_claude_task(prompt, agent_name="Task Agent")
        logger.debug(f"Claude raw response: {claude_response[:1000]}...")

        # Step 6: Fix JSON
        fixed_response = try_fix_json_response(claude_response)
        try:
            parsed = json.loads(fixed_response)
            if not isinstance(parsed, list):
                return json.dumps({"status": "error", "message": "Claude output is not a valid JSON array."})

            return json.dumps({"status": "success", "response": parsed})
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "message": f"⚠️ Invalid JSON from Claude: {str(e)}"})

    except Exception as e:
        logger.exception("Exception during group task query.")
        return json.dumps({"status": "error", "message": f"❌ Error: {type(e).__name__} - {str(e)}"})
