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
    """
    Attempt to extract a valid JSON array from a possibly incomplete Claude response.
    Truncates broken content and ensures closing of the array.
    """
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
def task_query(query: str, top_k: int = 20) -> str:
    """
    Queries the ChromaDB vector store with a user query, processes the results using Claude,
    and extracts tasks from AP Police WhatsApp chat datasets in a structured JSON format.
    """
    try:
        logger.debug(f"Processing query: {query}")

        # Step 1: Generate Titan embedding for the user query
        query_embedding = get_embedding(query)

        # Step 2: Search ChromaDB for similar documents
        search_results = search_vectorstore(query_embedding, top_k=top_k)
        print("theese are the results",search_results)
        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        if not matched_docs:
            logger.warning("No matched documents found in ChromaDB.")
            return json.dumps({"status": "error", "message": "⚠️ No relevant results found to answer the query."})

        logger.debug(f"Matched documents: {len(matched_docs)} found")
        for idx, doc in enumerate(matched_docs):
            logger.debug(f"Document {idx}: {doc[:100]}... (Type: {metadatas[idx].get('type', 'Unknown')})")

        # Step 3: Combine context from metadata + document
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            tag = f"(Document Type: {metadatas[idx].get('type', 'Unknown')}, Source: {metadatas[idx].get('s3_path', 'N/A')})"
            context_blocks.append(f"{tag}\n{doc.strip()}")
        context = "\n\n---\n\n".join(context_blocks)

        context=search_results
        # Trim context to avoid hitting model limits (e.g., 10K characters)
        # if len(context) > 10000:
        #     logger.warning("Trimming long context...")
        #     context = context[:10000]

        logger.debug(f"Combined context length: {len(context)} characters")

        # Step 4: Claude prompt  for the group2################
        prompt = f"""
Strict JSON Output Required.

You are an AI assistant analyzing AP Police WhatsApp chat datasets, along with related personnel, group, hierarchy, and rank data. Your goal is to extract *all* tasks,explore all topics in the group which have more conversation and assigned by senior officers to junior officers within the group from the chat data from starting--date to end chat date and return the results as a valid JSON array. Use the provided document context to map phone numbers to personnel details, roles, jurisdictions, and reporting structures. Extract the following fields for each task:

- user: The phone number of the person assigned the task, as listed in personnel data.
- name: The full name of the person assigned the task, retrieved from personnel data.
- task_name: A concise description of the task get into the words assigned,.
- assigned_by: The full name of the senior officer assigning the task, retrieved from personnel data.
- priority: Infer the priority as "High", "Medium", or "Low" based on explicit mentions (e.g., "priority", "crucial", "immediately") or context (e.g., tight deadlines or critical events). Default to "Medium" if unclear.
- deadline: The explicit or inferred deadline for the task (e.g., "by EOD", "by May 15"). Use "N/A" if no deadline is specified.
- status: The task status based on chat updates (e.g., "Completed", "In Progress", "Not Started"). Default to "In Progress" unless completion or non-started status is explicitly mentioned.
- group_id: Set to "group_id" for all tasks, as they are related to this group.Get the group id from the group_info.json.
- date: The date of the task assignment in "YYYY-MM-DD" format (e.g., "2025-04-26").
- timestamp: The timestamp of the task assignment in "HH:MM" 24-hour format (e.g., "10:20").
- jurisdiction_name: The jurisdiction name of the person assigned the task, retrieved from personnel data (e.g., "VZM Rural Circle").

Instructions:
1. Use the personnel data (from members_info.json) to map phone numbers to names, roles, and jurisdictions.
2. Cross-reference hierarchy and rank data to confirm the senior-junior relationship for task assignments (e.g., SP > Addl. SP > DSP > CI > SI).
3. Analyze the chat context (from Part1.txt and Part2.txt dated April-May 2025) to identify all tasks related to the  group_id, including those assigned to individuals or group-wide (e.g., "@All CIs", "@CI").
4. For group-wide assignments (e.g., "@All CIs", "@CI"), identify all Circle Inspectors (role: "Circle Inspector") from personnel data and create a separate JSON object for each CI, ensuring all relevant CIs are included.
5. If a task is implied but not explicitly stated, infer it from the context (e.g., a response confirming action implies a task).
6. Ensure the output is a valid JSON array of task objects, with each object containing all required fields.
7. For Identification of tasks you need to go through this words which have the meaning of assiging task If no tasks are found, return an empty JSON array [].
8. Return ONLY a valid JSON array as the response, with no additional text, comments, or explanations outside the JSON structure.
9. Do not include tasks for users not present in the personnel data or from dates outside.

Document Context:
{context}

Question:
{query}

Final Answer:
Return the extracted tasks as a valid JSON array, e.g., [] or [{{"user": "...", ...}}].
"""        # Step 5: Claude generation
        claude_response = run_traced_claude_task(prompt, agent_name="Task Agent")
        logger.debug(f"Claude raw response: {claude_response[:1000]}...")

        # Step 6: Fix and parse JSON
        fixed_response = try_fix_json_response(claude_response)

        try:
            parsed_response = json.loads(fixed_response)
            if not isinstance(parsed_response, list):
                logger.error("Claude response is not a valid JSON array.")
                return json.dumps({
                    "status": "error",
                    "message": f"⚠️ Claude response is not a JSON array: {fixed_response[:100]}..."
                })

            logger.debug(f"Extracted {len(parsed_response)} tasks")
            return json.dumps({
                "status": "success",
                "response": parsed_response
            })
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}. Raw response: {fixed_response[:100]}...")
            return json.dumps({
                "status": "error",
                "message": f"⚠️ Invalid JSON response from Claude: {str(e)}. Raw response: {fixed_response[:100]}..."
            })

    except Exception as e:
        logger.exception("Exception during task query.")
        return json.dumps({
            "status": "error",
            "message": f"❌ Error during query processing: {type(e).__name__} - {str(e)}"
        })
