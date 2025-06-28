from typing import List
from langsmith import traceable
import logging

from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@traceable(name="User Agent")
def user_query(query: str, top_k: int = 5) -> str:
    try:
        logger.info("Inside the user agent")
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
        logger.info("Context sent to LLM:\n%s", context)

        # Step 4: Compose Claude prompt
        prompt = f'''
You are an AI assistant for the AP Police AI Platform, designed to process and query police-related documents stored in a vector store. The documents include:

1. **members_info.json**: Contains user details with fields: mem_id, grp_id, Sub_Division, Circle, Rank, Officer_Name, Mobile_no.
2. **group_info.json**: Contains group details with fields: grp_id, gname, purpose.
3. **hierarchy.json**: Defines the reporting structure with fields: mem_id, Reports_to.
4. **ranks.json**: Lists ranks and their levels with fields: Rank, level.

Your task is to respond to queries about users and groups based on the provided documents. For each query, follow these instructions:

1. **Retrieve users from all groups**:
   - Extract all users from `members_info.json`, including their mem_id, Officer_Name, Rank, Sub_Division, Circle, Mobile_no, and grp_id.
   - For each user, include their group details (grp_id, gname, purpose) from `group_info.json`.
   - Include their rank level from `ranks.json` and their reporting officer (if any) from `hierarchy.json`, referencing the Officer_Name of the Reports_to mem_id.
   - Map fields to: user_id (mem_id), name (Officer_Name), role (Rank), reports_to_id (Reports_to), jurisdiction_type (Sub_Division), jurisdiction_name (Circle), phone_number (Mobile_no).

2. **Return users from specific groups with group name and metadata**:
   - For each group in `group_info.json`, list its users from `members_info.json` along with the group’s gname and purpose.
   - Include user details: user_id, name, role, reports_to_id, jurisdiction_type, jurisdiction_name, phone_number, rank_level, reports_to_name.

3. **Return users from a specific group based on the query**:
   - If the query specifies a group ID (e.g., "GRP001"), return only the users from that group with the above details, the group’s gname, purpose, and each user’s reporting officer’s name.
   - If no group ID is specified, assume the query applies to all groups.
   - If the group ID exists in `group_info.json` but has no users in `members_info.json`, return an empty user list with a note: "No users found for [group_id] in members_info.json".
   - If the group ID does not exist in `group_info.json`, return an empty user list with a note: "Group [group_id] not found in group_info.json".

**Response Format**:
- Return a JSON object with three sections:
  - `all_users`: List all users across all groups with their details (user_id, name, role, reports_to_id, jurisdiction_type, jurisdiction_name, phone_number, grp_id, group_name, group_purpose, rank_level, reports_to_name).
  - `groups_with_users`: Group users by group ID, including group_name, group_purpose, and user details (as above).
  - `specific_group_users`: Users from the queried group ID (if provided), with group_name, group_purpose, user details (as above), and a note if no users or group is found.
- Ensure the response is a valid JSON string, enclosed in triple backticks (```json```).
- If no users or groups match, return an empty list for the relevant section with a note explaining the result.

**Query Handling**:
- Parse the query to identify if a specific group ID is mentioned (e.g., "GRP001").
- If the query is ambiguous or does not specify a group, provide results for all groups in `specific_group_users` with a note: "No specific group ID provided; returning users from all groups".
- Cross-reference `members_info.json`, `group_info.json`, `hierarchy.json`, and `ranks.json` using the provided context.

**Example Query**:
- "Get users from GRP001"
  - Return users from GRP001 in `specific_group_users`, plus all users and grouped users in other sections.
- "Get users from GRP_BANDOBST_NORTH"
  - If no users, return empty list in `specific_group_users` with group metadata and note: "No users found for GRP_BANDOBST_NORTH in members_info.json".

**Current Context**:
- Documents are stored in a vector store, processed via a FastAPI `/users` endpoint.
- Documents include `members_info.json`, `group_info.json`, `hierarchy.json`, `ranks.json`, and chat logs (`Part1.txt`, `Part2.txt`), but focus on user/group data unless chat logs are requested.
- Context includes vector store search results with chunks of relevant documents.
- Current date and time: 11:45 AM IST, Saturday, June 28, 2025.

**Document Context**:
{context}

**Query**:
{query}

**Final Answer**:
Return the response as a JSON string in the format:
'''

        
        # Step 5: Get answer from Claude with LangSmith trace
        print("\n\n response from llm:", run_traced_claude_task(prompt, agent_name="User Agent"))
        return run_traced_claude_task(prompt, agent_name="User Agent")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"
