import logging
from typing import List, Optional
from langsmith import traceable
from app.model.vectorstore_model import search_vectorstore
from app.service.langstream_service import run_traced_claude_task
from app.model.embedding_model import get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@traceable(name="Summary Agent")
def summary_query(
    user_id: str,
    group_id: str,
    start_date: str,
    end_date: str,
    summary_rules: str ,
    top_k: int = 5
) -> str:
    try:
        logger.info("Inside summary controller for user_id: %s, group_id: %s, date range: %s to %s", 
                    user_id, group_id, start_date, end_date)
        
        # Step 1: Generate Titan embedding for the query
        query_embedding = get_embedding(summary_rules)
        # logger.info("Generated Titan embedding for query: %s", summary_rules)
        
        logger.info("Generated Titan embedding for query: %s", summary_rules)

        # Step 2: Search ChromaDB for relevant chat logs
        search_results = search_vectorstore(query_embedding, top_k=top_k)
        matched_docs: List[str] = search_results.get("documents", [[]])[0]
        metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        # # Step 2: Search ChromaDB for relevant chat logs
        # search_results = search_vectorstore(query_embedding, top_k=top_k)
        # matched_docs: List[str] = search_results.get("documents", [[]])[0]
        # metadatas: List[dict] = search_results.get("metadatas", [[]])[0]

        # if not matched_docs:
        #     logger.warning("No relevant documents found for group_id: %s", group_id)
            

        # # Step 3: Filter documents by group_id and date range
        context_blocks = []
        for idx, doc in enumerate(matched_docs):
            tag = f"(Document Type: {metadatas[idx].get('type', 'Unknown')}, Source: {metadatas[idx].get('s3_path', 'N/A')})"
            context_blocks.append(f"{tag}\n{doc.strip()}")

        context = "\n\n---\n\n".join(context_blocks)
        logger.info("Context sent to LLM:%s", context)

        if not context_blocks:
            logger.warning("No chat logs found for group_id: %s in date range: %s to %s", 
                          group_id, start_date, end_date)
        
        # Step 3: Filter documents by group_id, date range, and document type
        # context_blocks = []
        # for idx, doc in enumerate(matched_docs):
        #     metadata = metadatas[idx]
        #     doc_group_id = metadata.get('grp_id', 'N/A')
        #     doc_date = metadata.get('date', None)
        #     doc_type = metadata.get('type', 'Unknown')
        #     source = metadata.get('s3_path', metadata.get('path', 'N/A'))

        #     # Only include chat logs for the specified group_id and type 'chat_log'
        #     if doc_type != 'chat_log' or doc_group_id != group_id:
        #         logger.debug("Skipping document: type=%s, group_id=%s (expected: %s)", 
        #                     doc_type, doc_group_id, group_id)
        #         continue

        #     # Filter by date range if date is present
        #     if doc_date and (doc_date < start_date or doc_date > end_date):
        #         logger.debug("Skipping document: date=%s (outside range %s to %s)", 
        #                     doc_date, start_date, end_date)
        #         continue

        #     # Include documents mentioning the user_id
        #     if user_id not in doc:
        #         logger.debug("Skipping document: user_id=%s not found in content", user_id)
        #         continue

        #     tag = f"(Document Type: {doc_type}, Source: {source}, Group ID: {doc_group_id}, Date: {doc_date or 'N/A'})"
        #     context_blocks.append(f"{tag}\n{doc.strip()}")

        # context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant chat logs found."
        # logger.info("Context sent to LLM:\n%s", context)
           

        # Step 4: Compose Claude prompt
        prompt = f"""
You are an AI assistant for the AP Police AI Platform, designed to summarize chat logs from police-related WhatsApp groups stored in a vector store. The documents include:

1. **Chat logs** (e.g., `Part1.txt`, `Part2.txt`): Contain messages with metadata including group ID, user ID, date, and content for.
2. **members_info.json**: User details with fields: user_id (e.g:SP_DistrictHQ_001), name, ole, jusrisdiction_type, jusrisdiction_name,phone_number
3. **group_info.json**: Group details with fields: grp_id, gname, purpose.
4. **hierarchy.json**: Reporting structure with fields: mem_id, Reports_to.
5. **ranks.json**: Ranks and levels with fields: Rank, level.

**Task**:
Generate a summary of chat logs for a specific group ({group_id}) and user ({user_id}) within the date range {start_date} to {end_date}. The summary should:
- Be organized by day within the date range.
- List important tasks, operations, cases, and major incidents in bullet points.
- Only include information grounded in the provided chat logs; do not hallucinate or invent details.
- Include group and user metadata from `group_info.json` and `members_info.json`.
- If no relevant data is found, return an empty summary with a note.

**Response Format**:
- Return a JSON object with:
  - `summary`: List of daily summaries that is the summary of all the chats on that sepcified day, each with `date` and `points` (bullet points as strings).
  - `group_details`: Group metadata (grp_id, group_name, group_purpose).
  - `user_details`: User metadata (user_id, name, role, reports_to_id, jurisdiction_type, jurisdiction_name, phone_number, rank_level, reports_to_name).
  - `note`: Explanation if no data is found (e.g., "No chat logs found for group {group_id}").
- Enclose the response in triple backticks (```json```).
- Ensure the response is factual, based only on the provided context.

**Current Context**:
# - Documents are stored in a vector store, processed via a FastAPI `/users` endpoint.
- Documents include chat files for (`Part1.txt`,`Part2.txt`) for gname="Vizianagaram District Coordination"), (`Part1 1.txt`,`Part2 1.txt`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`Part1 2.txt`) for gname="District Crime Branch - Vizianagaram".
- Documents include chat files for  (`hierarchy.json`) gname="Vizianagaram District Coordination", (`hierarchy 1.json`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`hierarchy 2.json`) for gname="District Crime Branch - Vizianagaram".
- Documents include chat files for  (`memebers_info.json`) gname="Vizianagaram District Coordination", (`memebers_info 1.json`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`memebers_info 2.json`) for gname="District Crime Branch - Vizianagaram".
- Documents include chat files for  (`ranks.json`) gname="Vizianagaram District Coordination", (`ranks 1.json`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`ranks 2.json`) for gname="District Crime Branch - Vizianagaram".
- Documents include chat files for  (`summary_rules.json`) gname="Vizianagaram District Coordination", (`summary_rules 1.json`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`summary_rules 2.json.json`) for gname="District Crime Branch - Vizianagaram".
- Documents include chat files for  (`group_info.json`) gname="Vizianagaram District Coordination", (`group_info 1.json`) for gname="Bandobast & Resource Allocation - North Sub-Division",(`group_info 2.json`) for gname="District Crime Branch - Vizianagaram".
- Context includes vector store search results with chunks of relevant documents.

---

📄 Document Context:
{context}

❓ Question:
{summary_rules}

---
"""

   
        # Step 5: Get answer from Claude with LangSmith trace
        return run_traced_claude_task(prompt, agent_name="Summary Agent")

    except Exception as e:
        return f"❌ Error during query processing: {type(e).__name__} - {e}"