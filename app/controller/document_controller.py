from typing import List
from langsmith import traceable
from model.text_extractor_model import WhatsAppChatExtractor
from model.embedding_model import get_embedding
from model.vectorstore_model import add_to_vectorstore
from model.metadata_model import save_file_metadata
from controller.analyst_controller import run_analyst_agent
from ingestion.uploader import save_file_to_mongo

@traceable(name="WhatsApp Document Agent")
def process_uploaded_file(file: object, file_type: str) -> str:
    """
    Processes uploaded WhatsApp chat or metadata file from MongoDB.

    Parameters:
    - file: File object (Streamlit uploaded file).
    - file_type (str): Type of file ('chat_log' for .txt, 'metadata' for .json).

    Returns:
    - str: Analysis results (summaries, tasks, alerts).
    """
    try:
        # Validate file type
        if file_type not in ["chat_log", "metadata"]:
            return f"❌ Invalid file type: {file_type}. Expected 'chat_log' or 'metadata'."

        # Upload to MongoDB
        metadata = save_file_to_mongo(file, "GRP_DCB_VZM", file_type)
        save_file_metadata(metadata["file_id"], metadata["file_name"], file_type, metadata["group_id"])

        if file_type == "chat_log":
            # Extract messages
            extractor = WhatsAppChatExtractor()
            messages = extractor.extract_messages(metadata["file_id"])
            
            # Generate and store embeddings
            for msg in messages:
                embedding = get_embedding(msg["message"])
                vector_metadata = {
                    "file_id": msg["file_id"],
                    "group_id": msg["group_id"],
                    "sender_name": msg["sender_name"],
                    "sender_role": msg["sender_role"],
                    "timestamp": msg["timestamp"],
                    "language": msg["language"]
                }
                add_to_vectorstore(msg["_id"], embedding, vector_metadata, msg["message"])
            
            # Run analysis
            analysis = run_analyst_agent(metadata["file_id"])
            return analysis
        else:
            return "✅ Metadata file processed and stored."

    except Exception as e:
        return f"❌ Error processing file: {type(e).__name__} - {e}"