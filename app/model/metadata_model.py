from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["hackathon"]
metadata_col = db["metadata"]
messages_col = db["messages"]

def save_file_metadata(
    file_id: str,
    doc_name: str,
    doc_type: str,
    group_id: str
) -> Dict[str, Any]:
    """
    Saves file metadata to MongoDB.

    Parameters:
    - file_id (str): MongoDB file ID.
    - doc_name (str): File name.
    - doc_type (str): Type of document ('chat_log' or 'metadata').
    - group_id (str): Group ID.

    Returns:
    - Dict[str, Any]: Metadata entry.
    """
    try:
        entry = {
            "file_id": file_id,
            "filename": doc_name,
            "doc_type": doc_type,
            "group_id": group_id,
            "created_at": datetime.now().isoformat()
        }
        metadata_col.insert_one(entry)
        return entry
    except Exception as e:
        print(f"❌ Error saving file metadata: {type(e).__name__} - {e}")
        return {}

def save_message_metadata(
    file_id: str,
    message_id: str,
    timestamp: str,
    phone_number: str,
    message: str,
    language: str,
    emojis: List[str],
    sender_name: str,
    sender_role: str,
    group_id: str
) -> Dict[str, Any]:
    """
    Saves message metadata to MongoDB.

    Parameters:
    - file_id (str): MongoDB file ID.
    - message_id (str): Unique message ID.
    - timestamp (str): Message timestamp.
    - phone_number (str): Sender phone number.
    - message (str): Message content.
    - language (str): Message language.
    - emojis (List[str]): Emojis in the message.
    - sender_name (str): Sender name.
    - sender_role (str): Sender role.
    - group_id (str): Group ID.

    Returns:
    - Dict[str, Any]: Message metadata entry.
    """
    try:
        entry = {
            "_id": message_id,
            "file_id": file_id,
            "timestamp": timestamp,
            "phone_number": phone_number,
            "message": message,
            "language": language,
            "emojis": emojis,
            "sender_name": sender_name,
            "sender_role": sender_role,
            "group_id": group_id,
            "created_at": datetime.now().isoformat()
        }
        messages_col.insert_one(entry)
        return entry
    except Exception as e:
        print(f"❌ Error saving message metadata: {type(e).__name__} - {e}")
        return {}