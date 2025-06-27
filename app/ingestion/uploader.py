import os
from uuid import uuid4
from typing import Dict
from pymongo import MongoClient
from base64 import b64encode

from dotenv import load_dotenv
import os
load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["hackathon"]
files_col = db["files"]

def save_file_to_mongo(
    file: object,
    group_id: str,
    file_type: str = "chat_log"
) -> Dict[str, str]:
    """
    Saves a file to MongoDB Atlas and generates metadata for WhatsApp chat analysis.

    Parameters:
    - file: File object (e.g., from Streamlit file uploader or local file).
    - group_id (str): Group ID from group_info.json (e.g., 'GRP_DCB_VZM').
    - file_type (str): Type of file ('chat_log' for .txt files, 'metadata' for .json files).

    Returns:
    - Dict[str, str]: Metadata including file_id, file_name, group_id, and file_type.
    """
    try:
        # Validate file type
        valid_extensions = {
            "chat_log": [".txt"],
            "metadata": [".json"]
        }
        file_name = getattr(file, "name", file)
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_type not in valid_extensions or file_ext not in valid_extensions[file_type]:
            raise ValueError(f"Invalid file type: {file_ext}. Expected {valid_extensions[file_type]} for {file_type}.")

        # Read file content
        if hasattr(file, "read"):
            content = file.read()  # Streamlit file object
        else:
            with open(file, "rb") as f:
                content = f.read()  # Local file path

        # Encode content as base64
        encoded_content = b64encode(content).decode("utf-8")

        # Generate unique file ID
        file_id = str(uuid4())

        # Store file in MongoDB
        file_doc = {
            "_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "group_id": group_id,
            "content": encoded_content
        }
        files_col.insert_one(file_doc)

        # Return metadata
        return {
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type,
            "group_id": group_id
        }

    except Exception as e:
        raise Exception(f"Error saving file to MongoDB: {type(e).__name__} - {e}")

def save_local_file(
    file_path: str,
    group_id: str,
    file_type: str = "chat_log"
) -> Dict[str, str]:
    """
    Saves a local file to MongoDB (e.g., for fixed dataset).

    Parameters:
    - file_path (str): Path to the local file (e.g., 'Part1.txt').
    - group_id (str): Group ID from group_info.json.
    - file_type (str): Type of file ('chat_log' or 'metadata').

    Returns:
    - Dict[str, str]: Metadata including file_id, file_name, group_id, and file_type.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return save_file_to_mongo(file_path, group_id, file_type)
    except Exception as e:
        raise Exception(f"Error processing local file {file_path}: {e}")