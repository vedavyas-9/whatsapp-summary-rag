import os
from uuid import uuid4
from typing import Dict, Optional

def save_file_locally(
    file: object,
    group_id: str,
    file_type: str = "chat_log",
    folder: str = "uploads"
) -> Dict[str, str]:
    """
    Saves a file locally and generates metadata for WhatsApp chat analysis.

    Parameters:
    - file: File object (e.g., from Streamlit file uploader or local file).
    - group_id (str): Group ID from group_info.json (e.g., 'GRP_DCB_VZM').
    - file_type (str): Type of file ('chat_log' for .txt files, 'metadata' for .json files).
    - folder (str): Local folder path (default: 'uploads').

    Returns:
    - Dict[str, str]: Dictionary containing local file path and metadata (e.g., file_type, group_id).
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

        # Create directory structure: ./uploads/<group_id>/<file_type>/
        save_dir = os.path.join(folder, group_id, file_type)
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique file name
        unique_file_name = f"{uuid4()}_{file_name}"
        file_path = os.path.join(save_dir, unique_file_name)

        # Save file locally
        with open(file_path, "wb") as f:
            if hasattr(file, "read"):
                f.write(file.read())  # For file objects (e.g., Streamlit uploads)
            else:
                with open(file, "rb") as src_file:
                    f.write(src_file.read())  # For local file paths

        # Return metadata for storage in metadata_model.py
        return {
            "file_path": file_path,
            "file_name": file_name,
            "file_type": file_type,
            "group_id": group_id
        }

    except Exception as e:
        raise Exception(f"Error saving file locally: {type(e).__name__} - {e}")

def save_local_file(
    file_path: str,
    group_id: str,
    file_type: str = "chat_log",
    folder: str = "Uploads"
) -> Dict[str, str]:
    """
    Saves a local file to the specified directory (e.g., for fixed dataset or testing).

    Parameters:
    - file_path (str): Path to the local file (e.g., 'Part1.txt').
    - group_id (str): Group ID from group_info.json.
    - file_type (str): Type of file ('chat_log' or 'metadata').
    - folder (str): Local folder path.

    Returns:
    - Dict[str, str]: Dictionary containing local file path and metadata.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return save_file_locally(file_path, group_id, file_type, folder)
    except Exception as e:
        raise Exception(f"Error processing local file {file_path}: {e}")