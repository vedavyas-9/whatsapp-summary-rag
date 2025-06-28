import os
import sys
import json
import re
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# App imports
from app.ingestion.uploader import upload_file_to_s3
from app.model.metadata_model import save_metadata
from app.model.text_extractor_model import extract_pdf_text, extract_word_text, extract_excel_text
from app.model.embedding_model import get_embedding
from app.model.vectorstore_model import add_to_vectorstore
from app.model.graph_model import init_graph
from app.service.langstream_service import run_traced_claude_task
from app.controller.chat_controller import answer_query
from app.controller.task_controller import task_query
from app.controller.user_controller import user_query

# Initialize FastAPI app
app = FastAPI(title="AP Police AI Platform", description="API for processing police documents and querying data")

# Define dataset directory
DATASET_DIR = "dataset"

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Mock llm_utils for vector store and embedding (replace with actual implementation)
class LLMUtils:
    def __init__(self):
        # Initialize vector store (replace with actual vector store initialization)
        self.vector_store = {}  # Placeholder for vector store
    def get_embedding(self, text):
        # Replace with actual embedding logic from get_embedding
        return get_embedding(text)
    def add_to_vectorstore(self, doc_id, embedding, metadata, document_text):
        # Replace with actual add_to_vectorstore logic
        add_to_vectorstore(doc_id, embedding, metadata, document_text)
        self.vector_store[doc_id] = {"embedding": embedding, "metadata": metadata, "text": document_text}

llm_utils = LLMUtils()

def extract_json_text(file_bytes):
    """Extract text from JSON file bytes."""
    try:
        data = json.loads(file_bytes.decode('utf-8'))
        return json.dumps(data, indent=2)  # Convert JSON to string
    except Exception as e:
        logger.error(f"Error extracting JSON text: {str(e)}")
        return ""

def extract_txt_text(file_path):
    """Extract text from text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def chunk_text(text, max_tokens=8192, overlap=100):
    """Split text into chunks within the token limit with overlap."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    token_estimate = 0  # Simplified token counting (adjust based on model)

    for word in words:
        # Estimate tokens (rough: 1 word ≈ 1.3 tokens, adjust for your model)
        token_estimate += len(word) / 4 + 1
        if token_estimate > max_tokens - overlap:
            chunks.append(" ".join(current_chunk))
            # Keep some overlap for context
            current_chunk = current_chunk[-int(overlap/1.3):]
            token_estimate = sum(len(w) / 4 + 1 for w in current_chunk)
        current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
):
    """Upload and process files, skipping already processed files."""
    uploaded_files = []
    failed_files = []
    
    for file in files:
        file_path = os.path.join(DATASET_DIR, file.filename)
        try:
            # Check if file is already in vector store
            if file.filename in llm_utils.vector_store:
                logger.info(f"File {file.filename} already processed, skipping")
                uploaded_files.append(file.filename)
                continue

            # Save file to disk
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Uploaded file: {file_path}")

            # Read file bytes for processing
            file.file.seek(0)
            file_bytes = await file.read()

            # Extract text based on file type
            text = ""
            if file.filename.endswith(".json"):
                text = extract_json_text(file_bytes)
            elif file.filename.endswith(".txt"):
                text = extract_txt_text(file_path)
            elif file.filename.endswith(".pdf"):
                text = extract_pdf_text(file_path)
            elif file.filename.endswith(".docx"):
                text = extract_word_text(file_path)
            elif file.filename.endswith(".xlsx"):
                text = extract_excel_text(file_path)
            else:
                logger.warning(f"Unsupported file type: {file.filename}")
                failed_files.append({"filename": file.filename, "error": "Unsupported file type"})
                continue

            if not text:
                logger.warning(f"No text extracted from {file.filename}")
                failed_files.append({"filename": file.filename, "error": "No text extracted"})
                continue

            # # Generate embedding and add to vector store
            # embedding = llm_utils.get_embedding(text)
            # llm_utils.add_to_vectorstore(
            #     doc_id=file.filename,
            #     embedding=embedding,
            #     metadata={"type": "document", "path": file_path},
            #     document_text=text
            # )
            # uploaded_files.append(file.filename)
            # logger.info(f"Processed and added {file.filename} to vector store")
            # Chunk text if necessary
            chunks = chunk_text(text, max_tokens=8192, overlap=100)
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding for each chunk
                    embedding = llm_utils.get_embedding(chunk)
                    doc_id = f"{file.filename}_chunk_{i}"
                    llm_utils.add_to_vectorstore(
                        doc_id=doc_id,
                        embedding=embedding,
                        metadata={
                            "type": "document",
                            "path": file_path,
                            "original_file": file.filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        },
                        document_text=chunk
                    )
                except Exception as e:
                    logger.error(f"Failed to process chunk {i} of {file.filename}: {str(e)}")
                    failed_files.append({"filename": f"{file.filename}_chunk_{i}", "error": str(e)})
                    continue

            uploaded_files.append(file.filename)
            logger.info(f"Processed and added {file.filename} to vector store")

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {str(e)}")
            failed_files.append({"filename": file.filename, "error": str(e)})
            continue

    if not uploaded_files:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No files processed successfully", "failed_files": failed_files}
        )

    response = {"status": "success", "uploaded_files": uploaded_files}
    if failed_files:
        response["failed_files"] = failed_files
    return JSONResponse(content=response)

@app.post("/query")
async def query_data(query: str = Form(...)):
    """
    Endpoint to query processed documents.
    """
    try:
        response = answer_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

@app.post("/task")
async def query_task(query: str = Form(...)):
    """
    Endpoint to query processed documents.
    """
    try:
        response = task_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")
    
@app.post("/users")
async def query_user(query: str = Form(...)):
    """
    Endpoint to query users.
    """
    try:
        response = user_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)