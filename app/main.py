import os
import sys
import json
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === App Imports (replace with real implementations) ===
from app.model.metadata_model import save_metadata
from app.model.text_extractor_model import extract_pdf_text, extract_word_text, extract_excel_text
from app.model.vectorstore_model import add_to_vectorstore
from app.model.graph_model import init_graph
from app.service.langstream_service import run_traced_claude_task
from app.controller.chat_controller import answer_query
from app.controller.task_controller import task_query
from app.controller.user_controller import user_query


# === Embedding: Titan Embedder ===
def get_embedding(text: str) -> list[float]:
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    print("TEXT TYPE:", type(text))         
    print("TEXT PREVIEW:", text[:100])      

    payload = {
        "inputText": text
    }

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        print("ERROR during Titan embedding:", e)
        raise





# === Initialize FastAPI app ===
app = FastAPI(title="AP Whatsapp", description="API for processing police documents and querying data")

# Dataset directory
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)


# === LLM Utils for vector and embedding ===
class LLMUtils:
    def __init__(self):
        self.vector_store = {}

    def get_embedding(self, text):
        return get_embedding(text)

    def add_to_vectorstore(self, doc_id, embedding, metadata, document_text):
        add_to_vectorstore(doc_id, embedding, metadata, document_text)
        self.vector_store[doc_id] = {
            "embedding": embedding,
            "metadata": metadata,
            "text": document_text
        }

llm_utils = LLMUtils()

def split_text_into_chunks(text, max_words=400):
    words = text.split()
    first_part = " ".join(words[:max_words])
    return first_part


# === Text Extraction Utilities ===
def extract_json_text(file_bytes):
    try:
        data = json.loads(file_bytes.decode('utf-8'))
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Error extracting JSON text: {str(e)}")
        return ""

def extract_txt_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""


# === API Endpoints ===
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    failed_files = []

    for file in files:
        file_path = os.path.join(DATASET_DIR, file.filename)
        try:
            if file.filename in llm_utils.vector_store:
                logger.info(f"File {file.filename} already processed, skipping")
                uploaded_files.append(file.filename)
                continue

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Uploaded file: {file_path}")

            file.file.seek(0)
            file_bytes = await file.read()

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

            chunks = split_text_into_chunks(text, max_tokens=8000)
            for i, chunk in enumerate(chunks):
                try:
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
                    logger.error(f"❌ Failed to process chunk {i} of {file.filename}: {str(e)}")
                    failed_files.append({"filename": f"{file.filename}_chunk_{i}", "error": str(e)})
                    continue

            uploaded_files.append(file.filename)
            logger.info(f"✅ Fully processed {file.filename} with {len(chunks)} chunks.")

        except Exception as e:
            logger.error(f"❌ Failed to process file {file.filename}: {str(e)}")
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
    try:
        response = answer_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

@app.post("/task")
async def query_task(query: str = Form(...)):
    try:
        response = task_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

@app.post("/users")
async def query_user(query: str = Form(...)):
    try:
        response = user_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

@app.post("/users/task")
async def query_user_task(query: str = Form(...)):
    try:
        response = user_query(query)
        return JSONResponse(content={"status": "success", "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
