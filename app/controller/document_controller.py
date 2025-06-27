from app.model.file_model import upload_file_to_s3
from app.model.metadata_model import save_metadata
from app.model.text_extractor_model import extract_text
from app.model.embedding_model import get_embedding
from app.model.vectorstore_model import add_to_vectorstore
from app.model.llm_model import run_claude_task
from app.model.graph_model import init_graph
from app.service.langstream_service import build_langstream_pipeline

def process_document(uploaded, doc_type):
    file_bytes = uploaded.read()
    uploaded.seek(0)
    s3_path = upload_file_to_s3(uploaded)
    save_metadata(uploaded.name, doc_type, s3_path)
    text = extract_text(file_bytes, uploaded.name)
    embedding = get_embedding(text)
    add_to_vectorstore(doc_id=uploaded.name, embedding=embedding, metadata={"type": doc_type})
    summary = run_claude_task(text)
    triples = build_langstream_pipeline(text)
    init_graph(triples)
    return s3_path, summary, triples
