import os
import sys
import json
import re
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
from app.controller.chat_controller import answer_query  # Chat controller

# Streamlit config
st.set_page_config(page_title="AP Police AI Platform", layout="wide")
st.title("🚓 AP Police AI Hackathon Platform")

# ============ Document Upload Section ============
with st.expander("📄 Upload and Process Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "📁 Upload PDF, Word or Excel files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True
    )

    doc_type = st.selectbox("📂 Document Type", ["CDR", "IPDR", "FIR", "CAF", "BankStmt"])

    if uploaded_files:
        for uploaded in uploaded_files:
            st.markdown(f"### 🔍 Processing: `{uploaded.name}`")
            file_bytes = uploaded.read()
            uploaded.seek(0)

            # Step 1: Upload to S3
            with st.spinner("📤 Uploading to S3..."):
                s3_path = upload_file_to_s3(uploaded)
                st.success(f"✅ Uploaded to S3: `{s3_path}`")

            # Step 2: Save metadata
            with st.spinner("🧠 Saving metadata..."):
                metadata = save_metadata(uploaded.name, doc_type, s3_path)
                st.success("✅ Metadata saved")

            # Step 3: Extract text
            with st.spinner("📄 Extracting text..."):
                if uploaded.name.endswith(".pdf"):
                    text = extract_pdf_text(file_bytes)
                elif uploaded.name.endswith(".docx"):
                    text = extract_word_text(file_bytes)
                elif uploaded.name.endswith(".xlsx"):
                    text = extract_excel_text(file_bytes)
                else:
                    text = ""
                st.success("✅ Text extracted")

            # Step 4: Generate embedding
            with st.spinner("🔗 Generating embedding with Amazon Titan..."):
                embedding = get_embedding(text)
                st.success("✅ Embedding generated")

            # Step 5: Save to ChromaDB
            with st.spinner("💾 Saving to ChromaDB..."):
                try:
                    add_to_vectorstore(
                        doc_id=uploaded.name,
                        embedding=embedding,
                        metadata={"type": doc_type, "s3_path": s3_path},
                        document_text=text
                    )
                    st.success("✅ Saved to ChromaDB")
                except Exception as e:
                    st.error(f"❌ Failed to store vector: {type(e).__name__} - {e}")

            # Step 6: Extract Graph Triples
            with st.spinner("🧠 Extracting relationships using Claude..."):
                graph_prompt = f"""
                You are a graph extraction agent for police documents.
                Extract structured relationships from the document below.

                Return only a valid JSON array of dictionaries with 'source', 'relation', and 'target'.

                Example:
                [
                  {{ "source": "SUB001", "relation": "CALLED", "target": "SUB002" }},
                  {{ "source": "SUB003", "relation": "CONNECTED_TO", "target": "TWR001" }}
                ]

                Document:
                {text[:3000]}
                """

                triples_raw = run_traced_claude_task(graph_prompt)
                try:
                    match = re.search(r"\[\s*{.*?}\s*\]", triples_raw, re.DOTALL)
                    if not match:
                        raise ValueError("No valid JSON array found in response.")

                    json_str = match.group()
                    triples_dicts = json.loads(json_str)

                    triples = [
                        [t["source"], t["relation"], t["target"]]
                        for t in triples_dicts
                        if all(k in t for k in ("source", "relation", "target"))
                    ]

                    st.subheader("🔗 Extracted Relationships")
                    st.json(triples_dicts)
                except Exception as e:
                    st.error(f"❌ Failed to parse graph triples: {type(e).__name__} - {e}")
                    st.text_area("Claude Response (debug)", triples_raw, height=200)
                    triples = []

            # Step 7: Update Neo4j
            with st.spinner("📈 Updating Neo4j Graph..."):
                try:
                    init_graph(triples)
                    st.success("✅ Graph database updated")
                except Exception as e:
                    st.error(f"❌ Neo4j update failed: {type(e).__name__} - {e}")

# ============ Chat with Data Section ============
with st.expander("💬 Ask Questions from Documents", expanded=True):
    user_query = st.text_input("🔎 Enter your question")

    if user_query:
        with st.spinner("🧠 Thinking..."):
            response = answer_query(user_query)
            st.success("✅ Answer")
            st.markdown(response)
