# app/view/streamlit_ui.py

import streamlit as st
from app.controller.document_controller import process_uploaded_file

st.title("🔍 AP Police AI Hackathon Platform")

uploaded_files = st.file_uploader(
    "📄 Upload Documents", type=["pdf", "docx", "xlsx"], accept_multiple_files=True
)

doc_type = st.selectbox("📁 Document Type", ["CDR", "IPDR", "FIR", "CAF", "BankStmt"])

if uploaded_files:
    for uploaded in uploaded_files:
        st.header(f"📂 Processing: {uploaded.name}")
        process_uploaded_file(uploaded, doc_type)
