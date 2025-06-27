import streamlit as st
from ingestion.uploader import save_file_to_mongo
from model.text_extractor_model import WhatsAppChatExtractor
from controller.analyst_controller import run_analyst_agent
from controller.chat_controller import answer_query

st.title("🔍 AP Police WhatsApp Chat Analyzer")

# File upload section
with st.expander("📄 Upload WhatsApp Chats or Metadata", expanded=True):
    uploaded_files = st.file_uploader(
        "📁 Upload WhatsApp chat (.txt) or metadata (.json)", 
        type=["txt", "json"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            st.header(f"📂 Processing: {uploaded.name}")
            file_type = "chat_log" if uploaded.name.endswith(".txt") else "metadata"
            with st.spinner("📤 Uploading to MongoDB..."):
                metadata = save_file_to_mongo(uploaded, "GRP_DCB_VZM", file_type)
                st.success(f"✅ File saved with ID: {metadata['file_id']}")

            if file_type == "chat_log":
                with st.spinner("📄 Extracting messages..."):
                    extractor = WhatsAppChatExtractor()
                    messages = extractor.extract_messages(metadata["file_id"])
                    st.success(f"✅ Extracted {len(messages)} messages")

                with st.spinner("🧠 Analyzing chat..."):
                    analysis = run_analyst_agent(metadata["file_id"])
                    st.subheader("📊 Analysis Results")
                    st.markdown(analysis)

# Query section
with st.expander("💬 Ask Questions about Chats", expanded=True):
    user_query = st.text_input("🔎 Enter your question (e.g., 'Summarize drug case discussions')")
    if user_query and st.button("Search"):
        with st.spinner("🧠 Processing query..."):
            response = answer_query(user_query)
            st.subheader("📝 Query Response")
            st.markdown(response)