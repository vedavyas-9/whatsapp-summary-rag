import os
import sys
import json
import re
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ServerSelectionTimeoutError
from ingestion.uploader import save_file_to_mongo, save_local_file
from model.text_extractor_model import WhatsAppChatExtractor
from model.embedding_model import get_embedding
from model.vectorstore_model import add_to_vectorstore
from model.metadata_model import save_file_metadata, save_message_metadata
from service.langstream_service import run_traced_claude_task
from controller.chat_controller import answer_query
from controller.analyst_controller import run_analyst_agent

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# MongoDB client with explicit SSL settings
client = MongoClient(
    os.getenv("MONGODB_URI"),
    tls=True,
    tlsAllowInvalidCertificates=False,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    retryWrites=True,
    maxPoolSize=50
)
db = client["hackathon"]
graphs_col = db["graphs"]

# Streamlit config
st.set_page_config(page_title="AP Police WhatsApp AI Platform", layout="wide")
st.title("🚓 AP Police WhatsApp AI Hackathon Platform")

# Initialize fixed dataset
def initialize_dataset():
    files = [
        ("Part1.txt", "chat_log"),
        ("members_info.json", "metadata"),
        ("hierarchy.json", "metadata"),
        ("ranks.json", "metadata"),
        ("group_info.json", "metadata")
    ]
    extractor = WhatsAppChatExtractor()
    for file_path, file_type in files:
        try:
            file_metadata = save_local_file(file_path, "GRP_DCB_VZM", file_type)
            save_file_metadata(file_metadata["file_id"], file_metadata["file_name"], file_type, file_metadata["group_id"])
            if file_type == "chat_log":
                messages = extractor.extract_messages(file_metadata["file_id"])
                if not messages:
                    st.warning(f"⚠️ No messages extracted from {file_path}")
                    continue
                for msg in messages:
                    embedding = get_embedding(msg["message"])
                    if not embedding:
                        st.warning(f"⚠️ Failed to generate embedding for message ID: {msg['_id']}")
                        continue
                    vector_metadata = {
                        "file_id": msg["file_id"],
                        "group_id": msg["group_id"],
                        "sender_name": msg["sender_name"],
                        "sender_role": msg["sender_role"],
                        "timestamp": msg["timestamp"]
                    }
                    add_to_vectorstore(msg["_id"], embedding, vector_metadata, msg["message"])
                st.success(f"✅ Embeddings generated and stored in Chroma DB for {file_path}")
                
                # Extract graph triples
                with st.spinner("🧠 Extracting relationships using Claude..."):
                    text = "\n".join([f"{msg['sender_name']} ({msg['sender_role']}): {msg['message']}" for msg in messages])
                    if not text.strip():
                        st.warning(f"⚠️ No text available for graph extraction in {file_path}")
                        continue
                    graph_prompt = f"""
You are a graph extraction agent for WhatsApp chats. Extract structured relationships from the chat below, focusing on interactions between senders, tasks, and entities (e.g., suspects, locations). Return ONLY a valid JSON array of dictionaries with 'source', 'relation', and 'target'. Use sender names or roles from metadata, and identify relations like "ASSIGNED_TASK", "MENTIONED", "INVESTIGATING", or "ARRESTED".

Example:
[
  {{"source": "DSP Lovekik", "relation": "ASSIGNED_TASK", "target": "CI Yeshwanth"}},
  {{"source": "DSP Koushik", "relation": "INVESTIGATING", "target": "Raja"}}
]

Return an empty array [] if no relationships are found. Do NOT include any text outside the JSON array.

Chat:
{text[:3000]}
"""
                    try:
                        triples_raw = run_traced_claude_task(graph_prompt, agent_name="Graph Extraction Agent", task_type="graph_extraction", file_id=file_metadata["file_id"], max_tokens=2000)
                        # Attempt to parse JSON
                        try:
                            triples_dicts = json.loads(triples_raw.strip())
                            if not isinstance(triples_dicts, list):
                                raise ValueError("Response is not a JSON array")
                        except json.JSONDecodeError:
                            # Fallback: Extract JSON array
                            match = re.search(r'\[\s*{.*?}\s*\]', triples_raw, re.DOTALL)
                            if not match:
                                raise ValueError("No valid JSON array found in response")
                            json_str = match.group()
                            triples_dicts = json.loads(json_str)

                        triples = [
                            {"source": t["source"], "relation": t["relation"], "target": t["target"]}
                            for t in triples_dicts
                            if all(k in t for k in ("source", "relation", "target"))
                        ]

                        # Store triples in MongoDB with retry
                        for attempt in range(3):
                            try:
                                graphs_col.insert_one({
                                    "file_id": file_metadata["file_id"],
                                    "triples": triples,
                                    "created_at": datetime.now().isoformat()
                                })
                                break
                            except (AutoReconnect, ServerSelectionTimeoutError) as e:
                                if attempt == 2:
                                    st.error(f"❌ MongoDB connection error during graph storage: {type(e).__name__} - {e}")
                                    raise
                                time.sleep(2)

                        st.subheader("🔗 Extracted Relationships")
                        st.json(triples_dicts)
                    except (AutoReconnect, ServerSelectionTimeoutError) as e:
                        st.error(f"❌ Failed to parse graph triples: {type(e).__name__} - {e}")
                        st.text_area("Claude Response (debug)", triples_raw, height=200)
                    except Exception as e:
                        st.error(f"❌ Failed to parse graph triples: {type(e).__name__} - {e}")
                        st.text_area("Claude Response (debug)", triples_raw, height=200)
                
                # Run analysis
                with st.spinner("📊 Running analysis..."):
                    analysis = run_analyst_agent(file_metadata["file_id"])
                    st.subheader("📈 Analysis Results")
                    st.markdown(analysis)
            st.success(f"✅ Processed {file_path}")
        except Exception as e:
            st.error(f"❌ Error processing {file_path}: {type(e).__name__} - {e}")

# Document upload section
with st.expander("📄 Upload and Process WhatsApp Chats", expanded=True):
    uploaded_files = st.file_uploader(
        "📁 Upload WhatsApp chat (.txt) or metadata (.json)", 
        type=["txt", "json"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        extractor = WhatsAppChatExtractor()
        for uploaded in uploaded_files:
            st.markdown(f"### 🔍 Processing: `{uploaded.name}`")
            file_type = "chat_log" if uploaded.name.endswith(".txt") else "metadata"

            # Step 1: Upload to MongoDB
            with st.spinner("📤 Uploading to MongoDB..."):
                metadata = save_file_to_mongo(uploaded, "GRP_DCB_VZM", file_type)
                st.success(f"✅ Uploaded to MongoDB: `{metadata['file_id']}`")

            # Step 2: Save metadata
            with st.spinner("🧠 Saving metadata..."):
                save_file_metadata(metadata["file_id"], metadata["file_name"], file_type, metadata["group_id"])
                st.success("✅ Metadata saved")

            # Step 3: Extract messages (for chat logs)
            if file_type == "chat_log":
                with st.spinner("📄 Extracting messages..."):
                    messages = extractor.extract_messages(metadata["file_id"])
                    st.success(f"✅ Extracted {len(messages)} messages")

                # Step 4: Generate embeddings
                with st.spinner("🔗 Generating embeddings with Amazon Titan..."):
                    for msg in messages:
                        embedding = get_embedding(msg["message"])
                        if not embedding:
                            st.warning(f"⚠️ Failed to generate embedding for message ID: {msg['_id']}")
                            continue
                        vector_metadata = {
                            "file_id": msg["file_id"],
                            "group_id": msg["group_id"],
                            "sender_name": msg["sender_name"],
                            "sender_role": msg["sender_role"],
                            "timestamp": msg["timestamp"]
                        }
                        add_to_vectorstore(msg["_id"], embedding, vector_metadata, msg["message"])
                    st.success("✅ Embeddings generated and stored in Chroma DB")

                # Step 5: Extract graph triples
                with st.spinner("🧠 Extracting relationships using Claude..."):
                    text = "\n".join([f"{msg['sender_name']} ({msg['sender_role']}): {msg['message']}" for msg in messages])
                    if not text.strip():
                        st.warning(f"⚠️ No text available for graph extraction in {uploaded.name}")
                        continue
                    graph_prompt = f"""
You are a graph extraction agent for WhatsApp chats. Extract structured relationships from the chat below, focusing on interactions between senders, tasks, and entities (e.g., suspects, locations). Return ONLY a valid JSON array of dictionaries with 'source', 'relation', and 'target'. Use sender names or roles from metadata, and identify relations like "ASSIGNED_TASK", "MENTIONED", "INVESTIGATING", or "ARRESTED".

Example:
[
  {{"source": "DSP Lovekik", "relation": "ASSIGNED_TASK", "target": "CI Yeshwanth"}},
  {{"source": "DSP Koushik", "relation": "INVESTIGATING", "target": "Raja"}}
]

Return an empty array [] if no relationships are found. Do NOT include any text outside the JSON array.

Chat:
{text[:3000]}
"""
                    try:
                        triples_raw = run_traced_claude_task(graph_prompt, agent_name="Graph Extraction Agent", task_type="graph_extraction", file_id=metadata["file_id"], max_tokens=2000)
                        # Attempt to parse JSON
                        try:
                            triples_dicts = json.loads(triples_raw.strip())
                            if not isinstance(triples_dicts, list):
                                raise ValueError("Response is not a JSON array")
                        except json.JSONDecodeError:
                            # Fallback: Extract JSON array
                            match = re.search(r'\[\s*{.*?}\s*\]', triples_raw, re.DOTALL)
                            if not match:
                                raise ValueError("No valid JSON array found in response")
                            json_str = match.group()
                            triples_dicts = json.loads(json_str)

                        triples = [
                            {"source": t["source"], "relation": t["relation"], "target": t["target"]}
                            for t in triples_dicts
                            if all(k in t for k in ("source", "relation", "target"))
                        ]

                        # Store triples in MongoDB with retry
                        for attempt in range(3):
                            try:
                                graphs_col.insert_one({
                                    "file_id": metadata["file_id"],
                                    "triples": triples,
                                    "created_at": datetime.now().isoformat()
                                })
                                break
                            except (AutoReconnect, ServerSelectionTimeoutError) as e:
                                if attempt == 2:
                                    st.error(f"❌ MongoDB connection error during graph storage: {type(e).__name__} - {e}")
                                    raise
                                time.sleep(2)

                        st.subheader("🔗 Extracted Relationships")
                        st.json(triples_dicts)
                    except (AutoReconnect, ServerSelectionTimeoutError) as e:
                        st.error(f"❌ Failed to parse graph triples: {type(e).__name__} - {e}")
                        st.text_area("Claude Response (debug)", triples_raw, height=200)
                    except Exception as e:
                        st.error(f"❌ Failed to parse graph triples: {type(e).__name__} - {e}")
                        st.text_area("Claude Response (debug)", triples_raw, height=200)

                # Step 6: Run analysis
                with st.spinner("📊 Running analysis..."):
                    analysis = run_analyst_agent(metadata["file_id"])
                    st.subheader("📈 Analysis Results")
                    st.markdown(analysis)

# Chat with data section
with st.expander("💬 Ask Questions from Chats", expanded=True):
    user_query = st.text_input("🔎 Enter your question (e.g., 'Summarize drug case discussions')")
    if user_query:
        with st.spinner("🧠 Thinking..."):
            response = answer_query(user_query)
            st.success("✅ Answer")
            st.markdown(response)

# Initialize fixed dataset
if st.button("Initialize Fixed Dataset"):
    with st.spinner("📂 Processing fixed dataset..."):
        initialize_dataset()