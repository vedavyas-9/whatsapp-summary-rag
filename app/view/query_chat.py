import streamlit as st
from app.controller.chat_controller import answer_query  # Ensure this function exists

def main():
    st.title("🧠 Chat with Uploaded Police Documents")

    query_input = st.text_input("💬 Ask a question about the data you uploaded:")

    if query_input:
        with st.spinner("🔍 Searching and reasoning..."):
            response = answer_query(query_input)
            st.success("✅ Answer generated")
            st.markdown(response)
