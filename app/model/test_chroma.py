from vectorstore_model import add_to_vectorstore

# Sample test: Add a dummy document
add_to_vectorstore(
    doc_id="test-doc",
    embedding=[0.1] * 768,  # Ensure this matches your embedding model's vector size
    metadata={"type": "test"},
    document_text="This is a sample test document for ChromaDB."
)
