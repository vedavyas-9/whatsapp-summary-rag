# remove.py
import os
import sys

# Add your project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import your app and vector store utils
from app.main import llm_utils

def reset_vector_store():
    llm_utils.vector_store.clear()
    print("🧹 Cleared llm_utils.vector_store.")
    print("✅ Current vector store keys:", list(llm_utils.vector_store.keys()))

if __name__ == "__main__":
    reset_vector_store()
