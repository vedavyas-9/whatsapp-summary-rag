from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb+srv://omkark:p67yMVWsmrtoO3f1@cluster0.ismx81v.mongodb.net/")
db = client["hackathon"]
metadata_col = db["metadata"]

def save_metadata(doc_name, doc_type, s3_path, intelligence_level=1):
    entry = {
        "filename": doc_name,
        "doc_type": doc_type,
        "intelligence_level": intelligence_level,
        "s3_path": s3_path,
        "timestamp": datetime.now(),
    }
    metadata_col.insert_one(entry)
    return entry
