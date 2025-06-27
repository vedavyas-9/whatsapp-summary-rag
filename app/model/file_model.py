from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class UploadedFileModel:
    filename: str
    file_id: Optional[str] = None
    extracted_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    graph_triples: Optional[List] = None
    group_id: Optional[str] = None
    sender_name: Optional[str] = None
    sender_role: Optional[str] = None
    timestamp: Optional[str] = None
    language: Optional[str] = None