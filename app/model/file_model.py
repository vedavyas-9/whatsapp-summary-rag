from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class UploadedFileModel:
    filename: str
    bytes_content: bytes
    s3_path: Optional[str] = None
    extracted_text: Optional[str] = None
    embedding: Optional[list] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    graph_triples: Optional[list] = None
