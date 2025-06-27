import pdfplumber
import pandas as pd
from docx import Document
import io

def extract_pdf_text(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)

def extract_word_text(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_excel_text(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df.to_csv(index=False)
