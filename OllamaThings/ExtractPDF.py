import pdfplumber
import os

def extract_text_from_pdfs(folder="./docs"):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder, filename)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                texts.append(text)
    return texts
